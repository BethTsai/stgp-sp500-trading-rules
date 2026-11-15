"""
MOOValidator：在驗證期 (Val) 評估 MOO Pareto 前緣的個體。

供兩處共用，確保分數定義一致：
  - gp_manager_moo.MOOGPManager：演化中的 Val 冠軍追蹤與 early stopping
  - moo_selection.py：事後從前緣挑選代表策略

冠軍分數使用單目標適應度 (單一數值)，而非 MOO 的目標向量：
    soo_score = 窗口超額分數的聚合 (fitness.aggregation) - 節點數 × length_penalty
做法是以 FitnessEvaluator 並暫時設定 objective_mode = 'single' 計算，結果與 SOO 的 Val 分數相同。
不同 scheme 的第二目標意義不同，改用單一分數後，各 scheme 與 SOO 的冠軍可在同一尺度上比較。

Val 窗口的切法與 GPManager 相同，num_samples 固定為 None (全窗口評估，結果可重現)。
"""
import copy
import logging

import numpy as np
from deap import gp

from src.data_provider import DataProvider
from src.evaluator import FitnessEvaluator
from src.primitives import create_primitive_set

# moo_validation 區塊的預設值；config 沒寫的鍵一律用這裡
MOO_VALIDATION_DEFAULTS = {
    'enabled': False,              # False -> MOOGPManager 行為等同 GPManager
    'eval_every': 1,               # 每幾代驗證一次 (1 = 每代；前緣較大時可調大以節省時間)
    'max_candidates': 60,          # 每次最多拿幾個前緣成員去 Val (null = 全部)。超過時沿前緣等距抽樣
    'champion_length_penalty': None,  # None = 沿用 fitness.length_penalty (與 SOO 冠軍分數同口徑)；也可明確給數字
    'selection': 'val_champion',   # 事後挑代表策略的方法，見 moo_selection.py
    'filter_degenerate': True,     # topsis / knee_point 是否先濾掉無效點
}


def get_mv_cfg(config: dict) -> dict:
    """讀取 config['moo_validation']，缺的鍵補預設值"""
    return {**MOO_VALIDATION_DEFAULTS, **(config.get('moo_validation') or {})}


def _single_cfg(config: dict, length_penalty=None) -> dict:
    """回傳一份「單目標口徑」的 config 副本 (只在記憶體；原 config 不動)。

    把 objective_mode 改成 'single'、soo_training_mode 改成 'oos'：
      - 'single': FitnessEvaluator.evaluate_tree 只回傳 (soo_score,)
      - 'oos'   : DataProvider 才會讓 mode='val' 可用 (DataProvider 看 objective_mode/soo_training_mode
                  決定要不要用 oos_split 日期)。這樣驗證集就不依賴 evolution.moo_selection_method 的值。
    """
    cfg = copy.deepcopy(config)
    cfg['evolution']['objective_mode'] = 'single'
    cfg['evolution']['soo_training_mode'] = 'oos'
    cfg['evolution']['weights'] = [1.0]
    if length_penalty is not None:
        cfg.setdefault('fitness', {})['length_penalty'] = float(length_penalty)
    return cfg


def _fit_windows(dp, config: dict) -> dict:
    """把 dp 的窗口長度調成『剛好切滿整個期間、且長度最接近訓練窗口』，回傳新的 config 副本 (與 SOO 的 Val 做法相同)。
    副本同時寫回 dp.config，讓 dp.get_rsfgp_windows() 用新的窗口長度。"""
    wb = config['rsfgp'].get('window_bars', 756)
    n_win = max(1, round(dp.total_length / wb))          # 四捨五入，使窗口長度最接近訓練窗口
    cfg = copy.deepcopy(config)
    cfg['rsfgp']['window_bars'] = dp.total_length // n_win
    cfg['rsfgp']['num_samples'] = None                   # 一律全窗口，結果才可重現
    dp.config = cfg
    return cfg


class MOOValidator:
    """對個體算「Val 冠軍分數」。有快取 (Val 是確定性的，同一棵樹不必重算)。"""

    def __init__(self, config: dict, champion_rule: str = 'val'):
        if champion_rule not in ('val', 'min_train_val'):
            raise ValueError(f"champion_rule 不支援: {champion_rule} (可用: 'val' / 'min_train_val')")
        self.config = config
        self.champion_rule = champion_rule
        mv = get_mv_cfg(config)
        self.mv = mv
        self.clip = float((config.get('fitness') or {}).get('score_clip', 10.0))

        # ---- Val: 單目標口徑 (冠軍分數) ----
        scfg = _single_cfg(config, mv['champion_length_penalty'])
        self.val_dp = DataProvider(config=scfg, mode='val')
        self.val_cfg = _fit_windows(self.val_dp, scfg)
        self.val_eval = FitnessEvaluator(data_provider=self.val_dp, config=self.val_cfg)
        n_win = self.val_dp.total_length // self.val_cfg['rsfgp']['window_bars']
        logging.info(f"🧪 [MOO Validation] Val {self.val_dp.total_length} 根 K 棒 -> {n_win} 個窗口 x "
                     f"{self.val_cfg['rsfgp']['window_bars']} 根 | 冠軍規則: {champion_rule}")

        # ---- Val: 多目標向量 (只拿來記錄，不影響冠軍) ----
        # MOOFitnessEvaluator 在 config 有 moo 區塊且 objective_mode='multi' 時才回傳向量
        self.val_vec_eval = None
        if config.get('moo'):
            from src.evaluator_moo import MOOFitnessEvaluator
            mcfg = copy.deepcopy(self.val_cfg)
            mcfg['evolution']['objective_mode'] = 'multi'
            mcfg['evolution']['weights'] = list(config['evolution'].get('weights', [1.0, 1.0]))
            self.val_vec_eval = MOOFitnessEvaluator(data_provider=self.val_dp, config=mcfg)

        # ---- Train: 單目標口徑 (只有 min_train_val 才需要；用到才建，省一份記憶體) ----
        self.train_dp = None
        self.train_eval = None
        if champion_rule == 'min_train_val':
            self.train_dp = DataProvider(config=scfg, mode='train')
            tcfg = copy.deepcopy(scfg)
            tcfg['rsfgp']['num_samples'] = None          # Train 分數也用全窗口，才是確定性的
            self.train_dp.config = tcfg
            self.train_eval = FitnessEvaluator(data_provider=self.train_dp, config=tcfg)

        # 編譯用的 pset：與 val_dp 的特徵順序對齊 (和 multi_run_manager 的 oos 挑選做法相同)
        self.pset = create_primitive_set(self.val_dp.feature_cols, config['features'].get('erc_range', [-100.0, 100.0]))

        self._cache = {}          # str(個體) -> dict(val=, train=, val_vec=)

    # ------------------------------------------------------------------
    # 單一個體的分數
    # ------------------------------------------------------------------
    def _soo_score(self, evaluator, ind) -> float:
        func = gp.compile(expr=ind, pset=self.pset)
        s = float(evaluator.evaluate_tree(func, ind)[0])
        return s if np.isfinite(s) else -float('inf')    # NaN/inf 一律當最差，不能讓它成為冠軍

    def scores(self, ind) -> dict:
        """回傳 {'val': Val soo 分數, 'train': Train soo 分數或 None, 'val_vec': Val 的 MOO 向量或 None}"""
        key = str(ind)
        if key not in self._cache:
            rec = {'val': self._soo_score(self.val_eval, ind), 'train': None, 'val_vec': None}
            if self.train_eval is not None:
                rec['train'] = self._soo_score(self.train_eval, ind)
            if self.val_vec_eval is not None:
                func = gp.compile(expr=ind, pset=self.pset)
                rec['val_vec'] = tuple(float(x) for x in self.val_vec_eval.evaluate_tree(func, ind))
            self._cache[key] = rec
        return self._cache[key]

    def champion_score(self, ind) -> float:
        """冠軍分數：'val' -> Val 分數；'min_train_val' -> min(Train, Val)，以降低選擇偏誤 (winner's curse)"""
        rec = self.scores(ind)
        if self.champion_rule == 'min_train_val':
            return min(rec['val'], rec['train'])
        return rec['val']

    # ------------------------------------------------------------------
    # 候選人篩選
    # ------------------------------------------------------------------
    def filter_valid(self, front):
        """去掉無效點。evaluator_moo 對無效個體給 (-score_clip, 最差值)，所以第一目標 <= -clip 就是無效。
        (一個有效個體不可能第一目標剛好 <= -clip，因為那表示所有窗口都被夾到下限。)"""
        return [i for i in front if i.fitness.values[0] > -self.clip + 1e-9]

    def pick_candidates(self, front):
        """前緣 -> 候選清單：濾無效 -> 去重 -> 超過 max_candidates 時沿第一目標等距抽樣 (保留兩端)。

        去重：同一個適應度向量 (四捨五入 6 位) 只保留節點數最少者。
        等距抽樣而不是取前 N 名：前緣的價值在於覆蓋整條取捨曲線，只取第一目標最高的 N 個會把另一端全丟掉。
        """
        valid = self.filter_valid(front)
        seen, uniq = set(), []
        for ind in sorted(valid, key=lambda i: (i.fitness.values[0], len(i))):
            k = tuple(round(v, 6) for v in ind.fitness.values)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(ind)
        # 去重時同向量留下的是「排序後第一個」；上面排序的第二鍵是節點數 (小的在前)，所以留下較短的樹
        n = self.mv['max_candidates']
        if n is not None and len(uniq) > int(n):
            idx = np.unique(np.linspace(0, len(uniq) - 1, int(n)).round().astype(int))
            uniq = [uniq[i] for i in idx]
        return uniq

    def best_of(self, candidates):
        """在候選人裡挑冠軍分數最高者；回傳 (個體, 冠軍分數)。候選為空回傳 (None, -inf)。同分取較短的樹。"""
        best, best_s = None, -float('inf')
        for ind in candidates:
            s = self.champion_score(ind)
            if s > best_s or (s == best_s and best is not None and len(ind) < len(best)):
                best, best_s = ind, s
        return best, best_s

    # ------------------------------------------------------------------
    # 以整段 Val 的年化 Sterling 挑選，只給 moo_selection 的 oos_sterling 用
    # ------------------------------------------------------------------
    def val_sterling(self, ind) -> float:
        dp = self.val_dp
        func = gp.compile(expr=ind, pset=self.pset)
        signals = func(*dp.full_features_list)
        if not isinstance(signals, np.ndarray):
            signals = np.full(dp.full_close.shape, signals, dtype=bool)
        eq, _ = self.val_eval._run_state_machine(signals, dp.full_close, dp.full_eval_price, dp.full_market_returns,
                                            active=dp.full_active)
        return float(self.val_eval._calculate_sterling_ratio(eq))
