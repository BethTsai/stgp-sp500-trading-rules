"""
多目標適應度評估器 (MOO Fitness Evaluator)

繼承 src/evaluator.py 的 FitnessEvaluator，只覆寫 evaluate_tree()：
  - objective_mode = 'single'，或 config 沒有 `moo:` 區塊 -> 直接呼叫父類別
  - objective_mode = 'multi' 且有 `moo:` 區塊 -> 依 moo.scheme 計算兩個「越大越好」的目標

設計原則：
  1. 兩個目標都是「最大化」，要最小化的量 (樹大小、換手、標準差、回撤) 一律取負號，
     所以 config 的 evolution.weights 固定寫 [1.0, 1.0]。
  2. 第二目標的懲罰不會塞進第一目標。
  3. 無效策略 (近似全持有 / 全空手 / 低換手) 的兩個目標都給最差值，避免前緣被無效點佔據。
  4. 窗口分數的算法 (狀態機、基準、clip) 全部沿用父類別，與 SOO 同口徑。

支援的 scheme (大小寫不拘；括號內是別名)：
  complexity       (A)  目標1 = 窗口超額分數聚合,  目標2 = -樹節點數
  robust           (B)  目標1 = 窗口超額分數平均,  目標2 = -窗口分數標準差 (或 最差窗口分數)
  turnover         (C)  目標1 = 窗口超額分數聚合,  目標2 = -平均換手量
  sterling_split   (D)  目標1 = 窗口超額報酬聚合,  目標2 = 窗口回撤改善聚合 (Sterling 分子 / 分母拆開)
  skewness         (E)  目標1 = 窗口超額分數聚合,  目標2 = 報酬偏度
  payoff           (F)  目標1 = 窗口超額分數聚合,  目標2 = 盈虧比

"""
import numpy as np

from src.evaluator import FitnessEvaluator

# 別名 -> 正式名稱
SCHEME_ALIASES = {
    'a': 'complexity', 'complexity': 'complexity',
    'b': 'robust', 'robust': 'robust',
    'c': 'turnover', 'turnover': 'turnover',
    'd': 'sterling_split', 'sterling_split': 'sterling_split',
    'e': 'skewness', 'skewness': 'skewness',
    'f': 'payoff', 'payoff': 'payoff',
}

MOO_DEFAULTS = {
    'scheme': 'robust',
    'robust_second': 'std',           # scheme=robust：第二目標用 "std" (-標準差) 或 "min" (最差窗口分數)
    'degenerate_window_frac': 1.0,    # 無效窗口占比 >= 此值，整個個體兩目標都給最差值
    'length_penalty': 0.0,            # 每個樹節點從「第一目標」扣的分數 (scheme=complexity 一律不扣，因為樹大小已是目標2)
    'turnover_ref': 300.0,            # scheme=turnover：換手量的上限/最差值 (單一窗口累計換手；窗口 756 根)
    'skew_cap': 3.0,                  # scheme=skewness：偏度夾限 [-cap, +cap]
    'payoff_cap': 5.0,                # scheme=payoff：盈虧比夾限 [0, cap]
}

# 目標軸標籤 (報表與圖表使用)
SCHEME_LABELS = {
    'complexity':     ('超額窗口分數', '-樹節點數'),
    'robust':         ('超額窗口分數平均', '-窗口分數標準差 / 最差窗口分數'),
    'turnover':       ('超額窗口分數', '-平均換手量'),
    'sterling_split': ('超額窗口報酬 (Sterling 分子)', '回撤改善 (Sterling 分母, 取負)'),
    'skewness':       ('超額窗口分數', '報酬偏度'),
    'payoff':         ('超額窗口分數', '盈虧比'),
}


def resolve_scheme(config: dict) -> str:
    """從 config['moo']['scheme'] 取得正式 scheme 名稱；未設定 moo 區塊時回傳 None。"""
    moo = config.get('moo')
    if not moo:
        return None
    name = str(moo.get('scheme', MOO_DEFAULTS['scheme'])).strip().lower()
    if name not in SCHEME_ALIASES:
        raise ValueError(f"moo.scheme 不支援: {moo.get('scheme')} (可用: A/B/C/D/E/F 或 {sorted(set(SCHEME_ALIASES.values()))})")
    return SCHEME_ALIASES[name]


def objective_labels(config: dict) -> tuple:
    """回傳 (目標1 名稱, 目標2 名稱)，給報表 / 圖表軸標籤使用。沒有 moo 區塊時第二目標用 evolution.multi_target。"""
    scheme = resolve_scheme(config)
    if scheme is None:
        target = config.get('evolution', {}).get('multi_target', 'skewness')
        return ('超額窗口分數', target)
    return SCHEME_LABELS[scheme]


class MOOFitnessEvaluator(FitnessEvaluator):

    def __init__(self, data_provider, config: dict):
        super().__init__(data_provider, config)
        self._baseline_rm_cache = {}   # 窗口 -> (等權基準報酬, 等權基準 MDD)，只有 sterling_split 會用到
        resolve_scheme(config)         # 初始化時即檢查 scheme 名稱

    # ------------------------------------------------------------------
    # 設定
    # ------------------------------------------------------------------
    def _moo_cfg(self) -> dict:
        """每次呼叫時才讀 config，因為 GPManager 會在建好驗證用 evaluator 後替換它的 config"""
        cfg = {**MOO_DEFAULTS, **(self.config.get('moo') or {})}
        cfg['scheme'] = resolve_scheme(self.config)
        if cfg['robust_second'] not in ('std', 'min'):
            raise ValueError(f"moo.robust_second 只能是 'std' 或 'min'，收到: {cfg['robust_second']}")
        return cfg

    def _size_bound(self) -> int:
        """二元運算子的語法樹在 max_depth 下的最大節點數 = 2^(depth+1) - 1 (所有算子都是二元，見 primitives.py)"""
        depth = int(self.config['evolution'].get('max_depth', 6))
        return 2 ** (depth + 1) - 1

    def _worst_vector(self, scheme: str, cfg: dict, size: int) -> tuple:
        """無效個體的兩個目標值。第一目標一律 -score_clip；第二目標取各 scheme 的最差可能值。"""
        clip = self.fit_cfg['score_clip']
        if scheme == 'complexity':
            return (-clip, -float(max(self._size_bound(), size)))
        if scheme == 'robust':
            return (-clip, -clip)                 # 窗口分數 ∈ [-clip, clip]，所以 std 與 min 的最差值都是 -clip
        if scheme == 'turnover':
            return (-clip, -float(cfg['turnover_ref']))
        if scheme == 'sterling_split':
            return (-clip, -1.0)                  # 回撤改善 ∈ [-1, 1]
        if scheme == 'skewness':
            return (-clip, -float(cfg['skew_cap']))
        return (-clip, 0.0)                       # payoff：盈虧比最差 = 0

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------
    def _ret_mdd(self, equity_curve: np.ndarray):
        """窗口報酬與最大回撤 (拆開 score_window 的分子 / 分母)。淨值異常時回傳 None。
        算法與父類別 score_window 完全一致 (return_mode、MDD 公式)，只是不相除。"""
        if len(equity_curve) < 2 or not np.all(np.isfinite(equity_curve)):
            return None
        final_eq = max(float(equity_curve[-1]), 1e-9)
        ret = np.log(final_eq) if self.fit_cfg['return_mode'] == 'log' else final_eq - 1.0
        running_max = np.maximum.accumulate(equity_curve)
        mdd = float(np.max((running_max - equity_curve) / running_max))
        return float(ret), mdd

    def _baseline_ret_mdd(self, w: dict) -> tuple:
        """同窗口等權基準 (全部 True) 的 (報酬, MDD)。快取鍵的做法與父類別 _baseline_score 相同。"""
        key = (w['close'].__array_interface__['data'][0], w['close'].shape)
        if key not in self._baseline_rm_cache:
            eq, _ = self._run_state_machine(
                np.ones(w['close'].shape, dtype=bool), w['close'], w['eval_price'], w['market_returns'],
                active=w.get('active'))
            rm = self._ret_mdd(eq)
            self._baseline_rm_cache[key] = rm if rm is not None else (0.0, 0.0)
        return self._baseline_rm_cache[key]

    @staticmethod
    def _finite(x: float, fallback: float) -> float:
        """NaN / inf 一律換成最差值，避免影響 DEAP 的非支配排序"""
        return float(x) if np.isfinite(x) else float(fallback)

    # ------------------------------------------------------------------
    # DEAP 評估入口
    # ------------------------------------------------------------------
    def evaluate_tree(self, trading_func, individual) -> tuple:
        mode = self.config['evolution'].get('objective_mode', 'single')
        if mode != 'multi' or not self.config.get('moo'):
            return super().evaluate_tree(trading_func, individual)

        cfg = self._moo_cfg()
        scheme = cfg['scheme']
        clip = self.fit_cfg['score_clip']
        relative = self.fit_cfg['relative_to_baseline']
        size = len(individual)                      # DEAP 的 PrimitiveTree 長度 = 節點數
        worst = self._worst_vector(scheme, cfg, size)

        windows = self.dp.get_rsfgp_windows()
        if not windows:
            return worst

        # 每個窗口一筆；無效窗口填入最差值，使各串列長度一致
        scores, ret_ex, dd_gain, skews, payoffs, turnovers = [], [], [], [], [], []
        n_degenerate = 0

        for w in windows:
            close_prices = w['close']
            signals = trading_func(*w['feature_array'])
            # GP 可能產生常數訊號 (例如永遠 True)，廣播成 (T, N)
            if np.isscalar(signals) or signals.shape != close_prices.shape:
                signals = np.broadcast_to(signals, close_prices.shape)

            active = w.get('active')
            equity_curve, turnover = self._run_state_machine(
                signals, close_prices, w['eval_price'], w['market_returns'], active=active)
            turnovers.append(turnover)

            # ---- 無效窗口：近似 B&H / 低換手 / 幾乎永遠持有或空手 ----
            if turnover < self.fit_cfg['min_turnover'] or self._is_degenerate_exposure(signals, active):
                n_degenerate += 1
                scores.append(-clip)
                ret_ex.append(-clip)
                dd_gain.append(-1.0)
                skews.append(-cfg['skew_cap'])
                payoffs.append(0.0)
                continue

            # ---- 窗口分數 (超額分數，與 SOO 同口徑) ----
            score = self.score_window(equity_curve)
            if relative:
                score = float(np.clip(score - self._baseline_score(w), -clip, clip))
            scores.append(score)

            # ---- 各 scheme 需要的額外指標 (只計算該 scheme 用到的) ----
            if scheme == 'sterling_split':
                rm = self._ret_mdd(equity_curve)
                if rm is None:
                    ret_ex.append(-clip)
                    dd_gain.append(-1.0)
                else:
                    ret, mdd = rm
                    if relative:
                        b_ret, b_mdd = self._baseline_ret_mdd(w)
                        ret_ex.append(float(np.clip(ret - b_ret, -clip, clip)))   # 相對基準的超額報酬
                        dd_gain.append(float(np.clip(b_mdd - mdd, -1.0, 1.0)))    # 相對基準的回撤改善 (正值較佳)
                    else:
                        ret_ex.append(float(np.clip(ret, -clip, clip)))
                        dd_gain.append(float(np.clip(-mdd, -1.0, 1.0)))
            elif scheme in ('skewness', 'payoff'):
                returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
                if scheme == 'skewness':
                    s = self._calculate_skewness(returns)
                    skews.append(float(np.clip(s, -cfg['skew_cap'], cfg['skew_cap'])))
                else:
                    p = self._calculate_payoff(returns)
                    payoffs.append(float(np.clip(p, 0.0, cfg['payoff_cap'])))

        # ---- 整個個體無效：兩個目標都給最差值 ----
        if n_degenerate / len(windows) >= cfg['degenerate_window_frac'] - 1e-12:
            return worst

        # 長度懲罰：只加在第一目標且預設為 0；scheme=complexity 時樹大小已是目標2，避免重複懲罰
        length_pen = 0.0 if scheme == 'complexity' else size * cfg['length_penalty']

        if scheme == 'robust':
            s = np.asarray(scores, dtype=float)
            obj1 = float(np.mean(s)) - length_pen            # 第一目標只看平均；波動留給第二目標
            obj2 = -float(np.std(s)) if cfg['robust_second'] == 'std' else float(np.min(s))
        elif scheme == 'complexity':
            obj1 = self._aggregate(scores)
            obj2 = -float(size)
        elif scheme == 'turnover':
            obj1 = self._aggregate(scores) - length_pen
            obj2 = -float(min(np.mean(turnovers), cfg['turnover_ref']))
        elif scheme == 'sterling_split':
            obj1 = self._aggregate(ret_ex) - length_pen
            obj2 = self._aggregate(dd_gain)
        elif scheme == 'skewness':
            obj1 = self._aggregate(scores) - length_pen
            obj2 = float(np.mean(skews))
        else:  # payoff
            obj1 = self._aggregate(scores) - length_pen
            obj2 = float(np.mean(payoffs))

        return (self._finite(obj1, worst[0]), self._finite(obj2, worst[1]))
