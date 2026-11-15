"""
MOOGPManager：繼承 GPManager，替多目標 (MOO) 補上單目標已有的三項機制：

  1. Validation 評估與冠軍規則  (沿用 evolution.champion_rule：'val' / 'min_train_val')
  2. Early stopping             (沿用 evolution.early_stopping_patience / _min_delta / _warmup)
  3. 存活選擇去重與隨機新個體   (沿用 evolution.dedup_survival / dedup_by_fitness / immigrant_rate)

objective_mode = 'single'，或 moo_validation.enabled = false 時，所有方法直接呼叫父類別，行為等同 GPManager。
main.py 固定使用本類別。

啟用後在 checkpoint_dir 多存 (best_model_gen_N.pkl / best_model_final.pkl 仍是 Pareto 前緣)：
  moo_champion.pkl   Val 冠軍個體 (gzip pickle，dict：individual / val_score / train_score / gen / ...)
  moo_val_log.json   每次驗證的紀錄 (當代最佳、歷史最佳、耐受度)
"""
import gzip
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from deap import tools

from src.gp_manager import GPManager, worker_evaluate
from src.moo_validation import MOOValidator, get_mv_cfg


class MOOGPManager(GPManager):

    def __init__(self, config: dict, evaluator):
        super().__init__(config, evaluator)   # 建立 DEAP 環境、Pool、初始族群 (父類別對 multi 會 use_validation=False)

        evo = config.get('evolution', {})
        self.moo_mv = get_mv_cfg(config)
        # 啟用條件：多目標 + 開關為 true。其他情況全部走父類別
        self.use_moo_validation = (evo.get('objective_mode', 'single') == 'multi' and bool(self.moo_mv['enabled']))

        self.moo_champion = None          # 目前的 Val 冠軍 (個體的 clone)
        self.moo_champion_meta = {}       # 冠軍的分數、世代等資訊
        self.moo_val_log = []             # 每次驗證一筆
        self._moo_champion_dirty = False  # 冠軍剛更新、還沒寫檔
        self._last_val_gen = 0            # 上一次驗證的世代 (耐受度以世代數累計)
        self.validator = None

        if self.use_moo_validation:
            # 訓練期必須是 data.oos_split 的訓練期，Val 才會落在訓練期之後。
            # DataProvider 以 evolution.moo_selection_method == 'oos' 判斷是否使用 oos_split 日期，
            # 這裡直接擋下來，避免默默用 data.train_start / train_end 訓練。
            if evo.get('moo_selection_method', 'topsis') != 'oos' or 'oos_split' not in config['data']:
                self.pool.terminate()
                raise ValueError(
                    "moo_validation.enabled=true 需要訓練期使用 data.oos_split：請把 evolution.moo_selection_method 設成 'oos' "
                    "(它在 DataProvider 裡同時是『用 oos_split 日期』的開關；真正的挑選方式改由 moo_validation.selection 決定)，"
                    "並確認 data.oos_split 存在。")
            self.validator = MOOValidator(config, champion_rule=self.champion_rule)
            logging.info(f"🧪 [MOO] 驗證啟用 | 冠軍規則={self.champion_rule} | 每 {self.moo_mv['eval_every']} 代驗證 | "
                         f"耐受度={self.patience_limit} 代 | 候選上限={self.moo_mv['max_candidates']}")

    # ------------------------------------------------------------------
    # 演化一代：multi 模式加入去重與隨機新個體；其他情況交給父類別
    # ------------------------------------------------------------------
    def run_generation(self):
        # 僅在 multi 且 dedup_survival 開啟時使用本類別的版本，否則使用父類別
        if not (self.use_moo_validation and self.mode == 'multi' and self.dedup_survival):
            return super().run_generation()

        from src.gp_manager import custom_varOr
        cxpb = self.config['evolution'].get('cxpb', 0.9)
        mutpb = self.config['evolution'].get('mutpb', 0.1)

        # 前半段與父類別相同：同代固定種子、評估、產生子代
        self.current_gen += 1
        gen_seed = self.base_seed + (self.current_gen * 10000)

        invalid_pop = [ind for ind in self.population if not ind.fitness.valid]
        if invalid_pop:
            fits = self.pool.map(worker_evaluate, [(ind, gen_seed) for ind in invalid_pop])
            for ind, fit in zip(invalid_pop, fits):
                ind.fitness.values = fit

        offspring = custom_varOr(self.mode, self.population, self.toolbox, self.lambda_, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fits = self.pool.map(worker_evaluate, [(ind, gen_seed) for ind in invalid_ind])
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit

        # 後半段：(μ+λ) 合併 -> 去重 -> (補入隨機新個體) -> NSGA-II 選出 μ 個
        combined = self.population + offspring
        n_imm = int(round(self.immigrant_rate * self.mu))
        if n_imm > 0:
            combined = combined + self._make_immigrants(n_imm, gen_seed)

        unique = self._dedup_pool(combined)
        if len(unique) < self.mu:
            # 去重後不足 μ 個時補入隨機新個體 (與 SOO 相同)；被支配者由 NSGA-II 淘汰
            need = self.mu - len(unique)
            logging.info(f"   🧬 [MOO 去重] 不重複個體僅 {len(unique)} 個，補入 {need} 個隨機新個體")
            unique = unique + self._make_immigrants(need, gen_seed)

        # selNSGA2 同時會把 crowding_dist 寫回每個個體；custom_varOr 下一代需要它 (selTournamentDCD 的先決條件)
        self.population = self.toolbox.select(unique, self.mu)

        front = tools.sortNondominated(self.population, len(self.population), first_front_only=True)[0]
        n_vec = len({tuple(round(v, 6) for v in i.fitness.values) for i in front})
        logging.info(f"   📊 [MOO] 前緣 {len(front)} 個 (不同適應度向量 {n_vec}) | "
                     f"族群不重複樹 {len({self._key(i) for i in self.population})}/{self.mu}")

    def _dedup_pool(self, individuals):
        """去重：同一個『適應度向量』(dedup_by_fitness=true) 或同一個『結構鍵』只留一個 (留節點數較少者)。

        為什麼用向量而不是單一目標：MOO 的 fitness 是 tuple，兩個目標都相同才算「行為一樣的複製品」；
        只看第一目標會誤殺『第一目標相同、第二目標不同』的前緣點。
        """
        seen, out = set(), []
        for ind in sorted(individuals, key=lambda i: (len(i),)):     # 短的樹優先留下
            if self.dedup_fitness:
                key = tuple(round(v, 6) for v in ind.fitness.values)
            else:
                key = self._key(ind)
            if key in seen:
                continue
            seen.add(key)
            out.append(ind)
        return out

    # ------------------------------------------------------------------
    # 驗證 + Early Stopping (main.py 每代都會呼叫)
    # ------------------------------------------------------------------
    def check_early_stopping(self) -> tuple:
        if not self.use_moo_validation:
            return super().check_early_stopping()

        gens_since = self.current_gen - self._last_val_gen
        if gens_since < int(self.moo_mv['eval_every']):
            return False, self.best_val_score            # 還沒到驗證的世代
        self._last_val_gen = self.current_gen

        # 候選 = 第一前緣 (濾無效 -> 去重 -> 等距抽樣到 max_candidates)
        front = tools.sortNondominated(self.population, len(self.population), first_front_only=True)[0]
        cands = self.validator.pick_candidates(front)
        cur_ind, cur_best = self.validator.best_of(cands)
        if cur_ind is None:
            # 前緣全為無效點 (常見於演化初期)：不更新冠軍，也不累計耐受度
            logging.info("   ⏳ [MOO Val] 前緣沒有有效個體，略過本次驗證")
            return False, self.best_val_score

        improved = cur_best > self.best_val_score + self.es_min_delta
        if cur_best > self.best_val_score:
            # 分數高於歷史最佳即更新冠軍；min_delta 只決定耐受度是否歸零
            rec = self.validator.scores(cur_ind)
            self.best_val_score = cur_best
            self.best_val_ind = self.toolbox.clone(cur_ind)
            self.moo_champion = self.best_val_ind
            self.moo_champion_meta = {
                'gen': self.current_gen,
                'rule': self.champion_rule,
                'champion_score': cur_best,
                'val_score': rec['val'],
                'train_score': rec['train'],
                'val_vector': rec['val_vec'],
                'train_vector': tuple(cur_ind.fitness.values),
                'tree': str(cur_ind),
            }
            self._moo_champion_dirty = True

        if improved:
            self.patience_counter = 0
            msg = f"🏆 [MOO Val 破紀錄] 新高: {self.best_val_score:.4f} (耐受度歸零)"
        elif self.current_gen <= self.es_warmup:
            msg = (f"⏳ [MOO Val 暖機中 {self.current_gen}/{self.es_warmup}] 當代最佳: {cur_best:.4f} | "
                   f"歷史高點: {self.best_val_score:.4f} (不累計耐受度)")
        else:
            self.patience_counter += gens_since             # 以『代數』累計，eval_every=5 時一次加 5
            msg = (f"⚠️ [MOO Val 未突破] 當代最佳: {cur_best:.4f} | 歷史高點: {self.best_val_score:.4f} | "
                   f"耐受度: {self.patience_counter}/{self.patience_limit}")
        logging.info(f"   {msg} | 候選 {len(cands)}/{len(front)}")

        self.moo_val_log.append({
            'gen': self.current_gen, 'front_size': len(front), 'n_candidates': len(cands),
            'cur_best': cur_best, 'best_so_far': self.best_val_score,
            'patience': self.patience_counter, 'improved': bool(improved),
        })
        return (self.patience_counter >= self.patience_limit), self.best_val_score

    # ------------------------------------------------------------------
    # 存檔
    # ------------------------------------------------------------------
    def save_best_model(self, filepath):
        super().save_best_model(filepath)        # 仍存 Pareto 前緣 (multi_run_manager 繪製前緣與計算 hypervolume 使用)
        if self.use_moo_validation and self._moo_champion_dirty and self.moo_champion is not None:
            self._write_champion(Path(filepath).parent)

    def _write_champion(self, out_dir: Path):
        with gzip.open(out_dir / 'moo_champion.pkl', 'wb') as f:
            pickle.dump({'individual': self.moo_champion, **self.moo_champion_meta}, f)
        self._moo_champion_dirty = False
        logging.info(f"   💾 已儲存 MOO Val 冠軍 (第 {self.moo_champion_meta['gen']} 代產生，"
                     f"冠軍分數 {self.moo_champion_meta['champion_score']:.4f})")

    def save_history(self, filepath):
        if not self.use_moo_validation:
            return super().save_history(filepath)

        out_dir = Path(filepath).parent
        if self._moo_champion_dirty and self.moo_champion is not None:
            self._write_champion(out_dir)
        with open(out_dir / 'moo_val_log.json', 'w', encoding='utf-8') as f:
            json.dump({'champion': {k: v for k, v in self.moo_champion_meta.items()},
                       'log': self.moo_val_log}, f, ensure_ascii=False, indent=1, default=float)

        # 父類別只保留最後一代族群的祖先；冠軍可能不在最後一代，因此暫時加入 population 以保留其族譜，存檔後還原。
        saved_pop = self.population
        if self.moo_champion is not None and hasattr(self.moo_champion, 'history_index'):
            self.population = list(saved_pop) + [self.moo_champion]
        try:
            super().save_history(filepath)
        finally:
            self.population = saved_pop
