"""
FitnessEvaluator：以 NumPy 向量化交易狀態機回測 GP 規則 (含交易成本)，
並依 config.yaml 的 fitness 區塊計算適應度。

適應度 (單目標)：
  對每個 RSFGP 窗口：
    1. 跑狀態機 -> equity_curve, turnover
    2. 無效策略 -> 窗口分數 = -score_clip：
         (a) turnover < min_turnover                         (近似 Buy&Hold / 幾乎不交易)
         (b) 平均持倉比例 > max_long_ratio 或 < min_long_ratio  (幾乎永遠持有 / 永遠空手；
             fee_mode='rebalanced' 時全 True 的換手量會超過 min_turnover，所以另外用持倉比例判斷)
    3. 否則 window_score = clip(窗口報酬 / (MDD + 0.1), -score_clip, +score_clip)
  (選用) fitness.relative_to_baseline=true 時，窗口分數改為「策略分數 − 同窗口基準分數」(超額表現)，
         fitness.baseline 決定基準：
           "bh"      = 真正 Buy&Hold (窗口開頭等權買入、不再平衡，同 src/benchmarks.py 的 true_bh)
           "ew_same" = 全部 True 訊號，經同一個狀態機與計費方式
  聚合各窗口分數 (預設 mean - std_k * std；mean_downside = mean - std_k * 下方標準差)，再扣長度懲罰 / 換手懲罰。
"""
import numpy as np

FITNESS_DEFAULTS = {
    'return_mode': 'simple',      # "simple": 窗口總報酬(不年化) / "log": log(期末淨值)
    'aggregation': 'mean_std',    # "mean" | "median" | "mean_std" | "mean_downside" | "min" | "quantile"
    'std_k': 0.5,                 # mean_std：平均 - std_k * 標準差；mean_downside：平均 - std_k * 下方標準差
    'downside_target': 'mean',    # mean_downside 的門檻 τ："mean" (各窗口分數平均) 或數字 (例如 0)
    'quantile': 0.25,             # aggregation="quantile" 時使用
    'score_clip': 10.0,           # 單一窗口分數限制在 [-score_clip, +score_clip]
    'min_turnover': 4.0,          # 窗口內累計換手量下限，低於就判定為近似 B&H
    'relative_to_baseline': False,  # True: 窗口分數 = 策略 − 基準 (超額表現)，基準自己 = 0
    'baseline': 'ew_same',        # relative_to_baseline 的基準："bh" (真正 Buy&Hold) | "ew_same" (全 True + 同狀態機)
    'max_long_ratio': 0.95,       # 平均持倉比例上限 (None = 不檢查)
    'min_long_ratio': 0.05,       # 平均持倉比例下限 (None = 不檢查)
    'length_penalty': 0.002,      # 每個樹節點扣的分數
    'turnover_penalty': 0.0,      # 每單位平均換手額外扣分
}


class FitnessEvaluator:
    def __init__(self, data_provider, config: dict):
        self.dp = data_provider
        self.config = config
        self.fee = config['trading'].get('transaction_fee', 0.003)
        self.fit_cfg = {**FITNESS_DEFAULTS, **(config.get('fitness') or {})}
        if self.fit_cfg['aggregation'] not in ('mean', 'median', 'mean_std', 'mean_downside', 'min', 'quantile'):
            raise ValueError(f"fitness.aggregation 不支援: {self.fit_cfg['aggregation']}")
        if self.fit_cfg['return_mode'] not in ('simple', 'log'):
            raise ValueError(f"fitness.return_mode 不支援: {self.fit_cfg['return_mode']}")
        if self.fit_cfg['baseline'] not in ('bh', 'ew_same'):
            raise ValueError(f"fitness.baseline 不支援: {self.fit_cfg['baseline']}")
        self._baseline_cache = {}   # 窗口 -> 基準分數 (窗口資料固定，算一次就好)
        self._growth_cache = {}     # 窗口 -> 各股累積漲幅 (drift 模式用，同樣只算一次)

    # ------------------------------------------------------------------
    # 適應度
    # ------------------------------------------------------------------
    def score_window(self, equity_curve: np.ndarray) -> float:
        """單一窗口分數 = 報酬 / (MDD + 0.1)，不年化，並限制在 [-score_clip, score_clip]"""
        clip = self.fit_cfg['score_clip']
        if len(equity_curve) < 2 or not np.all(np.isfinite(equity_curve)):
            return -clip
        final_eq = max(float(equity_curve[-1]), 1e-9)   # 避免淨值 <= 0 時 log 無定義
        ret = np.log(final_eq) if self.fit_cfg['return_mode'] == 'log' else final_eq - 1.0
        running_max = np.maximum.accumulate(equity_curve)
        mdd = float(np.max((running_max - equity_curve) / running_max))
        return float(np.clip(ret / (mdd + 0.1), -clip, clip))

    def _baseline_score(self, w: dict) -> float:
        """同窗口的基準分數，經同一個 score_window：
          baseline="bh"      真正 Buy&Hold (與報表的 true_bh 同一個函數)
          baseline="ew_same" 全部 True 訊號，經同一個狀態機 / 手續費
        窗口是 DataProvider 底層矩陣的切片，用 (記憶體位址, 長度) 當快取鍵。"""
        key = (w['close'].__array_interface__['data'][0], w['close'].shape)
        if key not in self._baseline_cache:
            active = w.get('active')
            if self.fit_cfg['baseline'] == 'bh':
                # 函數內 import，避免與 benchmarks 循環 import；股票池會異動時改用 point-in-time 版 B&H (與報表相同)
                from src.benchmarks import true_buy_and_hold_curve, pit_buy_and_hold_curve
                if active is None:
                    eq, _ = true_buy_and_hold_curve(w['market_returns'], self.fee)
                else:
                    eq, _ = pit_buy_and_hold_curve(w['market_returns'], active, self.fee, reinvest=True)
            else:
                eq, _ = self._run_state_machine(
                    np.ones(w['close'].shape, dtype=bool), w['close'], w['eval_price'], w['market_returns'],
                    active=active)
            self._baseline_cache[key] = self.score_window(eq)
        return self._baseline_cache[key]

    def _aggregate(self, scores: list) -> float:
        if not scores:
            return -self.fit_cfg['score_clip']
        s = np.asarray(scores, dtype=float)
        agg = self.fit_cfg['aggregation']
        if agg == 'mean':
            return float(np.mean(s))
        if agg == 'median':
            return float(np.median(s))
        if agg == 'min':
            return float(np.min(s))
        if agg == 'quantile':
            return float(np.quantile(s, self.fit_cfg['quantile']))
        if agg == 'mean_downside':
            # 下方標準差：只計入低於門檻 τ 的窗口，高於門檻的窗口不受懲罰
            target = self.fit_cfg['downside_target']
            tau = float(np.mean(s)) if target == 'mean' else float(target)
            downside = np.sqrt(np.mean(np.minimum(s - tau, 0.0) ** 2))
            return float(np.mean(s) - self.fit_cfg['std_k'] * downside)
        return float(np.mean(s) - self.fit_cfg['std_k'] * np.std(s))   # mean_std

    def _is_degenerate_exposure(self, signals: np.ndarray, active: np.ndarray = None) -> bool:
        """平均持倉比例是否過高 (幾乎永遠持有) 或過低 (幾乎永遠空手)。t 期持倉來自 t-1 期訊號，所以看 signals[:-1]。
        active 不為 None 時，只看池內股票：比例 = 池內被選中的 (股票, K 棒) 數 / 池內 (股票, K 棒) 總數。"""
        if len(signals) < 2:
            return True
        if active is None:
            long_ratio = float(np.mean(signals[:-1]))
        else:
            pool = np.asarray(active, dtype=bool)[:-1]
            n_pool = int(pool.sum())
            if n_pool == 0:
                return True
            long_ratio = float(np.sum(np.asarray(signals[:-1], dtype=bool) & pool)) / n_pool
        hi, lo = self.fit_cfg['max_long_ratio'], self.fit_cfg['min_long_ratio']
        return (hi is not None and long_ratio > hi) or (lo is not None and long_ratio < lo)

    def evaluate_tree(self, trading_func, individual) -> tuple:
        """
        DEAP 評估入口。依 DEAP 規範回傳 tuple：
          單目標 (score,)；多目標 (score, payoff) 或 (score, skewness)
        """
        mode = self.config['evolution'].get('objective_mode', 'single')
        target = self.config['evolution'].get('multi_target', 'skewness')
        clip = self.fit_cfg['score_clip']
        relative = self.fit_cfg['relative_to_baseline']
        windows = self.dp.get_rsfgp_windows()

        score_list, payoff_list, skew_list, turnover_list = [], [], [], []

        for w in windows:
            close_prices = w['close']
            signals = trading_func(*w['feature_array'])
            # GP 可能產生常數訊號 (例如永遠 True)，廣播成 (T, N)
            if np.isscalar(signals) or signals.shape != close_prices.shape:
                signals = np.broadcast_to(signals, close_prices.shape)

            active = w.get('active')
            equity_curve, turnover = self._run_state_machine(
                signals, close_prices, w['eval_price'], w['market_returns'], active=active)
            turnover_list.append(turnover)

            if turnover < self.fit_cfg['min_turnover'] or self._is_degenerate_exposure(signals, active):
                score_list.append(-clip)
                payoff_list.append(0.0)
                skew_list.append(-10.0)
                continue

            score = self.score_window(equity_curve)
            if relative:
                score = float(np.clip(score - self._baseline_score(w), -clip, clip))
            score_list.append(score)

            if mode != 'single':
                returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
                payoff_list.append(self._calculate_payoff(returns))
                skew_list.append(self._calculate_skewness(returns))

        mean_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0
        adjusted = (self._aggregate(score_list)
                    - len(individual) * self.fit_cfg['length_penalty']
                    - mean_turnover * self.fit_cfg['turnover_penalty'])

        if mode == 'single':
            return (adjusted,)

        # ---- 多目標 ----
        SKEW_CAP = 3.0                 # 偏度上限
        SKEW_PENALTY_THRESHOLD = 2.0   # 偏度超過此值開始扣分
        PENALTY_RATE = 0.1             # 偏度每超過門檻 1.0，扣除的分數

        raw_skewness = np.mean(skew_list)
        final_score = adjusted
        if raw_skewness > SKEW_PENALTY_THRESHOLD:
            final_score = adjusted - (raw_skewness - SKEW_PENALTY_THRESHOLD) * PENALTY_RATE
        final_skewness = max(-3.0, min(raw_skewness, SKEW_CAP))

        if target == 'payoff':
            return (adjusted, np.mean(payoff_list))
        elif target == 'skewness':
            return (final_score, final_skewness)
        return (adjusted, np.mean(skew_list))

    # ------------------------------------------------------------------
    # 狀態機與績效指標 (報表 / 回測也會用到)
    # ------------------------------------------------------------------
    def _run_state_machine(
        self,
        signals: np.ndarray,
        close_prices: np.ndarray,
        eval_prices: np.ndarray,
        market_returns: np.ndarray,
        active: np.ndarray = None
    ) -> tuple:
        """
        向量化交易狀態機：t-1 期訊號決定 t 期持倉，扣除手續費後回傳 (淨值曲線, 總換手量)。

        active (T, N) bool：point-in-time 股票池 (None = 全部一直在池內)。
          t 期持有股票 i  <=>  signals[t-1,i] 且 active[t-1,i] 且 active[t,i] 且 active[t+1,i]
          -> 加入池子的第一根 s：最早在 s+1 買進；最後一根在池內的 e：在 e 的 eval_price 賣出 (手續費由換手自動計入)。
        """
        executed_signals = np.zeros_like(signals, dtype=float)
        executed_signals[1:] = signals[:-1]
        executed_signals[0] = 0.0  # 第 0 根沒有前一根訊號，空手

        if active is not None:
            a = np.asarray(active, dtype=bool)
            in_pool = a.copy()
            in_pool[1:] &= a[:-1]    # active[t-1]：訊號形成時已在池內
            in_pool[:-1] &= a[1:]    # active[t+1]：持有到下一根時仍在池內 (最後一根在池內的 K 棒 e 因此在 e 賣出)
            executed_signals *= in_pool

        executed_signals[-1, :] = 0.0  # 期末強制平倉

        fee_mode = self.config.get('trading', {}).get('fee_mode', 'signal_driven')
        if fee_mode == 'drift':
            return self._run_drift(executed_signals, market_returns)

        allocation_mode = self.config.get('trading', {}).get('allocation_mode', 'dynamic')

        if allocation_mode == 'fixed':
            num_assets = executed_signals.shape[1]
            weights = executed_signals / num_assets
        else:
            # dynamic：持有的股票等權
            active_counts = np.sum(executed_signals, axis=1, keepdims=True)
            weights = np.divide(
                executed_signals,
                active_counts,
                out=np.zeros_like(executed_signals),
                where=active_counts != 0
            )

        portfolio_returns = np.zeros(close_prices.shape[0])

        # 時間對齊：weights[t] 在 t 以 eval_price 買入、持有到 t+1 的 eval_price，
        # 這段報酬就是 market_returns[t+1]，所以 weights[:-1] 對應 market_returns[1:]
        portfolio_returns[1:] = np.sum(weights[:-1, :] * market_returns[1:, :], axis=1)

        turnover_per_bar = np.zeros(close_prices.shape[0])

        # 價格變動後的實際權重 = 前期權重 × (1 + 個股報酬) / (1 + 組合報酬)
        drifted_weights = np.zeros_like(weights[:-1, :])
        drifted_weights = weights[:-1, :] * (1.0 + market_returns[1:, :]) / (1.0 + portfolio_returns[1:, None] + 1e-9)

        if fee_mode == 'rebalanced':
            # 每根 K 棒都調回目標權重並收費
            turnover_per_bar[1:] = np.sum(np.abs(weights[1:, :] - drifted_weights), axis=1)

        else: # 'signal_driven'
            # 每根都調回目標權重，但只有訊號改變的那根收費
            signal_changed = np.any(executed_signals[1:] != executed_signals[:-1], axis=1)
            turnover_per_bar[1:] = np.sum(np.abs(weights[1:, :] - drifted_weights), axis=1) * signal_changed

        turnover_per_bar[0] = np.sum(np.abs(weights[0, :]))  # 第 0 根建倉

        portfolio_returns -= (turnover_per_bar * self.fee)

        equity_curve = np.cumprod(1.0 + portfolio_returns)
        total_turnover = np.sum(turnover_per_bar)

        return equity_curve, total_turnover

    def _run_drift(self, executed_signals: np.ndarray, market_returns: np.ndarray) -> tuple:
        """
        fee_mode='drift'：只在持股組合改變的那根 K 棒調回目標權重 (並收費)，
        其餘時間不交易，權重隨價格自然飄移 (全部 True = 真正 Buy&Hold)。
        時間對齊、手續費與回傳值與通用狀態機相同，但只在調倉點計算權重與換手，計算量較低。

        做法：把時間切成「段」，每段從一個調倉點 s 開始。
          growth[t] = 各股從頭到 t 的累積漲幅 (每個窗口只算一次並快取)
          該段持股在 t 的價值 = 目標權重[s] × growth[t] / growth[s]，另有現金 1 - Σ目標權重[s]
          t 期報酬 = (t-1 所屬那段持股在 t 的總值) / (同一批持股在 t-1 的總值) - 1
        """
        T, N = executed_signals.shape
        # 快取同時保留 market_returns 本身：陣列不會被回收，記憶體位址就不會被別的陣列重用
        key = (market_returns.__array_interface__['data'][0], market_returns.shape)
        cached = self._growth_cache.get(key)
        if cached is None:
            mr = np.nan_to_num(market_returns, nan=0.0, posinf=0.0, neginf=0.0)
            cached = self._growth_cache[key] = (market_returns, np.cumprod(1.0 + mr, axis=0))
        growth = cached[1]

        # 1. 調倉點 (第 0 根一律視為起點) 與每根 K 棒所屬的段
        changed = np.ones(T, dtype=bool)
        changed[1:] = np.any(executed_signals[1:] != executed_signals[:-1], axis=1)
        starts = np.flatnonzero(changed)
        seg = np.cumsum(changed) - 1

        # 2. 只在調倉點計算目標權重 (資金分配邏輯同上)
        ex_s = executed_signals[starts]
        if self.config.get('trading', {}).get('allocation_mode', 'dynamic') == 'fixed':
            w_s = ex_s / N
        else:
            cnt = ex_s.sum(axis=1, keepdims=True)
            w_s = np.divide(ex_s, cnt, out=np.zeros_like(ex_s), where=cnt != 0)
        cash_s = 1.0 - w_s.sum(axis=1)
        b_s = w_s / growth[starts]                   # 該段持股在 t 的價值 = growth[t] * b_s

        # 3. 報酬：t-1 所屬那段的持股，從 t-1 到 t 的總值變化
        prev = seg[:-1]
        held = b_s[prev]
        v_before = np.einsum('ij,ij->i', growth[:-1], held) + cash_s[prev]
        v_after = np.einsum('ij,ij->i', growth[1:], held) + cash_s[prev]
        portfolio_returns = np.zeros(T)
        portfolio_returns[1:] = v_after / v_before - 1.0

        # 4. 換手：只有調倉點有，= |目標權重 - 舊持股飄移後的權重|
        turnover_per_bar = np.zeros(T)
        t = starts[starts > 0]
        if t.size:
            drifted = (b_s[seg[t - 1]] * growth[t]) / v_before[t - 1, None] / (1.0 + portfolio_returns[t, None] + 1e-9)
            turnover_per_bar[t] = np.abs(w_s[seg[t]] - drifted).sum(axis=1)

        portfolio_returns -= turnover_per_bar * self.fee
        return np.cumprod(1.0 + portfolio_returns), np.sum(turnover_per_bar)

    def _calculate_sterling_ratio(self, equity_curve: np.ndarray) -> float:
        """Sterling-type ratio = 年化報酬 / (MDD + 0.1)；0.1 避免 MDD 接近 0 時比值發散。"""
        if len(equity_curve) < 2:
            return 0.0

        total_bars = len(equity_curve)
        annual_factor = self.config.get('trading', {}).get('bars_per_year', 252)  # 65 分鐘 K 棒為 1512
        annualized_return = (equity_curve[-1]) ** (annual_factor / total_bars) - 1.0

        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_drawdown = np.max(drawdowns)

        sterling_ratio = annualized_return / (max_drawdown + 0.1)

        return float(sterling_ratio)

    def _calculate_payoff(self, returns: np.ndarray) -> float:
        """盈虧比 = 平均獲利 / 平均虧損"""
        winning_ret = returns[returns > 0]
        losing_ret = returns[returns < 0]
        avg_win = np.mean(winning_ret) if len(winning_ret) > 0 else 0.0
        avg_loss = np.abs(np.mean(losing_ret)) if len(losing_ret) > 0 else 1e-9
        return avg_win / avg_loss if avg_win > 0 else 0.0
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """報酬偏度 E[(X - μ)³] / σ³，只計算非零報酬"""
        valid_returns = returns[returns != 0]
        n = len(valid_returns)
        if n < 3:
            return 0.0

        mean = np.mean(valid_returns)
        std = np.std(valid_returns)
        if std < 1e-8:
            return 0.0

        z_scores = (valid_returns - mean) / std
        skewness = np.mean(z_scores ** 3)

        return float(skewness)

    def _calculate_ror(self, equity_curve: np.ndarray) -> float:
        """年化報酬率"""
        bars_per_year = self.config.get('trading', {}).get('bars_per_year', 252)
        total_bars = len(equity_curve)
        return (equity_curve[-1]) ** ((bars_per_year) / total_bars) - 1.0 if total_bars > 0 else 0.0
