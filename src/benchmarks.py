"""
benchmarks.py
==========================================================================
回測報表使用的基準，時間對齊、手續費與期末平倉規則皆與狀態機相同：
  true_bh  Buy & Hold：第 1 根 K 棒以 eval_price 等權買入，之後不再交易，最後 1 根平倉。
           手續費為進場與出場各一次 (各為 fee × 資金)，換手 = 2。
           有 active (成分股異動) 時改為事件驅動等權：成分股異動時才調回等權 (賣出被剔除者、買入新成員)，
           其餘時間權重隨價格飄移。另提供 true_bh_cash：被剔除者賣出後持有現金、不買入新成員 (參考用)。
  ew_same  每根 K 棒等權再平衡，使用與 GP 相同的 fee_mode；作為不選股、全額等權配置的對照。
  ew_old   全 True + signal_driven (再平衡不收費)，僅供與早期結果對照。

用法：
    bench = compute_benchmarks(test_evaluator, test_close, test_eval_price, test_market_returns)
    bh = bench['true_bh']   # dict: curve / turnover / sterling / ror / mdd
不會修改傳入 evaluator.config (用深拷貝的 config 建立暫時的 evaluator)。
自我檢查：python -m src.benchmarks
==========================================================================
"""
import copy

import numpy as np

from src.evaluator import FitnessEvaluator


def _curve_stats(evaluator, curve):
    """與報表其他地方同一套指標定義 (年化 Sterling / 年化報酬 / MDD)。"""
    running_max = np.maximum.accumulate(curve)
    mdd = float(np.max((running_max - curve) / running_max))
    return {
        'sterling': float(evaluator._calculate_sterling_ratio(curve)),
        'ror': float(evaluator._calculate_ror(curve)),
        'mdd': mdd,
    }


def true_buy_and_hold_curve(market_returns: np.ndarray, fee: float):
    """
    真正的 Buy & Hold 淨值曲線 (長度與狀態機輸出相同)。

    時間對齊 (與 evaluator._run_state_machine 一致)：
      狀態機的 executed_signals[0]=0，第一個持倉在 t=1 (以 eval_price[1] 成交)，
      最後一根 executed_signals[-1]=0，所以持有到 t=T-1 的 eval_price 並平倉。
      -> 進場 t=1，出場 t=T-1；進出場各收 fee × 100% 資金 (換手合計 = 2)。

    計算：
      每檔股票的成長倍數 G_i[t] = Π_{s=2..t} (1 + market_returns[s, i])  (= P_t / P_1)
      等權買入後不再平衡 -> 組合淨值 = 各檔 G_i[t] 的平均 (各檔初始權重 1/N，隨價格自然飄移)
      淨值 = (1 - fee) × mean_i G_i[t]，最後一根再乘 (1 - fee) 代表出場費用。
    """
    mr = np.nan_to_num(np.asarray(market_returns, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    T, _ = mr.shape
    curve = np.ones(T)
    if T < 3:
        return curve, 0.0
    growth = np.cumprod(1.0 + mr[2:, :], axis=0)                 # t=2..T-1 的 P_t / P_1
    growth = np.vstack([np.ones((1, mr.shape[1])), growth])      # 加上 t=1 (=1)
    portfolio = growth.mean(axis=1)                              # 等權、不再平衡
    curve[1:] = (1.0 - fee) * portfolio                          # 進場費
    curve[-1] *= (1.0 - fee)                                     # 出場費 (期末強制平倉)
    turnover = 2.0
    return curve, turnover


def pit_buy_and_hold_curve(market_returns: np.ndarray, active: np.ndarray, fee: float, reinvest: bool = True):
    """
    股票池會異動時的 Buy & Hold (全部訊號 True 的持有規則與狀態機相同)：
      t 期持有股票 i  <=>  active[t-1,i] 且 active[t,i] 且 active[t+1,i]  (第 0 根與最後 1 根空手)
    reinvest=True ：持有名單一有變動 (有人被剔除 / 有人加入) 就調回等權 (被剔除的錢買新成員)；沒變動不動。
    reinvest=False：期初買入持有名單；被剔除的賣出後放現金 (不計息)，不買新成員；其他不動。
    手續費 = 換手 × fee，從該根報酬扣除 (與狀態機相同)。回傳 (淨值曲線, 總換手)。
    """
    mr = np.nan_to_num(np.asarray(market_returns, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    a = np.asarray(active, dtype=bool)
    T, N = mr.shape
    held = np.zeros((T, N), dtype=bool)
    if T >= 3:
        held[1:-1] = a[:-2] & a[1:-1] & a[2:]
    curve = np.ones(T)
    w = np.zeros(N)                       # 各檔占淨值的比例 (其餘為現金)
    nav, total_to = 1.0, 0.0
    initial = None
    for t in range(T):
        r = float(np.sum(w * mr[t])) if t > 0 else 0.0
        drift = w * (1.0 + mr[t]) / (1.0 + r + 1e-9) if t > 0 else w
        if reinvest:
            target = drift
            if t == 0 or np.any(held[t] != held[t - 1]):
                n = held[t].sum()
                target = held[t] / n if n > 0 else np.zeros(N)
        else:
            if initial is None and held[t].any():
                initial = held[t].copy()
            target = drift.copy()
            if initial is None:
                target = np.zeros(N)
            elif t == 0 or np.any(held[t] != held[t - 1]):
                if t > 0 and held[t - 1].any() and not held[t].any():
                    target = np.zeros(N)                   # 全部被剔除 / 期末平倉
                else:
                    target = np.where(held[t] & initial, drift, 0.0)   # 被剔除的賣掉；不買新成員
                    if t > 0 and not held[t - 1].any():    # 第一次建倉：等權買入
                        target = initial.astype(float) / initial.sum() * held[t]
        to = float(np.abs(target - drift).sum())
        total_to += to
        nav *= 1.0 + r - fee * to
        curve[t] = nav
        w = target
    return curve, total_to


def compute_benchmarks(evaluator: FitnessEvaluator, close, eval_price, market_returns, active=None) -> dict:
    """
    回傳 {'true_bh': {...}, 'ew_same': {...}, 'ew_old': {...}}，每個內容：
        curve, turnover, sterling, ror, mdd
    不會修改傳入 evaluator.config (用深拷貝的 config 建立暫時的 evaluator)。
    """
    cfg = evaluator.config
    fee = evaluator.fee
    out = {}

    # 1) 真正 Buy & Hold
    if active is None:
        curve, turnover = true_buy_and_hold_curve(market_returns, fee)
    else:
        curve, turnover = pit_buy_and_hold_curve(market_returns, active, fee, reinvest=True)
        c2, t2 = pit_buy_and_hold_curve(market_returns, active, fee, reinvest=False)
        out['true_bh_cash'] = {'curve': c2, 'turnover': float(t2), **_curve_stats(evaluator, c2)}
    out['true_bh'] = {'curve': curve, 'turnover': turnover, **_curve_stats(evaluator, curve)}

    all_true = np.ones_like(close, dtype=bool)

    # 2) 等權每根 K 棒再平衡：沿用 config 的 fee_mode (與 GP 同口徑)
    ew_cfg = copy.deepcopy(cfg)
    ew_cfg.setdefault('trading', {})['allocation_mode'] = 'dynamic'
    ew_eval = FitnessEvaluator(data_provider=None, config=ew_cfg)
    curve, turnover = ew_eval._run_state_machine(all_true, close, eval_price, market_returns, active=active)
    out['ew_same'] = {'curve': curve, 'turnover': float(turnover), **_curve_stats(evaluator, curve)}

    # 3) 全 True + signal_driven (再平衡不收費)，僅供對照
    old_cfg = copy.deepcopy(cfg)
    old_cfg.setdefault('trading', {})['fee_mode'] = 'signal_driven'
    old_cfg['trading']['allocation_mode'] = 'dynamic'
    old_eval = FitnessEvaluator(data_provider=None, config=old_cfg)
    curve, turnover = old_eval._run_state_machine(all_true, close, eval_price, market_returns, active=active)
    out['ew_old'] = {'curve': curve, 'turnover': float(turnover), **_curve_stats(evaluator, curve)}
    return out


if __name__ == '__main__':
    # 合成資料自我檢查：真 B&H 與「直接用價格算」的結果應完全一致
    rng = np.random.default_rng(1)
    T, N, fee = 500, 6, 0.003
    prices = 100 * np.cumprod(1 + rng.normal(0.0008, 0.01, (T, N)), axis=0)
    mr = np.zeros((T, N))
    mr[1:] = prices[1:] / prices[:-1] - 1

    curve, to = true_buy_and_hold_curve(mr, fee)
    manual = np.ones(T)
    manual[1:] = (1 - fee) * (prices[1:] / prices[1]).mean(axis=1)
    manual[-1] *= (1 - fee)
    print('與手算最大差異:', np.abs(curve - manual).max(), '| 換手:', to)

    cfg = {'trading': {'transaction_fee': fee, 'allocation_mode': 'dynamic', 'fee_mode': 'rebalanced',
                       'bars_per_year': 1512},
           'evolution': {'objective_mode': 'single'}}
    ev = FitnessEvaluator(None, cfg)
    res = compute_benchmarks(ev, prices, prices, mr)
    for k, v in res.items():
        print(f"{k:8s} 期末淨值={v['curve'][-1]:.4f} 換手={v['turnover']:.2f} "
              f"Sterling={v['sterling']:.3f} 年化={v['ror']:.3f} MDD={v['mdd']:.3f}")
    print('config 未被修改 (fee_mode):', ev.config['trading']['fee_mode'])
