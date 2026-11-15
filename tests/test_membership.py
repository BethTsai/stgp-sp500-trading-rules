"""
test_membership.py
成分股加入 / 剔除規則的測試。用法：python tests/test_membership.py

以逐根 K 棒、逐檔股票的 for 迴圈實作參考模擬器，
與 FitnessEvaluator._run_state_machine(..., active=...) 的向量化結果比對。

規則：
    t 這根持有股票 i  <=>  signals[t-1, i] 且 active[t-1, i] 且 active[t, i] 且 active[t+1, i]
    第 0 根與最後 1 根一律空手。
    -> 加入池子的第一根 K 棒 s：最早在 s+1 這根的 eval_price 買進；
    -> 最後一根在池內的 K 棒 e：在 e 這根的 eval_price 賣出 (持有到 e 的報酬仍計入)。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 讓 src 可被 import
from src.evaluator import FitnessEvaluator

FEE = 0.001


def make_eval(fee_mode='rebalanced', alloc='dynamic', **fit):
    cfg = {'trading': {'transaction_fee': FEE, 'allocation_mode': alloc, 'fee_mode': fee_mode, 'bars_per_year': 1512},
           'evolution': {'objective_mode': 'single'},
           'fitness': {'relative_to_baseline': False, **fit}}
    return FitnessEvaluator(None, cfg)


def ref_sim(signals, mr, A, fee):
    """逐根逐檔的參考模擬器 (只支援 dynamic + rebalanced)。回傳 (淨值曲線, 總換手)。"""
    T, N = mr.shape
    held = np.zeros((T, N))
    for t in range(1, T - 1):
        for i in range(N):
            if signals[t - 1, i] and A[t - 1, i] and A[t, i] and A[t + 1, i]:
                held[t, i] = 1.0
    w = np.zeros((T, N))
    for t in range(T):
        n = held[t].sum()
        if n > 0:
            w[t] = held[t] / n
    nav, curve, total_to = 1.0, np.zeros(T), 0.0
    for t in range(T):
        if t == 0:
            r, to = 0.0, np.abs(w[0]).sum()
        else:
            r = float((w[t - 1] * mr[t]).sum())
            drift = w[t - 1] * (1.0 + mr[t]) / (1.0 + r + 1e-9)
            to = np.abs(w[t] - drift).sum()
        nav *= 1.0 + r - fee * to
        curve[t] = nav
        total_to += to
    return curve, total_to


def random_membership(rng, T, N):
    """每檔股票有 1~2 段成分股期間 (模擬加入、剔除與重新加入)，部分股票全期都是成分股。"""
    A = np.zeros((T, N), dtype=bool)
    for i in range(N):
        kind = rng.integers(0, 4)
        if kind == 0:
            A[:, i] = True
        elif kind == 1:
            s = rng.integers(0, T // 2); A[s:, i] = True
        elif kind == 2:
            e = rng.integers(T // 2, T); A[:e, i] = True
        else:
            a, b, c, d = np.sort(rng.integers(0, T, 4)); A[a:b, i] = True; A[c:d, i] = True
    return A


def rand_data(rng, T=300, N=8):
    mr = np.zeros((T, N)); mr[1:] = rng.normal(0.0003, 0.01, (T - 1, N))
    sig = rng.random((T, N)) < rng.uniform(0.2, 0.8)
    return sig, mr


def test_matches_reference():
    ev = make_eval()
    rng = np.random.default_rng(0)
    for k in range(200):
        sig, mr = rand_data(rng, T=int(rng.integers(30, 300)), N=int(rng.integers(2, 12)))
        A = random_membership(rng, *sig.shape)
        eq, to = ev._run_state_machine(sig, mr, mr, mr, active=A)
        ref_eq, ref_to = ref_sim(sig, mr, A, FEE)
        assert np.allclose(eq, ref_eq, atol=1e-12), f"case {k}: 淨值與參考模擬器不同"
        assert abs(to - ref_to) < 1e-9, f"case {k}: 換手不同 {to} vs {ref_to}"
    print("PASS test_matches_reference  (200 組隨機股票池 / 訊號)")


def test_none_and_all_true_identical():
    ev = make_eval()
    rng = np.random.default_rng(1)
    for fee_mode in ('rebalanced', 'signal_driven'):
        for alloc in ('dynamic', 'fixed'):
            ev = make_eval(fee_mode, alloc)
            sig, mr = rand_data(rng)
            e0, t0 = ev._run_state_machine(sig, mr, mr, mr)
            e1, t1 = ev._run_state_machine(sig, mr, mr, mr, active=np.ones(sig.shape, dtype=bool))
            assert np.array_equal(e0, e1) and t0 == t1, f"{fee_mode}/{alloc}: active 全 True 應與 None 逐位元相同"
    print("PASS test_none_and_all_true_identical  (4 種 fee_mode x allocation_mode)")


def test_entry_exit_timing():
    """股票 1 在 bar 2~5 在池內，訊號全 True。持有 bar 3、4 -> 只有 mr[4]、mr[5] 會影響淨值。"""
    ev = make_eval()
    T = 12
    sig = np.ones((T, 2), dtype=bool)
    A = np.ones((T, 2), dtype=bool)
    A[:, 1] = False; A[2:6, 1] = True
    base = np.zeros((T, 2))

    def final(bar):
        mr = base.copy(); mr[bar, 1] = 0.5            # 在某一根給股票 1 一個 +50% 的報酬
        eq, _ = ev._run_state_machine(sig, mr, mr, mr, active=A)
        return eq[-1]

    ref = final(1)       # 池外的報酬，應該完全不影響
    for bar in (1, 2, 3, 6, 7):
        assert abs(final(bar) - ref) < 1e-12, f"bar {bar} 不應影響淨值 (買進那根的跳空 / 剔除後)"
    for bar in (4, 5):
        assert abs(final(bar) - ref) > 1e-4, f"bar {bar} 應影響淨值 (持有期間的報酬)"
    print("PASS test_entry_exit_timing  (加入後最早下一根買進；最後一根在池內的 K 棒賣出)")


def test_signal_outside_pool_ignored():
    """池外股票 (active=False 的那些根) 的訊號不論真假，結果都相同。"""
    ev = make_eval()
    rng = np.random.default_rng(2)
    for _ in range(30):
        sig, mr = rand_data(rng)
        A = random_membership(rng, *sig.shape)
        e1, _ = ev._run_state_machine(sig, mr, mr, mr, active=A)
        flip = np.where(A, sig, ~sig)                 # 只翻轉池外的訊號
        e2, _ = ev._run_state_machine(flip, mr, mr, mr, active=A)
        assert np.allclose(e1, e2), "池外訊號翻轉後結果不同"
    print("PASS test_signal_outside_pool_ignored")


def test_degenerate_exposure_uses_pool():
    """持倉比例的分母應是池內股票數；池外股票的訊號不能把比例拉高或拉低。"""
    ev = make_eval(max_long_ratio=0.95, min_long_ratio=0.05)
    T, N = 50, 4
    A = np.zeros((T, N), dtype=bool); A[:, :2] = True      # 只有 2 檔在池內
    sig = np.zeros((T, N), dtype=bool); sig[:, 2:] = True   # 只有池外的 2 檔有訊號
    assert not ev._is_degenerate_exposure(sig)               # 不考慮股票池：平均持倉 50%，看似正常
    assert ev._is_degenerate_exposure(sig, A)               # 考慮股票池：池內完全空手 -> 無效
    sig2 = np.ones((T, N), dtype=bool); sig2[:, :2] = False
    assert not ev._is_degenerate_exposure(sig2) and ev._is_degenerate_exposure(sig2, A)
    sig3 = np.zeros((T, N), dtype=bool); sig3[::2, :2] = True
    assert not ev._is_degenerate_exposure(sig3, A)
    print("PASS test_degenerate_exposure_uses_pool")


def test_baseline_uses_pool():
    """基準 (全 True) 經過同一個狀態機 = 等權持有池內股票；與直接以 active 當訊號相同。"""
    ev = make_eval()
    rng = np.random.default_rng(3)
    sig, mr = rand_data(rng)
    A = random_membership(rng, *sig.shape)
    w = {'close': mr, 'eval_price': mr, 'market_returns': mr, 'active': A}
    s_base = ev._baseline_score(w)
    eq, _ = ev._run_state_machine(A.copy(), mr, mr, mr, active=A)
    assert abs(s_base - ev.score_window(eq)) < 1e-12
    print("PASS test_baseline_uses_pool")


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    fails = 0
    for t in tests:
        try:
            t()
        except Exception as e:                       # noqa: BLE001
            fails += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} 通過")
    sys.exit(1 if fails else 0)
