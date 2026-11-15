"""
features_pit.py
以 universe_data_build.py 產生的 universe_raw.pkl 計算 point-in-time 股票池的特徵，輸出 DataProvider 使用的特徵檔。

用法：
    python data_pipeline/features_pit.py --stage 27 --market         # 27 檔股票池 -> data/processed/features_stage27.pkl
    python data_pipeline/features_pit.py --stage 55 --market         # 55 檔股票池 -> data/processed/features_stage55.pkl
    (--market：加入 3 個大盤狀態特徵 Rank_Mkt_*)

每個股票池各自計算：橫斷面排名與 CAPM 的市場報酬只在池內股票之間計算，股票池不同，特徵值也不同。

輸出 pkl (gzip pickle，dict)：
    'eval_price' 'close'          DataFrame (index = UTC 無時區的 K 棒時間，columns = 股票)
    特徵名                         DataFrame float32，排名版落在 [0,1]，暖機期/池外 = 0.5
    'active'                      bool，K 棒 t 股票是否在池內
    'filled'                      bool，該 K 棒是否為補值
    ('active'、'filled' 為遮罩，不是特徵；DataProvider 會將其排除)

設計重點：
  * 排名只在當下的成分股之間計算。
  * CAPM：日報酬；市場報酬 = 當天其他成分股的等權平均 (不含該股本身)；扣除 FRED DTB3 日利率；
    滾動 OLS (N=20/60 日)；延遲一日；同一天 6 根 K 棒共用同一個值。
  * 大盤特徵 (--market)：命名為 Rank_Mkt_*，使 primitives.get_feature_type 判定為 RankType。
"""
import argparse
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 專案根目錄
RAW_PKL = os.path.join(ROOT, 'data', 'processed', 'universe_raw.pkl')
sys.path.insert(0, ROOT)
from data_pipeline.universe_data_build import build_active  # noqa: E402

N_ROC = (1, 30, 240, 720)
N_HV = (30, 240)
HV_ANNUAL = np.sqrt(365)          # 年化係數；排名後不影響結果


# ---------------------------------------------------------------- 技術指標
def roc(C, n):
    return (C / C.shift(n) - 1.0) * 100.0


def ibs(C, H, L, n):
    hh, ll = H.rolling(n).max(), L.rolling(n).min()
    den = hh - ll
    out = (C - ll) / den.where(den > 0)
    # 補值棒或一字棒 (H=L) 分母為 0 -> 0.5；暖機期維持 NaN
    return out.where(den.isna() | (den > 0), 0.5)


def seeded_ema(C, n):
    """TradingView 式：以前 n 根簡單平均為種子，之後 ema = alpha*x + (1-alpha)*ema，alpha = 2/(n+1)。"""
    a = 2.0 / (n + 1)
    out = np.full(C.shape, np.nan)
    arr = C.values
    for j in range(arr.shape[1]):
        v = arr[:, j]
        ok = np.flatnonzero(~np.isnan(v))
        if len(ok) < n:
            continue
        s = ok[0]
        e = v[s:s + n].mean()
        out[s + n - 1, j] = e
        for t in range(s + n, len(v)):
            e = a * v[t] + (1 - a) * e
            out[t, j] = e
    return pd.DataFrame(out, index=C.index, columns=C.columns)


def hv(C, n):
    lr = np.log(C / C.shift(1))
    return lr.rolling(n).std(ddof=0) * HV_ANNUAL * 100.0


def relvol(V, n):
    m = V.rolling(n).mean()
    return V / m.where(m > 0)


def xrank(raw, active):
    """橫斷面百分位排名，只在池內股票之間 (pandas rank(pct=True))。"""
    return raw.where(active).rank(axis=1, pct=True)


# ---------------------------------------------------------------- CAPM (日線)
def rf_daily(rf_pct):
    """FRED DTB3 (% 年利率) -> 日利率 (1+y/100)^(1/252)-1；負值保留。"""
    return (1.0 + rf_pct / 100.0) ** (1.0 / 252.0) - 1.0


def capm_daily(dclose, dactive, rf_pct, windows=(20, 60), market=None):
    """
    dclose : 日 close (index = 日期, columns = 股票)；dactive：同形狀 bool。
    market : None -> 自己股票池「排除自己」的等權平均；或 Series (例如 SPY 日報酬) -> 所有股票共用。
    回傳 {'alpha20':..., 'beta20':..., ...}，尚未延遲一日。
    """
    r = dclose.pct_change()
    valid = r.notna() & dactive
    if market is None:
        rz = r.where(valid, 0.0)
        S = rz.sum(axis=1)
        N = valid.sum(axis=1)
        m_sum = (-rz).add(S, axis=0)                           # 排除自己：總和 - 自己
        m_cnt = (-valid.astype(float)).add(N, axis=0)
        mkt = m_sum / m_cnt.where(m_cnt > 0)
    else:
        mkt = pd.DataFrame({c: market for c in r.columns})
    rf = rf_daily(rf_pct).reindex(r.index)
    y = r.sub(rf, axis=0)
    x = mkt.sub(rf, axis=0)
    out = {}
    for n in windows:
        mx, my = x.rolling(n).mean(), y.rolling(n).mean()
        cov = (x * y).rolling(n).mean() - mx * my
        var = (x * x).rolling(n).mean() - mx * mx
        beta = cov / var.where(var > 1e-14)
        out[f'beta{n}'] = beta
        out[f'alpha{n}'] = my - beta * mx
    return out


# ---------------------------------------------------------------- 大盤特徵
def rolling_pct(s, w):
    """過去 w 根 (含當根) 中，<= 當根值的比例；暖機期 NaN。只用過去資料。"""
    return s.rolling(w, min_periods=w).apply(lambda a: (a <= a[-1]).mean(), raw=True)


def market_features(bars, active, filled, w=1512, breadth_ema=240):
    C = bars['close']
    ret = C / C.shift(1) - 1.0
    ok = active & ~filled & ret.notna()
    rm = ret.where(ok).mean(axis=1).fillna(0.0)                 # 等權指數報酬 (只含池內且為真實報價的股票)
    idx = (1.0 + rm).cumprod()
    trend = idx / idx.shift(240) - 1.0
    lr = np.log(idx / idx.shift(1))
    vol = lr.rolling(30).std(ddof=0) * HV_ANNUAL * 100.0
    ema = seeded_ema(C, breadth_ema)
    valid = active & ema.notna()
    breadth = ((C > ema) & valid).sum(axis=1) / valid.sum(axis=1).where(valid.sum(axis=1) > 0)
    res = {'Rank_Mkt_Trend': rolling_pct(trend, w), 'Rank_Mkt_Vol': rolling_pct(vol, w), 'Rank_Mkt_Breadth': breadth}
    return {k: pd.DataFrame({c: v for c in C.columns}) for k, v in res.items()}


# ---------------------------------------------------------------- 主流程
def load_raw(path=RAW_PKL):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def build_features(raw, stage, market=False, capm_market=None):
    m = raw['membership']
    syms = [s for s in raw['symbols'] if s in set(m[m['stage'] <= stage]['symbol'])]
    bars = {k: v[syms] for k, v in raw['bars'].items()}
    daily = {k: v[syms] for k, v in raw['daily'].items()}
    idx = bars['close'].index
    active = build_active(m, idx, syms, stage)
    bday = idx.tz_localize(None).normalize()
    C, H, L, V = bars['close'], bars['high'], bars['low'], bars['volume']
    feats = {}

    # B 組：技術指標
    for n in N_ROC:
        feats[f'Rank_ROC{n}'] = xrank(roc(C, n), active)
    feats['IBS3'] = ibs(C, H, L, 3).where(active)
    feats['Rank_Bias_EMA6'] = xrank(C / seeded_ema(C, 6) - 1.0, active)
    for n in N_HV:
        feats[f'Rank_HV{n}'] = xrank(hv(C, n), active)
    feats['Rank_RelVol20'] = xrank(relvol(V, 20), active)

    # A 組：CAPM (日線 -> 延遲一日 -> 對應到每根 K 棒)
    dact = active.groupby(bday).first().reindex(daily['close'].index).fillna(False)
    cap = capm_daily(daily['close'], dact, raw['rf'], market=capm_market)
    for key in ('alpha20', 'alpha60', 'beta60'):
        lagged = cap[key].shift(1)
        per_bar = lagged.reindex(bday)
        per_bar.index = idx
        feats[f'Rank_{key}'] = xrank(per_bar, active)

    if market:
        feats.update(market_features(bars, active, bars['filled'], ))

    out = {'eval_price': ((H + L) / 2.0).ffill(), 'close': C}
    for k, v in feats.items():
        out[k] = v.fillna(0.5).astype(np.float32)
    out['active'] = active
    out['filled'] = bars['filled']
    # 時間索引：美東時間 -> 無時區 UTC (DataProvider 以此切分日期)
    utc = idx.tz_convert('UTC').tz_localize(None)
    for k in out:
        out[k] = out[k].set_axis(utc, axis=0)
        out[k].index.name = 'time'
    return out, cap


def summarize(out, stage):
    L = [f"stage {stage}：{out['close'].shape[1]} 檔，{len(out['close'])} 根 K 棒 "
         f"({out['close'].index.min()} ~ {out['close'].index.max()})"]
    act = out['active']
    for k, v in out.items():
        if k in ('eval_price', 'close', 'active', 'filled'):
            continue
        a = v.values[act.values]
        L.append(f"  {k:18s} 範圍[{a.min():.3f},{a.max():.3f}] 池內均值{a.mean():.3f} 恰為0.5的比例{(a == 0.5).mean():.3%}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, default=55)
    ap.add_argument('--market', action='store_true', help='加入 3 個大盤特徵 Rank_Mkt_*')
    ap.add_argument('--raw', default=RAW_PKL)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    raw = load_raw(a.raw)
    out, _ = build_features(raw, a.stage, market=a.market)
    path = a.out or os.path.join(ROOT, 'data', 'processed', f'features_stage{a.stage}.pkl')
    with gzip.open(path, 'wb') as f:
        pickle.dump(out, f)
    print(summarize(out, a.stage))
    print(f"已輸出 {path}")


if __name__ == '__main__':
    main()
