"""
universe_data_build.py
將 TradingView 匯出的 65 分鐘與日 K CSV 整理成「時間 × 股票」矩陣，並依成分股期間建立 active 矩陣。

用法：
    python data_pipeline/universe_data_build.py                 # 建立 data/processed/universe_raw.pkl + build_report.txt

輸出 data/processed/universe_raw.pkl (gzip pickle，dict)：
    'symbols'  股票代號 list (字母序)
    'bars'     dict[str, DataFrame]：open/high/low/close/volume/filled，index = 美東時間 K 棒，columns = 股票
    'daily'    dict[str, DataFrame]：open/high/low/close/volume/filled，index = 日期 (午夜)，columns = 股票
    'rf'       Series：FRED DTB3 (% 年利率)，ffill 到日 K 日期
    'spy'      Series：SPY 日 K 收盤 (只用來驗證 CAPM，不進實驗)
    'membership' DataFrame：data/raw/sp500_membership.csv
    'meta'     建立參數與每檔補值統計

補值規則 (補值的 K 棒仍可交易)：
  65 分鐘缺棒/缺整天：open=high=low=close=前一根 close，volume=0，filled=True
  日 K 缺日：若該日有 65 分鐘資料 -> 以 65 分鐘合成 (乘上前一個有資料日的 日K/65m 價格比例，修正不同下載批次的調整差異)；
             否則用前收補平；filled=True
"""
import argparse
import glob
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 專案根目錄
TZ = 'America/New_York'
START = pd.Timestamp('2013-01-02')
END = pd.Timestamp('2025-12-31')
OHLCV = ['open', 'high', 'low', 'close', 'volume']

RAW_DIR = os.path.join(ROOT, 'data', 'raw')
PRICES_65 = os.path.join(RAW_DIR, 'prices_65min')      # 每檔股票一個 65 分鐘 K 棒 CSV
PRICES_1D = os.path.join(RAW_DIR, 'prices_daily')      # 每檔股票一個日 K CSV
SPY_CSV = os.path.join(RAW_DIR, 'spy_daily.csv')
FRED_CSV = os.path.join(RAW_DIR, 'fred_dtb3.csv')
MEMBER_CSV = os.path.join(RAW_DIR, 'sp500_membership.csv')
OUT_DIR = os.path.join(ROOT, 'data', 'processed')


# ---------------------------------------------------------------- 讀檔
def _sym(path):
    """由檔名取得股票代號：'AAPL.csv'、'BATS_AAPL, 65.csv' (TradingView 預設檔名)、'AAPL_65min.csv' -> 'AAPL'"""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith('BATS_'):
        return stem[5:].split(',')[0].strip()
    return stem.split('_')[0]


def _files(folder):
    files = {}
    for p in sorted(glob.glob(os.path.join(folder, '*.csv'))):
        s = _sym(p)
        if s in files:
            raise ValueError(f"{folder} 內有兩個檔案對應同一個代號 {s}：{files[s]}、{p}")
        files[s] = p
    return files


def read_65(path):
    d = pd.read_csv(path, usecols=lambda c: c in ('time', 'open', 'high', 'low', 'close', 'Volume'))
    d = d.rename(columns={'Volume': 'volume'})
    t = pd.to_datetime(d['time'], utc=True).dt.tz_convert(TZ)
    d = d.drop(columns='time').set_index(t)[OHLCV].astype(float)
    d.index.name = None
    d = d[~d.index.duplicated(keep='last')].sort_index()
    day = d.index.tz_localize(None).normalize()
    return d[(day >= START) & (day <= END)]


def read_1d(path):
    d = pd.read_csv(path, usecols=lambda c: c in ('time', 'open', 'high', 'low', 'close', 'Volume'))
    d = d.rename(columns={'Volume': 'volume'})
    d = d.set_index(pd.to_datetime(d['time']).dt.normalize()).drop(columns='time')[OHLCV].astype(float)
    d.index.name = None
    d = d[~d.index.duplicated(keep='last')].sort_index()
    return d[(d.index >= START) & (d.index <= END)]


def collect_sources():
    """回傳 (代號 -> 65 分鐘 CSV 路徑, 代號 -> 日 K CSV 路徑)。"""
    f65, f1d = _files(PRICES_65), _files(PRICES_1D)
    if not f65 or not f1d:
        raise FileNotFoundError(f"找不到價格資料：請把 CSV 放在 {PRICES_65} 與 {PRICES_1D} (格式見 data/README.md)")
    return f65, f1d


# ---------------------------------------------------------------- 65 分鐘
def _drop_postclose(cal, raw65, min_frac=0.5):
    """半天交易日 (13:00 收盤) 少數股票會出現收盤後的零星成交 K 棒 (13:50 / 14:55)，不屬於可交易的 K 棒。
    判斷：該日有 >= 50% 股票報價的 K 棒只有 4 根，且 12:45 之後的 K 棒少於 min_frac (50%) 的股票有資料 -> 移除。"""
    n = len(raw65)
    cnt = pd.Series(0, index=cal)
    for d in raw65.values():
        cnt[d.index.intersection(cal)] += 1
    frac = cnt / n
    day = cal.tz_localize(None).normalize()
    main_bars = pd.Series((frac >= 0.5).values, index=day).groupby(level=0).sum()
    half = pd.Series(main_bars.reindex(day).values == 4, index=cal)
    late = pd.Series(cal.strftime('%H:%M') > '12:45', index=cal)
    drop = half & late & (frac < min_frac)
    return cal[~drop.values], [str(t) for t in cal[drop.values]]


def build_bars(raw65):
    """raw65: dict[sym] -> DataFrame。回傳 (bars dict, filled DataFrame, 每檔統計)。"""
    syms = sorted(raw65)
    cal = sorted(set().union(*[set(d.index) for d in raw65.values()]))
    cal = pd.DatetimeIndex(cal).tz_convert(TZ) if not isinstance(cal, pd.DatetimeIndex) else cal
    cal = pd.DatetimeIndex(sorted(cal))
    cal, dropped = _drop_postclose(cal, raw65)
    out = {k: pd.DataFrame(index=cal, columns=syms, dtype=float) for k in OHLCV}
    filled = pd.DataFrame(False, index=cal, columns=syms)
    stats = {}
    for s in syms:
        d = raw65[s].reindex(cal)
        real = d['close'].notna()
        first = real.idxmax()
        started = pd.Series(np.arange(len(cal)) >= cal.get_loc(first), index=cal)
        miss = ~real & started
        c = d['close'].ffill()
        for k in ('open', 'high', 'low'):
            d[k] = d[k].where(real, c)
        d['close'] = c
        d['volume'] = d['volume'].where(real, 0.0)
        d.loc[~started, OHLCV] = np.nan
        for k in OHLCV:
            out[k][s] = d[k]
        filled[s] = miss
        stats[s] = dict(bars_real=int(real.sum()), bars_filled=int(miss.sum()),
                        filled_pct=round(100 * miss.sum() / max(1, started.sum()), 3),
                        first=str(first), leading_nan=int((~started).sum()))
    out['filled'] = filled
    stats['_dropped_postclose'] = dropped
    return out, stats


# ---------------------------------------------------------------- 日 K
def build_daily(raw1d, bars):
    """raw1d: dict[sym] -> 日 K。缺日用 65 分鐘合成 (含比例修正) 或前收補平。"""
    syms = sorted(raw1d)
    cal = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in raw1d.values()])))
    out = {k: pd.DataFrame(index=cal, columns=syms, dtype=float) for k in OHLCV}
    filled = pd.DataFrame(False, index=cal, columns=syms)
    stats = {}
    bar_day = bars['close'].index.tz_localize(None).normalize()
    for s in syms:
        d = raw1d[s].reindex(cal)
        real = d['close'].notna()
        first = real.idxmax()
        started = pd.Series(cal >= first, index=cal)
        miss = ~real & started
        n_synth = n_flat = 0
        if s in bars['close'].columns and miss.any():
            b = pd.DataFrame({k: bars[k][s].values for k in OHLCV}, index=bar_day)
            b_real = ~bars['filled'][s].values & bars['close'][s].notna().values
            for dt in cal[miss.values]:
                m = (bar_day == dt) & b_real
                if m.sum() == 0:
                    continue
                prev = [x for x in cal[cal < dt] if real[x] and (bar_day == x).any()]
                ratio = 1.0
                if prev:
                    p = prev[-1]
                    lastp = b[(bar_day == p) & b_real]['close']
                    if len(lastp):
                        ratio = d.loc[p, 'close'] / lastp.iloc[-1]
                g = b[m]
                d.loc[dt, ['open', 'high', 'low', 'close']] = [g['open'].iloc[0] * ratio, g['high'].max() * ratio,
                                                               g['low'].min() * ratio, g['close'].iloc[-1] * ratio]
                d.loc[dt, 'volume'] = g['volume'].sum()
                n_synth += 1
        still = d['close'].isna() & started
        n_flat = int(still.sum())
        c = d['close'].ffill()
        for k in ('open', 'high', 'low'):
            d[k] = d[k].where(~still, c)
        d['close'] = c
        d['volume'] = d['volume'].where(~still, 0.0)
        d.loc[~started, OHLCV] = np.nan
        for k in OHLCV:
            out[k][s] = d[k]
        filled[s] = miss
        stats[s] = dict(days_real=int(real.sum()), days_synth=n_synth, days_flat=n_flat,
                        missing_dates=[str(x.date()) for x in cal[miss.values]][:10])
    out['filled'] = filled
    return out, stats


# ---------------------------------------------------------------- 成員資格
def read_membership():
    m = pd.read_csv(MEMBER_CSV, parse_dates=['start_date', 'end_date'])
    return m


def build_active(membership, bar_index, symbols, stage):
    """active[t, i]：K 棒 t 的美東日期落在股票 i 任一段 [start_date, end_date] 內。stage = 含此階段以前加入的股票。"""
    day = bar_index.tz_localize(None).normalize() if bar_index.tz is not None else bar_index.normalize()
    act = pd.DataFrame(False, index=bar_index, columns=list(symbols))
    m = membership[membership['stage'] <= stage]
    for _, r in m.iterrows():
        if r['symbol'] not in act.columns:
            continue
        hi = r['end_date'] if pd.notna(r['end_date']) else pd.Timestamp.max
        act[r['symbol']] |= ((day >= r['start_date']) & (day <= hi))
    return act


# ---------------------------------------------------------------- 驗證報告
def make_report(bars, daily, bstats, dstats, membership, rf, spy):
    L = []
    cal = bars['close'].index
    days = cal.tz_localize(None).normalize()
    per_day = pd.Series(1, index=days).groupby(level=0).sum()
    L.append(f"65 分鐘：{len(bars['close'].columns)} 檔，{len(cal)} 根 K 棒，{per_day.shape[0]} 個交易日 "
             f"({days.min().date()} ~ {days.max().date()})")
    L.append(f"  每日 K 棒數分布：{per_day.value_counts().sort_index().to_dict()}")
    times = pd.Series(cal.strftime('%H:%M')).value_counts()
    L.append(f"  盤中時間點 (應只有 09:30/10:35/11:40/12:45/13:50/14:55)：{times.to_dict()}")
    dd = daily['close'].index
    L.append(f"日 K：{len(daily['close'].columns)} 檔，{len(dd)} 個交易日 ({dd.min().date()} ~ {dd.max().date()})")
    only_bar = sorted(set(per_day.index) - set(dd))
    only_day = sorted(set(dd) - set(per_day.index))
    L.append(f"  只在 65m 日曆、不在日 K 日曆：{[str(x.date()) for x in only_bar][:10]}")
    L.append(f"  只在日 K 日曆、不在 65m 日曆：{[str(x.date()) for x in only_day][:10]}")
    L.append("每檔補值統計 (65m 補棒 / 補棒% / 日K 合成天 / 日K 前收補天 / 開頭 NaN 棒)：")
    L.append(f"  收盤後零星棒已丟掉 {len(bstats.get('_dropped_postclose', []))} 根：{bstats.get('_dropped_postclose', [])}")
    for s in sorted(k for k in bstats if not k.startswith('_')):
        b, d = bstats[s], dstats.get(s, {})
        L.append(f"  {s:5s} 補{b['bars_filled']:5d} ({b['filled_pct']:6.3f}%) 起{b['first'][:10]} "
                 f"| 日K合成 {d.get('days_synth', '-')} 前收補 {d.get('days_flat', '-')} 缺日 {d.get('missing_dates', '')}"
                 f"{' | 開頭NaN ' + str(b['leading_nan']) if b['leading_nan'] else ''}")
    L.append("缺 65m 或缺日 K 的股票：" + str(sorted(set(k for k in bstats if not k.startswith('_')) ^ set(dstats))))
    L.append(f"rf：{rf.index.min().date()} ~ {rf.index.max().date()}，NaN={int(rf.isna().sum())}，範圍 {rf.min():.3f}~{rf.max():.3f} %")
    L.append(f"SPY 日 K：{len(spy)} 天；與日 K 日曆差：{sorted(set(dd) - set(spy.index))[:5]}")
    for stage in (27, 55):
        act = build_active(membership, cal, bars['close'].columns, stage)
        n = act.sum(axis=1)
        n = n[n.index.tz_localize(None) >= pd.Timestamp('2013-12-31')]
        L.append(f"stage {stage} 在池股票數 (2013-12-31 起)：min={n.min()} median={int(n.median())} max={n.max()}")
    return "\n".join(L)


# ---------------------------------------------------------------- 主程式
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'universe_raw.pkl'))
    args = ap.parse_args()

    f65, f1d = collect_sources()
    print(f"65m 來源 {len(f65)} 檔，日 K 來源 {len(f1d)} 檔")
    raw65 = {s: read_65(p) for s, p in f65.items()}
    raw1d = {s: read_1d(p) for s, p in f1d.items()}
    bars, bstats = build_bars(raw65)
    daily, dstats = build_daily(raw1d, bars)

    rf = pd.read_csv(FRED_CSV, parse_dates=['observation_date']).set_index('observation_date')['DTB3']
    rf = rf.reindex(rf.index.union(daily['close'].index)).ffill().reindex(daily['close'].index)
    spy = read_1d(SPY_CSV)['close']
    membership = read_membership()

    rep = make_report(bars, daily, bstats, dstats, membership, rf, spy)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, 'wb') as f:
        pickle.dump(dict(symbols=sorted(raw65), bars=bars, daily=daily, rf=rf, spy=spy, membership=membership,
                         meta=dict(start=str(START.date()), end=str(END.date()), bar_stats=bstats, daily_stats=dstats)), f)
    with open(os.path.join(os.path.dirname(args.out), 'build_report.txt'), 'w', encoding='utf-8') as f:
        f.write(rep)
    print(rep)
    print(f"\n已輸出 {args.out}")


if __name__ == '__main__':
    sys.exit(main())
