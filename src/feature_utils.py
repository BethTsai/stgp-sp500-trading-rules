"""
特徵檢查與診斷工具。

  select_features_exact()  依 config.features.selected_features_exact 精確挑選特徵；名稱不存在時報錯並列出相近名稱
  check_feature_scales()   確認選用的特徵皆為 RankType 且數值落在 [0, 1]，與 U(0, 1) 隨機常數的尺度一致
  feature_report()         計算各特徵與下一期報酬的橫斷面 Rank IC，並列出高度相關的特徵對

用法 (在專案根目錄)：
    python -m src.feature_utils --report                      # 分析 config.data.file_path 指定的特徵檔
    python -m src.feature_utils --report --pkl <特徵檔路徑>
"""
import argparse
import difflib
import gzip
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def select_features_exact(all_features, exact_names):
    """
    all_features : 資料裡實際存在的特徵名稱 (dict keys，已排除 close/eval_price)
    exact_names  : config['features']['selected_features_exact']
    回傳依名稱排序的 list。
    名稱不存在時拋出 ValueError，並列出相近名稱。
    """
    avail = set(all_features)
    seen, dup = set(), set()
    for n in exact_names:
        (dup if n in seen else seen).add(n)
    if dup:
        logging.warning(f"selected_features_exact 有重複名稱 (已忽略重複): {sorted(set(dup))}")
    missing = [n for n in seen if n not in avail]
    if missing:
        lines = []
        for n in sorted(missing):
            sug = difflib.get_close_matches(n, sorted(avail), n=3, cutoff=0.6)
            lines.append(f"  - '{n}' 不存在" + (f"，你是不是要找: {sug}" if sug else ""))
        raise ValueError("selected_features_exact 有名稱不在特徵矩陣中:\n" + "\n".join(lines) +
                         "\n(若是新加的特徵，請刪除舊 pkl 後重新建立)")
    return sorted(seen)


# ==========================================================================
# 特徵尺度檢查
# ==========================================================================
def check_feature_scales(feature_cols, data, strict=True, tol=1e-6):
    """
    檢查每個被選中的特徵：
      (a) 型別必須是 RankType (primitives.get_feature_type)，才有 [0,1] 的 ERC 常數可以比較；
      (b) 數值範圍必須在 [0,1] (容許 tol)。
    data : DataProvider.data (dict: 特徵名稱 -> DataFrame)。strict=True 違規就丟 ValueError。
    """
    from src.primitives import get_feature_type, RankType
    bad, rows = [], []
    for c in feature_cols:
        arr = np.asarray(data[c].values, dtype='float64')
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        ftype = get_feature_type(c)
        ok = (ftype is RankType) and lo >= -tol and hi <= 1 + tol
        rows.append((c, ftype.__name__, lo, hi, ok))
        if not ok:
            bad.append(f"{c} (型別 {ftype.__name__}, 範圍 [{lo:.3g}, {hi:.3g}])")
    logging.info("🔎 特徵尺度檢查 (需為 RankType 且落在 [0,1])：" +
                 "；".join(f"{r[0]}[{r[2]:.2f},{r[3]:.2f}]" for r in rows))
    if bad:
        msg = ("以下特徵不是 0~1 尺度，GP 的 ERC 常數 [0,1] 無法與其比較 (RSI/HV/alpha 等原始值請改用 Rank_ 版本):\n  - "
               + "\n  - ".join(bad))
        if strict:
            raise ValueError(msg)
        logging.warning(msg)
    return rows


# ==========================================================================
# 特徵診斷報告 (Rank IC + 相關性)
# ==========================================================================
def _load_pkl(path):
    with open(path, 'rb') as test_f:
        gz = test_f.read(2) == b'\x1f\x8b'
    with (gzip.open(path, 'rb') if gz else open(path, 'rb')) as f:
        return pickle.load(f)


def feature_report(pkl_path, features=None, periods=None, corr_warn=0.7, out=print):
    """
    每個特徵與「下一段持有報酬」的橫斷面 Spearman IC (每日平均後算 t 值)，並列出高度相關的特徵對。
    持有報酬 = eval_price(t+2)/eval_price(t+1) - 1  (t 收盤訊號 -> t+1 的 (H+L)/2 成交 -> 持有到 t+2)。
    """
    ds = _load_pkl(pkl_path)
    ep = ds['eval_price']
    feats = features or [k for k in ds if k not in ('close', 'eval_price')]
    ret = ep.shift(-2) / ep.shift(-1) - 1
    ret_rank = ret.rank(axis=1)
    periods = periods or [('2013-2017', '2013-01-01', '2017-12-31'), ('2018-2021', '2018-01-01', '2021-12-31'),
                          ('2022-2025', '2022-01-01', '2025-12-31')]
    out(f"{'feature':<24}" + "".join(f"{p[0]:>20}" for p in periods))
    for f in feats:
        F = ds[f].reindex(ep.index).astype('float64')
        Fr = F.rank(axis=1)
        a, b = Fr.sub(Fr.mean(axis=1), axis=0), ret_rank.sub(ret_rank.mean(axis=1), axis=0)
        ic = (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
        daily = ic.groupby(ic.index.normalize()).mean()
        cells = []
        for _, s, e in periods:
            d = daily.loc[s:e].dropna()
            cells.append(f"{d.mean():+.4f} (t={d.mean() / d.std() * np.sqrt(len(d)):+5.1f})" if len(d) > 30 else "n/a")
        out(f"{f:<24}" + "".join(f"{c:>20}" for c in cells))
    # 相關性 (橫斷面排名後的時間平均相關，抽樣加速)
    sample = ep.index[::6]
    M = pd.DataFrame({f: ds[f].reindex(sample).rank(axis=1).stack() for f in feats})
    cm = M.corr()
    out(f"\n特徵兩兩相關 |corr| >= {corr_warn} (代表資訊重複):")
    pairs = [(cm.index[i], cm.columns[j], cm.iloc[i, j]) for i in range(len(cm)) for j in range(i + 1, len(cm))
             if abs(cm.iloc[i, j]) >= corr_warn]
    for a_, b_, v in sorted(pairs, key=lambda x: -abs(x[2])):
        out(f"  {a_:<22}{b_:<22}{v:+.2f}")
    if not pairs:
        out("  (無)")
    return cm


if __name__ == "__main__":
    import sys
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--report', action='store_true', help="計算 Rank IC 與特徵相關性")
    ap.add_argument('--pkl', default=None)
    ap.add_argument('--features', nargs='*', default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    if a.report:
        feature_report(a.pkl or cfg['data']['file_path'], a.features)
