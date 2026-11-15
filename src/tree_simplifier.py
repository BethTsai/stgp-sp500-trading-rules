import sympy
from sympy import And, Or
from pathlib import Path
import re
from collections import Counter

def simplify_tree_string(tree_str):
    """
    將 GP 樹字串轉換為 SymPy 表達式並進行代數與邏輯化簡
    """
    # 建立映射字典：將 primitives.py 中的自定義算子映射為 SymPy 可理解的邏輯與數學運算
    local_dict = {
        'and_': And,
        'or_': Or,
        'gt_price': lambda x, y: x > y,
        'lt_price': lambda x, y: x < y,
        'gt_ratio': lambda x, y: x > y,
        'lt_ratio': lambda x, y: x < y,
        'gt_rank': lambda x, y: x > y,
        'lt_rank': lambda x, y: x < y,
        'gt': lambda x, y: x > y,
        'lt': lambda x, y: x < y,
        'add_price': lambda x, y: x + y,
        'sub_price': lambda x, y: x - y,
        'add_ratio': lambda x, y: x + y,
        'sub_ratio': lambda x, y: x - y,
        'add_rank': lambda x, y: x + y,
        'sub_rank': lambda x, y: x - y,
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'TrueConst': True,
        'FalseConst': False
    }
    
    try:
        # sympify 會解析字串。未定義在 local_dict 中的特徵名稱(如 alpha_10, Sharpe_20) 會自動被視為 SymPy 的 Symbol
        expr = sympy.sympify(tree_str, locals=local_dict)
        
        # 使用 sympy.simplify 進行數學與邏輯化簡 (例如 A + B - A 會化簡為 B)
        simplified_expr = sympy.simplify(expr)
        
        return str(simplified_expr)
    except Exception as e:
        return f"{tree_str} (化簡失敗: {e})"

def export_simplified_trees_to_txt(metrics_summary, output_dir_path):
    """
    讀取 metrics_summary 中各個 Run 的最佳樹，化簡後輸出到指定的資料夾中
    """
    output_file = Path(output_dir_path) / "simplified_trees.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== GP 最佳樹化簡結果 (Simplified GP Trees) ===\n\n")
        
        for m in metrics_summary:
            run_name = m.get("項目", "Unknown_Run")
            original_tree = m.get("最優公式樹結構", "")
            
            if original_tree and original_tree != "N/A":
                simplified_tree = simplify_tree_string(original_tree)
                f.write(f"[{run_name}]\n")
                f.write(f"原始樹 (Raw)    : {original_tree}\n")
                f.write(f"化簡後 (Simplified): {simplified_tree}\n")
                f.write("-" * 70 + "\n")
                
    print(f"📄 成功！已輸出化簡後的最佳樹至: {output_file.resolve()}")
    
    
def extract_features_from_tree(tree_str):
    """
    使用正規表示式從公式中萃取特徵變數名稱
    假設變數名稱為英文字母開頭，可能包含底線與數字 (例如 alpha_22, beta_10, sharpe_60)
    """
    # 抓取連續的英文字母、數字與底線
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', tree_str)
    
    # 排除常見的邏輯算子或布林常數
    ignore_list = {'And', 'Or', 'True', 'False', 'TrueConst', 'FalseConst', 'add', 'sub', 'gt', 'lt'}
    
    features = [w for w in words if w not in ignore_list]
    return features

# ==========================================================================
# 產生給 LLM 解讀公式用的 prompt 檔 (generate_llm_prompt_file) 與其輔助函式
# ==========================================================================
import numpy as np
import yaml

# 標的資訊對照表；config 內若出現表外代號，則只顯示代號
SYMBOL_INFO = {
    'AAPL': ('Apple', 'Information Technology'), 'MSFT': ('Microsoft', 'Information Technology'),
    'NVDA': ('Nvidia', 'Information Technology'), 'CRM': ('Salesforce', 'Information Technology'),
    'INTC': ('Intel', 'Information Technology'),
    'JPM': ('JPMorgan Chase', 'Financials'), 'GS': ('Goldman Sachs', 'Financials'),
    'V': ('Visa', 'Financials'), 'BAC': ('Bank of America', 'Financials'),
    'JNJ': ('Johnson & Johnson', 'Healthcare'), 'UNH': ('UnitedHealth', 'Healthcare'),
    'PFE': ('Pfizer', 'Healthcare'), 'ABT': ('Abbott Laboratories', 'Healthcare'),
    'AMZN': ('Amazon', 'Consumer Discretionary'), 'TSLA': ('Tesla', 'Consumer Discretionary'),
    'NKE': ('Nike', 'Consumer Discretionary'), 'HD': ('Home Depot', 'Consumer Discretionary'),
    'KO': ('Coca-Cola', 'Consumer Staples'), 'PG': ('Procter & Gamble', 'Consumer Staples'),
    'PEP': ('PepsiCo', 'Consumer Staples'),
    'BA': ('Boeing', 'Industrials'), 'CAT': ('Caterpillar', 'Industrials'),
    'UPS': ('United Parcel Service', 'Industrials'),
    'XOM': ('ExxonMobil', 'Energy'), 'CVX': ('Chevron', 'Energy'),
    'GOOGL': ('Alphabet', 'Communication Services'), 'NFLX': ('Netflix', 'Communication Services'),
    'DUK': ('Duke Energy', 'Utilities'), 'PLD': ('Prologis', 'Real Estate'),
    'LIN': ('Linde', 'Materials'),
}

# 特徵家族的金融意涵 (僅在該家族實際出現在特徵池 / 公式中時才寫進 prompt)
FAMILY_DESC = {
    'alpha': "CAPM alpha：資產相對大盤回歸所得的「超額報酬」，反映個股自身的強勢動能。數值高代表近 N 日具備超越大盤的報酬。",
    'beta': "CAPM beta：資產對大盤的「波動敏感度 / 系統性風險暴露」。數值高代表行情好時漲更多、行情差時跌更深。",
    'sharpe': "Sharpe ratio：近 N 日「每單位風險所獲得的超額報酬」，衡量風險調整後的報酬效率。",
}

# 天期分群 (以日為單位，與公式中的 alpha10~alpha60 等數字對應)
HORIZON_BUCKETS = [("短天期 (10~25日)", 10, 25), ("中天期 (26~45日)", 26, 45), ("長天期 (46~60日)", 46, 60)]


def _load_config(config, output_dir_path):
    """
    取得本次實驗所使用的設定。優先順序：
    呼叫端傳入 > 實驗資料夾內 run_xx/temp_config.yaml (當次實驗的真實設定) > 專案根目錄 config.yaml
    """
    if isinstance(config, dict):
        return config
    candidates = sorted(Path(output_dir_path).glob("run_*/temp_config.yaml"))
    candidates.append(Path(__file__).resolve().parents[1] / "config.yaml")
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            continue
    return {}


def _unique_features(tree_str):
    """
    回傳單一公式中不重複的特徵名稱集合。
    同一棵樹內重複出現的特徵只計一次，科學記號常數 (如 1e-05) 不會被誤判為特徵。
    """
    cleaned = re.sub(r'(?<![a-zA-Z_\d.])\d+\.?\d*[eE][-+]?\d+', ' ', tree_str)   # 先移除科學記號數字
    cleaned = re.sub(r'(?<![a-zA-Z_\d.])\d+\.?\d*', ' ', cleaned)   # 再移除獨立數字常數 (不可吃掉 alpha10 的 10)
    return {w for w in extract_features_from_tree(cleaned) if re.search(r'\d', w)}


def _feature_family(feat):
    """alpha41 -> 'alpha'；Rank_ROC6 -> 'Rank_ROC'"""
    return re.sub(r'\d+$', '', feat)


def _feature_days(feat):
    m = re.search(r'(\d+)$', feat)
    return int(m.group(1)) if m else None


def _pct(count, total):
    return f"{(count / total * 100) if total else 0:.1f}%"


def _freq_lines(runs_features, top_n=10):
    """runs_features: list[set]，回傳「出現在幾 % 的 Run」的條列文字"""
    total = len(runs_features)
    counter = Counter(f for feats in runs_features for f in feats)
    if not counter:
        return "(無)\n"
    return "".join(f"- {feat}: 出現在 {cnt}/{total} 個 Run ({_pct(cnt, total)})\n"
                   for feat, cnt in counter.most_common(top_n))


def _structure_stats(runs_features):
    """家族 / 天期分群的使用比例，以及每棵樹平均使用的特徵數"""
    total = len(runs_features)
    if total == 0:
        return "(無)\n"
    text = ""
    fam_runs = Counter()
    for feats in runs_features:
        for fam in {_feature_family(f) for f in feats}:
            fam_runs[fam] += 1
    text += "- 各特徵家族被使用的 Run 比例：" + "、".join(
        f"{fam} {_pct(c, total)}" for fam, c in fam_runs.most_common()) + "\n"
    bucket_runs = Counter()
    for feats in runs_features:
        used = set()
        for f in feats:
            d = _feature_days(f)
            for name, lo, hi in HORIZON_BUCKETS:
                if d is not None and lo <= d <= hi:
                    used.add(name)
        for name in used:
            bucket_runs[name] += 1
    text += "- 各天期分群被使用的 Run 比例：" + "、".join(
        f"{name} {_pct(bucket_runs[name], total)}" for name, _, _ in HORIZON_BUCKETS) + "\n"
    mixed = sum(1 for feats in runs_features
                if len({name for f in feats for name, lo, hi in HORIZON_BUCKETS
                        if _feature_days(f) is not None and lo <= _feature_days(f) <= hi}) >= 2)
    text += f"- 同時混用兩種以上天期分群 (Multi-Timeframe) 的 Run 比例：{_pct(mixed, total)}\n"
    text += f"- 每棵樹平均使用的不重複特徵數：{np.mean([len(f) for f in runs_features]):.1f}\n"
    return text


def _describe_periods(config, test_times, train_times=None):
    """還原 DataProvider 的時間切割邏輯，取得訓練 / 測試區間與標的清單"""
    data_conf = config.get('data', {})
    evo = config.get('evolution', {})
    # 與 DataProvider 相同：SOO 看 soo_training_mode，MOO 看 moo_selection_method
    if evo.get('objective_mode', 'single') == 'single':
        use_oos = evo.get('soo_training_mode', 'regular') == 'oos'
    else:
        use_oos = evo.get('moo_selection_method', 'topsis') == 'oos'
    use_oos = use_oos and 'oos_split' in data_conf
    if use_oos:
        o = data_conf['oos_split']
        train_range = (o.get('train_start'), o.get('train_end'))
        val_range = (o.get('val_start'), o.get('val_end'))
    else:
        train_range = (data_conf.get('train_start'), data_conf.get('train_end'))
        val_range = None
    if train_times is not None and len(train_times) > 0:
        train_range = (train_times[0].strftime('%Y-%m-%d'), train_times[-1].strftime('%Y-%m-%d'))
    return train_range, val_range


def _universe_text(config):
    data_conf = config.get('data', {})
    train_syms = data_conf.get('train_symbols') or []
    if train_syms == 'all':
        return "- 訓練標的：pkl 內全部股票 (train_symbols: all)\n"
    test_only = data_conf.get('test_only_symbols') or []
    if not train_syms and not test_only:
        return "(未能從設定檔取得標的清單)\n"

    def fmt(sym):
        name, sector = SYMBOL_INFO.get(sym, (None, None))
        return f"{sym} ({name}, {sector})" if name else sym

    text = f"- 訓練標的 {len(train_syms)} 檔：" + "、".join(fmt(s) for s in train_syms) + "\n"
    if test_only:
        text += f"- 僅測試用標的 {len(test_only)} 檔 (演化時未見過)：" + "、".join(fmt(s) for s in test_only) + "\n"
    text += f"- 樣本外測試實際使用標的：訓練標的 + 僅測試標的，共 {len(train_syms) + len(test_only)} 檔\n"
    sectors = Counter(SYMBOL_INFO[s][1] for s in train_syms + test_only if s in SYMBOL_INFO)
    if sectors:
        text += "- 產業分布：" + "、".join(f"{k} {v} 檔" for k, v in sectors.most_common()) + "\n"
    return text


def generate_llm_prompt_file(metrics_summary, test_times, output_dir_path, baseline_metrics=None,
                             config=None, train_times=None):
    """
    產生給 LLM 解讀公式用的 prompt.txt。內容包含標的清單、訓練 / 測試區間、回測機制與適應度定義、
    演化超參數、特徵說明、各 run 的 Train / Test 績效與風險指標，以及相對 Buy & Hold 的勝負分群統計。
    特徵使用頻率以「出現於幾 % 的 run」計算。

    config / train_times 皆為選填；未提供時會自動讀取 output_dir_path/run_xx/temp_config.yaml 或根目錄 config.yaml。
    """
    output_file = Path(output_dir_path) / "prompt.txt"
    config = _load_config(config, output_dir_path)
    evo = config.get('evolution', {})
    trd = config.get('trading', {})
    rsf = config.get('rsfgp', {})
    fit = config.get('fitness') or {}
    bars_per_year = trd.get('bars_per_year', 1512)

    # ---------- 1. 期間 ----------
    start_date = test_times[0].strftime('%Y-%m')
    end_date = test_times[-1].strftime('%Y-%m')
    test_years = len(test_times) / bars_per_year if bars_per_year else 0
    train_range, val_range = _describe_periods(config, test_times, train_times)

    # ---------- 2. 整理每個 Run ----------
    sorted_metrics = sorted(metrics_summary, key=lambda x: x.get("Test_Sterling", 0), reverse=True)
    bh_sterling = baseline_metrics.get("bh_sterling") if baseline_metrics else None

    runs = []
    for m in sorted_metrics:
        original_tree = m.get("最優公式樹結構", "")
        if not original_tree or original_tree == "N/A":
            continue
        simplified = simplify_tree_string(original_tree)
        ind = m.get("個體物件")
        runs.append({
            "rank": len(runs) + 1,
            "run_name": m.get("項目", f"Run_{len(runs) + 1}"),
            "m": m,
            "tree": simplified,
            "nodes": len(ind) if ind is not None else None,
            "features": _unique_features(simplified),
            "beat": (m.get("Test_Sterling", 0) > bh_sterling) if bh_sterling is not None else None,
        })
    total_runs = len(runs)
    all_run_features = [r["features"] for r in runs]
    families_in_trees = sorted({_feature_family(f) for fs in all_run_features for f in fs})

    # ---------- 3. 特徵池說明 (依 config 的 selected_features_exact 動態決定，避免描述未啟用的特徵) ----------
    selected = config.get('features', {}).get('selected_features_exact') or []
    pool_families = [k for k in FAMILY_DESC if (not selected or any(k in s for s in selected) or k in families_in_trees)]
    # 公式中實際出現、但說明表沒有的家族 (例如之後加入 ROC / RSI / Rank_ 等)，也提醒 LLM
    extra_families = [f for f in families_in_trees if f not in FAMILY_DESC and f not in pool_families]
    feature_text = "".join(f"- {FAMILY_DESC[k]}\n" for k in pool_families)
    if extra_families:
        feature_text += "- 公式中另出現下列家族 (請依名稱推論其意涵，並於回答中註明為推論)：" + "、".join(extra_families) + "\n"
    excluded = [k for k in FAMILY_DESC if k not in pool_families]
    if excluded:
        feature_text += f"- 注意：{'、'.join(excluded)} 本次實驗『未』放入特徵池，公式中不會出現，請勿討論。\n"

    # ---------- 4. 整體統計 ----------
    def col(key):
        return np.array([r["m"].get(key, np.nan) for r in runs], dtype=float)

    def ms(arr, pct=False):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return "N/A"
        k = 100 if pct else 1
        suffix = "%" if pct else ""
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        return f"{np.mean(arr) * k:.2f}{suffix} ± {std * k:.2f}{suffix}"

    stats_text = f"- 樣本外 Sterling Ratio (Mean ± Std)：{ms(col('Test_Sterling'))}\n"
    stats_text += f"- 樣本外年化報酬率：{ms(col('Test_Return'), True)}\n"
    stats_text += f"- 樣本外最大回撤 (MDD)：{ms(col('Test_Drawdown'), True)}\n"
    stats_text += f"- 樣本外年化波動率：{ms(col('Test_StdDev'), True)}\n"
    stats_text += f"- 樣本外累計換手量：{ms(col('總交易次數 (Turnover)'))}\n"
    stats_text += f"- 訓練期 Sterling Ratio (Mean ± Std)：{ms(col('Train_Sterling'))}\n"
    stats_text += f"- 訓練期年化報酬率：{ms(col('Train_Return'), True)}\n"
    tr, te = col('Train_Sterling'), col('Test_Sterling')
    if not np.all(np.isnan(tr)):
        stats_text += (f"- 訓練→測試平均 Sterling 衰減：{np.nanmean(tr):.4f} → {np.nanmean(te):.4f}；"
                       f"測試期 Sterling 低於訓練期的 Run 比例：{_pct(int(np.sum(te < tr)), total_runs)}\n")

    # ---------- 5. Buy & Hold 基準線與勝負分群 ----------
    bh_text = ""
    group_text = ""
    if baseline_metrics:
        bh_ror_pct = baseline_metrics.get("bh_ror", 0) * 100
        bh_text += "【Buy & Hold 基準線】\n"
        bh_text += ("定義：在樣本外全期間，對『所有測試標的』等權重持有、不擇時 (訊號恆為 True)；"
                    "為本研究池內標的的等權重組合，並非 S&P 500 指數。真正的 Buy & Hold：第 1 根 K 棒等權買入後不再平衡，進場與出場各收一次手續費。\n")
        bh_text += f"Sterling Ratio: {bh_sterling:.4f} | 年化報酬率: {bh_ror_pct:.2f}%"
        if baseline_metrics.get("bh_mdd") is not None:
            bh_text += f" | 最大回撤: {baseline_metrics['bh_mdd'] * 100:.2f}%"
        bh_text += "\n\n"

        beat = [r for r in runs if r["beat"]]
        lose = [r for r in runs if not r["beat"]]
        group_text += f"以 Sterling Ratio 對比 Buy & Hold (判準：{bh_sterling:.4f})，已預先分群，請直接採用此分群，不必自行重算：\n"
        group_text += f"- 勝過 Buy & Hold：{len(beat)}/{total_runs} 個 Run ({_pct(len(beat), total_runs)})"
        group_text += f"：{'、'.join(r['run_name'] for r in beat) if beat else '無'}\n"
        group_text += f"- 落後 Buy & Hold：{len(lose)}/{total_runs} 個 Run ({_pct(len(lose), total_runs)})"
        group_text += f"：{'、'.join(r['run_name'] for r in lose) if lose else '無'}\n\n"
        group_text += f"▶ 勝過組的高頻特徵 (樣本數 {len(beat)})：\n{_freq_lines([r['features'] for r in beat])}"
        group_text += f"{_structure_stats([r['features'] for r in beat])}\n"
        group_text += f"▶ 落後組的高頻特徵 (樣本數 {len(lose)})：\n{_freq_lines([r['features'] for r in lose])}"
        group_text += f"{_structure_stats([r['features'] for r in lose])}\n"
        group_text += "(樣本數偏小時，請避免過度解讀兩組頻率差異，並說明其統計上的不確定性。)\n"
    else:
        group_text = "(本次未提供 Buy & Hold 基準資訊，請僅就公式與績效本身分析。)\n"

    # ---------- 6. 各 Run 公式 ----------
    trees_text = ""
    for r in runs:
        m = r["m"]
        tag = "" if r["beat"] is None else (" | 勝過B&H" if r["beat"] else " | 落後B&H")
        trees_text += f"[排名 {r['rank']} | {r['run_name']}{tag}]\n"
        trees_text += (f"  樣本外：Sterling {m.get('Test_Sterling', 0):.4f} | 年化報酬 {m.get('Test_Return', 0) * 100:.2f}% | "
                       f"MDD {m.get('Test_Drawdown', 0) * 100:.2f}% | 年化波動 {m.get('Test_StdDev', 0) * 100:.2f}% | "
                       f"累計換手量 {m.get('總交易次數 (Turnover)', 0):.1f}\n")
        trees_text += (f"  訓練期：Sterling {m.get('Train_Sterling', 0):.4f} | 年化報酬 {m.get('Train_Return', 0) * 100:.2f}% | "
                       f"年化波動 {m.get('Train_StdDev', 0) * 100:.2f}%\n")
        trees_text += f"  複雜度：原始樹節點數 {r['nodes'] if r['nodes'] is not None else 'N/A'} | 不重複特徵數 {len(r['features'])}\n"
        trees_text += f"  公式: {r['tree']}\n\n"

    # ---------- 7. 設定描述 ----------
    obj_mode = evo.get('objective_mode', 'single')
    if obj_mode == 'single':
        objective_text = "單目標：最大化 Sterling Ratio。"
    else:
        objective_text = (f"多目標：同時最大化 Sterling Ratio 與 {evo.get('multi_target', 'skewness')}；"
                          f"自 Pareto 前緣挑選代表策略的方法為 {evo.get('moo_selection_method', 'topsis')}。")
    fee = trd.get('transaction_fee', 0.003)
    fee_mode = trd.get('fee_mode', 'signal_driven')
    fee_mode_text = {"rebalanced": "每根 K 棒都把價格飄移後的持倉調回目標權重，並對權重差收費 (換手較頻繁)",
                     "signal_driven": "僅在訊號改變時才收取調倉費用",
                     "drift": "只在持股組合改變的那根 K 棒把持股調回等權並收費，其餘時間不交易、權重隨價格自然飄移 (全部持有即等同 Buy & Hold)"}.get(fee_mode, fee_mode)
    agg = fit.get('aggregation', 'mean_std')
    std_k = fit.get('std_k', 0.5)
    agg_text = {"mean_std": f"『平均 − {std_k} × 標準差』聚合（單一窗口特別高會被標準差扣回）",
                "mean_downside": f"『平均 − {std_k} × 下方標準差』聚合（下方標準差只計算低於門檻 {fit.get('downside_target', 'mean')} 的窗口，表現特別好的窗口不扣分）"
                }.get(agg, f"『{agg}』聚合")
    if fit.get('relative_to_baseline'):
        base_name = {"bh": "真正 Buy & Hold（窗口開頭等權買入、之後不再平衡）",
                     "ew_same": "全部持有、使用同一套交易模式的等權組合"}.get(fit.get('baseline', 'ew_same'), fit.get('baseline'))
        baseline_text = f"每個窗口的分數再減去同窗口{base_name}的同一指標（即超額表現，基準本身 = 0），"
    else:
        baseline_text = ""
    if rsf.get('window_mode', 'bars') == 'calendar_half':
        window_text = "以日曆半年（1～6 月、7～12 月）為一個窗口，每窗口約 750 根 K 線"
    else:
        window_text = f"每窗口 {rsf.get('window_bars', 756)} 根 K 線，約半年"
    alloc = trd.get('allocation_mode', 'dynamic')
    alloc_text = {"dynamic": "動態等權重：當下訊號為 True 的標的平分資金；全部為 False 則全數空手 (現金)",
                  "fixed": "固定權重：每檔標的上限 1/N，訊號為 False 的部位以現金持有"}.get(alloc, alloc)

    pool_note = (f"；特徵池僅限 {'、'.join(selected)} 相關特徵" if selected else "；特徵池為全部可用特徵")
    train_line = f"{train_range[0]} 至 {train_range[1]}" if train_range[0] else "N/A"
    val_line = f"\n- 驗證期 (Validation，用於 OOS 挑選策略)：{val_range[0]} 至 {val_range[1]}" if val_range else ""

    prompt_content = f"""你現在是一位頂尖的量化金融研究員與經濟學家。我們使用遺傳規劃（Genetic Programming, GP）演算法，在美股 65 分鐘 K 線上進行了 {total_runs} 次獨立演化（每次使用不同亂數種子），並產出了 {total_runs} 個交易訊號公式（每個 Run 取最終一棵最佳樹）。

請根據以下完整資訊進行深度的金融分析。請區分「由數據直接支持的結論」與「你的推測」，並在推測處明確標註。

========================================
【一、研究標的 (Universe)】
{_universe_text(config)}
- 每檔標的獨立套用同一條公式：公式對該標的當下的特徵值輸出 True/False，True 代表「持有該檔」，並非跨標的排名或選股。
- 僅做多、無放空、無槓桿。標的皆為現今仍存續的大型權值股，存在倖存者偏誤 (Survivorship Bias) 的可能。

【二、資料期間與資料切割】
- 訓練期（In-Sample，演化期間）：{train_line}{val_line}
- 樣本外測試期（Out-of-Sample）：{start_date} 至 {end_date}（約 {test_years:.2f} 年）
- 演化時從未見過測試期資料。
(請利用你的搜尋能力，分別查詢並分析『訓練期』與『測試期』的全球總體經濟環境（升降息循環、通膨、重大事件）、美股牛熊與波動狀態、以及各產業表現；並特別比較兩段期間的『市場環境差異（Regime Shift）』，作為後續分析的基礎。)

【三、特徵定義】
特徵皆以『日線』資料計算（視窗 10~60 日），再依日期對齊到該日所有 65 分鐘 K 線上，因此同一個交易日內（約 6 根 K 線）特徵值固定不變，訊號變動主要來自日與日之間的更新。公式中的 alpha41 即為『41 日視窗的 alpha』，其餘類推。
{feature_text}- 數值尺度：alpha / sharpe 的數值多在小數點後兩位上下，公式中的門檻常數（如 0.07）即與此尺度對應；請以此判斷門檻的相對大小。

【四、GP 演化設定】
- 目標函數：{objective_text}
- 族群 {evo.get('pop_size')}、世代數 {evo.get('max_gen')}、語法樹最大深度 {evo.get('max_depth')}、交配率 {evo.get('cxpb')}、突變率 {evo.get('mutpb')}。
- 可用算子（強型別）：邏輯 AND / OR；比較 > 與 <；加法與減法；隨機常數（ratio 型常數範圍約 -1~1）。沒有乘除、沒有 if-else{pool_note}，因此公式本質上是『特徵的線性組合與門檻比較，再以 AND/OR 組合』。
- 公式已經過 SymPy 代數化簡（如同類項合併成 2*alpha34、重複子句被消除），所以公式長度已不等於原始節點數，請參考每個 Run 標示的『原始樹節點數』。
- 適應度 = 各回測窗口（{window_text}，互不重疊）的『窗口報酬 / (最大回撤 + 10%)』，{baseline_text}以{agg_text}，再扣除『每個節點 {fit.get('length_penalty', 0.002)}』的長度懲罰；若窗口內累計換手量低於 {fit.get('min_turnover', 4.0)}，或平均持倉比例高於 {fit.get('max_long_ratio', 0.95)} / 低於 {fit.get('min_long_ratio', 0.05)}（用以淘汰近似 Buy & Hold 或幾乎不持有的策略），該窗口給予最低分。因此演化本身就偏好『較短、且會交易』的公式。

【五、回測機制與績效指標定義】
- 訊號時序：t 期收盤後產生訊號，於 t+1 期執行，以該根 K 線的 (最高價+最低價)/2 作為成交價；首根無部位，回測最後一根強制平倉。
- 資金配置：{alloc_text}。
- 交易成本：單邊手續費率 {fee * 100:.2f}%（已涵蓋手續費與滑價的簡化假設），收費方式 fee_mode='{fee_mode}'：{fee_mode_text}。
- Sterling Ratio = 年化報酬率 / (最大回撤 + 10%)；年化以每年 {bars_per_year} 根 65 分鐘 K 線換算。
- 『累計換手量』為各期權重變動絕對值的加總（以整體資金為 1），並非下單筆數；數值越大代表交易越頻繁、成本拖累越大。
- 『年化報酬率』『年化波動率』『MDD』皆為樣本外全期間的單一路徑結果。

========================================
【六、{total_runs} 次獨立演化的整體績效統計】
{stats_text}
{bh_text}【七、勝過 / 落後 Buy & Hold 的分群與特徵統計】
{group_text}
【八、{total_runs} 次獨立演化統計之高頻特徵（全體）】
(頻率定義：該特徵出現在多少比例的 Run 中；同一棵樹內重複出現只算一次。每個特徵家族有 51 個可選天期，若為隨機選用，單一特徵的出現率應偏低，因此高頻特徵值得留意。)
{_freq_lines(all_run_features)}
{_structure_stats(all_run_features)}
【九、樣本外績效排名之簡化 GP 公式（依 Sterling Ratio 由高至低）】
(公式結果為 True 代表持有該檔標的。)
{trees_text}
========================================
【請回答以下問題】
1. 市場環境與高頻特徵解析：
根據你搜尋到的訓練期與 {start_date} 至 {end_date} 的總體經濟與市場環境（請簡述趨勢、波動、利率與重大事件），解釋為何 GP 會高頻選用上述特徵與天期？這些特徵反映了當時市場何種交易行為或情緒？訓練期與測試期的 Regime 差異，對策略的樣本外表現有什麼影響？

2. 群體對比與隱藏 Pattern 探勘（以 Buy & Hold 為分水嶺）：
請採用【七】的預先分群：
- **負面陷阱（Negative Patterns）：** 落後組偏好哪些特徵／天期／結構？為何容易失效或過度交易？
- **勝率組合模版（Positive Structural Patterns）：** 勝過組是否存在跨天數但結構固定的運算模式（舉例：alpha_X + sharpe_X 同天數加總、長短天期 alpha 的雙重確認、alpha 與 beta 的對沖結構；若上述結構不存在，請直接說明並描述實際看到的結構）？即使具體天數被替換，核心數學結構與金融邏輯為何？
- **金融意涵解讀：** 這些好與壞的 Pattern 分別代表何種市場邏輯或行為金融學現象（動量、均值回歸、風險溢酬等）？

3. 頂尖策略群解構與時間天數搭配（Time Horizon Alignment）：
針對排名前列（Rank 1 ~ 3）的公式：
- **決策邏輯：** 用白話文拆解買進／持倉／空手的觸發條件，說明其屬順勢、逆勢或風險控管導向。
- **時間天數搭配：** 分析不同天期混搭的合理性，以及多時間週期如何達到濾波、防偽訊號與風控作用。
- **常數門檻：** 公式中的常數門檻（如 0.07）在特徵尺度下代表多嚴格的條件？它們是否過度貼合測試前的特定資料？

4. 績效品質與歸因（請善用各 Run 的 MDD、波動、換手量）：
- 勝過 Buy & Hold 的策略，優勢主要來自『報酬更高』還是『回撤更小／波動更低』？是否可能只是因為較多時間持有現金（降低市場曝險）所致？
- 以累計換手量與 {fee * 100:.2f}% 的手續費假設，交易成本對各策略的侵蝕程度為何？績效最佳者是否同時是高換手策略？
- 相較僅持有現金或僅靠 beta 曝險，策略是否有可信的『選股／擇時』能力？

5. 過度擬合與穩健性檢查：
- 比較各 Run 的訓練期 vs 樣本外 Sterling 與報酬，哪些 Run 衰減最明顯？衰減程度與公式複雜度（節點數／特徵數）是否有關？
- 在只有 {total_runs} 個 Run、單一樣本外路徑的情況下，上述結論的統計信心有多高？哪些結論應視為『假說』而非『結論』？
- 日線特徵對齊到 65 分鐘 K 線時，是否可能存在前視偏誤或資訊落後問題？請列出需要檢查的項目。
- 標的組成（現今大型權值股、倖存者偏誤）與測試期過短，對結論有何影響？

6. 基金經理人實務操作與構建建議：
- **指標黑白名單：** 根據本次實驗，哪些特徵／天期／組合結構應採用，哪些應列入黑名單？
- **實盤執行建議與風險警告：** 如何將這些規律組合成更穩健的交易系統？請考量滑價、手續費、流動性、極端行情與市場 Regime 轉換。
- **後續實驗建議：** 請提出 3~5 個可驗證的後續實驗（例如：特徵消融、換手懲罰、滾動式 Walk-Forward、加入價格或成交量特徵、擴充標的），並說明各自要驗證的假說。

【回答格式要求】
請使用條列與表格整理，先給出 5 點以內的重點摘要（Executive Summary），再依上述問題逐項作答；引用公式時請標明 Run 編號。
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(prompt_content)

    print(f"🤖 成功！已產生供 LLM 分析的提示詞文件至: {output_file.resolve()}")
