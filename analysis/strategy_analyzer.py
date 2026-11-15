"""
Strategy Analyzer (策略視覺化與分析工具)
讀取特定世代之序列化模型 (.pkl)，解析其遺傳規劃 (GP) 語法樹公式，
並依據使用者指定之時間區間 (Train / Test) 執行向量化回測與資金曲線繪製。
"""
import pickle
import gzip
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 讓 src 可被 import (請在專案根目錄執行)

# 反序列化 pkl 內的 DEAP 個體前，必須先匯入 DEAP 並建立 creator 類別
from deap import creator, base, gp
from src.primitives import create_primitive_set
from src.data_provider import DataProvider
from src.evaluator import FitnessEvaluator


# 依 config 的 weights 建立 creator 類別 (反序列化 pkl 需要)
def init_deap_from_config():
    import yaml
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            w = tuple(cfg.get('evolution', {}).get('weights', [1.0]))
    except Exception:
        w = (1.0,)
        
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=w)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

init_deap_from_config()


def load_config(config_path="config.yaml"):
    """載入 YAML 系統配置檔"""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"❌ 找不到設定檔 {config_path}，請確認檔案路徑。")
        exit(1)

def main():
    print("🔍 啟動策略視覺化與分析工具 (Model Visualizer)...\n")

    config = load_config()

    mode_str = config['evolution'].get('objective_mode', 'single').upper()
    multi_target = config['evolution'].get('multi_target', 'skewness').capitalize()
    _ts = config['data'].get('train_symbols', [])
    num_stocks = _ts if _ts == 'all' else len(_ts)   # 'all' = pkl 內全部股票

    # 決定顯示用的標籤
    opt_label = "MOO" if mode_str == "MULTI" else "SOO"
    target_label = f"Sterling & {multi_target}" if mode_str == "MULTI" else "Sterling"

    # ==========================================
    # 1. 選擇模式與世代 (簡化為 Train / Test)
    # ==========================================
    print("請選擇回測區間模式：")
    print("  [1] Train (樣本內訓練期 - 僅評估 20 支訓練股)")
    print("  [2] Test  (樣本外測試期 - 評估全體 30 支股票)")
    mode_input = input("👉 請輸入代碼 (1/2): ").strip()
    selected_mode = 'train' if mode_input == '1' else 'test'

    # 針對 Test 模式，詢問標的範圍
    test_subset = 'all'
    if selected_mode == 'test':
        print("\n請選擇 Test 要評估的股票：")
        print("  [1] 僅評估 Training 的 20 支股票")
        print("  [2] 評估全體 30 支股票 (包含 10 支未參與訓練的股票)")
        subset_input = input("👉 請輸入代碼 (1/2): ").strip()
        test_subset = 'train_only' if subset_input == '1' else 'all'

    gen_input = input("\n👉 請輸入欲解析之「世代數字」 (例如 50): ").strip()
    try:
        gen_num = int(gen_input)
    except ValueError:
        print("❌ 輸入無效，請輸入整數。")
        exit(1)

    checkpoint_dir = Path(config.get('system', {}).get('checkpoint_dir', 'checkpoint_multi30_v3'))
    model_path = checkpoint_dir / f"best_model_gen_{gen_num}.pkl"

    if not model_path.exists():
        print(f"❌ 找不到模型檔案: {model_path}")
        exit(1)

    # 以檔頭 magic number 判斷是否為 gzip 壓縮
    with open(model_path, "rb") as test_f:
        is_gzipped = test_f.read(2) == b'\x1f\x8b'

    # ==========================================
    # 2. 載入模型與建立環境
    # ==========================================
    try:
        if is_gzipped:
            with gzip.open(model_path, "rb") as f:
                data = pickle.load(f)
            print(f"📦 成功載入 GZIP 壓縮模型: {model_path.name}")
        else:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            print(f"📦 成功載入一般模型: {model_path.name}")
    except Exception as e:
        print(f"❌ 載入模型時發生錯誤: {e}")
        exit(1)
        
    pareto_front = None
    if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'fitness'):
        # MOO 存的是 Pareto front (個體 list)
        pareto_front = data
        
        print(f"📦 偵測到多目標優化模型，Pareto Front 共包含 {len(data)} 個解。")
        method = config.get('evolution', {}).get('moo_selection_method', 'topsis')
        print(f"🎯 正在執行後置篩選法: 【{method.upper()}】")
        if method == 'knee_point':
            best_ind = select_by_knee_point(data)
        elif method == 'oos':
            # 傳入 config 以利 DataProvider 切換為 oos_split 時間
            best_ind = select_by_oos(data, config)
        else:
            best_ind = select_by_topsis(data)
            
        print(f"🏆 篩選完成！代表策略在訓練期的適應度為: {best_ind.fitness.values}")
        
    else:
        # SOO 存的是單一個體
        best_ind = data
        
    if pareto_front is not None:
        print(f"📊 成功讀取 Pareto Front，共包含 {len(pareto_front)} 個非支配策略！")
    else:
        print(f"沒有pareto front!")

    print(f"\n======================================================================")
    print(f"🏆 【第 {gen_num} 代 - AI 演化之最佳交易邏輯公式】")
    print(f"======================================================================")
    print(str(best_ind))
    print(f"======================================================================\n")

    dp_mode = 'test' if selected_mode == 'test' else 'train'
    dp = DataProvider(config=config, mode=dp_mode)

    evaluator = FitnessEvaluator(data_provider=dp, config=config)
    dataset = dp.data
    times = dataset['close'].index

    # ==========================================
    # 3. 依區間與標的切出價格矩陣
    # ==========================================
    train_symbols = config['data'].get('train_symbols', [])
    test_only_symbols = config['data'].get('test_only_symbols', [])
    available_symbols = dataset['close'].columns.tolist()
    if train_symbols == 'all':
        train_symbols = available_symbols

    if selected_mode == 'train':
        target_symbols = train_symbols
        start_date = pd.to_datetime(config['data'].get('train_start', '2014-01-01'))
        end_date = pd.to_datetime(config['data'].get('train_end', '2018-06-30'))
    else: # test
        if test_subset == 'train_only':
            target_symbols = train_symbols
        else:
            target_symbols = train_symbols + test_only_symbols
        start_date = pd.to_datetime(config['data'].get('test_start', '2018-07-01'))
        end_date = pd.to_datetime(config['data'].get('test_end', '2019-12-31'))

    # 保留存在於 dataset 的股票，並找出對應的 column indices
    final_symbols = [s for s in target_symbols if s in available_symbols]
    symbol_indices = [available_symbols.index(s) for s in final_symbols]

    # DataProvider 已依 mode / oos_split 切好時間區間，這裡以其實際範圍為準，
    # 避免 oos 模式下 config 的 train_start/test_start 與 oos_split 不一致造成空資料
    start_date, end_date = times.min(), times.max()

    time_mask = (times >= start_date) & (times <= end_date)
    times_sliced = times[time_mask]
    
    # 價格矩陣 (時間長度, 股票數)
    close_prices_sliced = dataset['close'].values[time_mask][:, symbol_indices]
    eval_prices_sliced = dp.full_eval_price[time_mask][:, symbol_indices]
    market_returns_sliced = dp.full_market_returns[time_mask][:, symbol_indices]
    active_sliced = dp.full_active[time_mask][:, symbol_indices] if dp.full_active is not None else None
 
    # ==========================================
    # 4. 建構 Safe Context 並執行訊號解析
    # ==========================================
    safe_context = {
        'and_': np.logical_and,
        'or_': np.logical_or,
        # float
        'gt': np.greater, 'lt': np.less, 'add': np.add, 'sub': np.subtract,
        # 價格型
        'gt_price': np.greater, 'lt_price': np.less, 'add_price': np.add, 'sub_price': np.subtract,
        # 比率型
        'gt_ratio': np.greater, 'lt_ratio': np.less, 'add_ratio': np.add, 'sub_ratio': np.subtract,
        # 排名型
        'gt_rank': np.greater, 'lt_rank': np.less, 'add_rank': np.add, 'sub_rank': np.subtract,
        # 常數
        'TrueConst': True, 'FalseConst': False
    }

    # 將需要的特徵欄位對齊切片後放入 context
    for feat in dp.feature_cols:
        if feat in dataset:
            safe_context[feat] = dataset[feat].values[time_mask][:, symbol_indices]

    expr_str = str(best_ind)
    try:
        signals = eval(expr_str, {}, safe_context)
    except Exception as e:
        print(f"❌ 執行公式時發生錯誤: {e}")
        exit(1)

    # 公式為常數 True/False 時，廣播成 (T, N)
    if not isinstance(signals, np.ndarray):
        signals = np.full(close_prices_sliced.shape, signals, dtype=bool)

    # ==========================================
    # 5. 執行向量化回測 (總體績效)
    # ==========================================
    equity_curve, turnover = evaluator._run_state_machine(signals, close_prices_sliced, eval_prices_sliced, market_returns_sliced, active=active_sliced)
    
    # 1. 逐根報酬
    returns_s = pd.Series(np.diff(equity_curve, prepend=1.0) / (np.roll(equity_curve, 1) + 1e-9))
    returns_s.iloc[0] = 0

    # 2. 計算累積 Sterling (每一點代表從開始到當天的表現)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / (running_max + 1e-9)
    running_mdd = np.maximum.accumulate(drawdowns)
    
    # 計算當下的年化收益 (假設 65min K線，每天 6 根)
    bars_per_year = config.get('trading', {}).get('bars_per_year', 252)
    progress_days = np.arange(1, len(equity_curve) + 1)
    running_ror = (equity_curve) ** ((bars_per_year) / progress_days) - 1.0
    running_sterling = running_ror / (running_mdd + 0.1)

    # 3. 計算近一年 (約 1512 根 K 棒) 的滾動偏度 (如果是 MOO 模式)
    rolling_skew = returns_s.rolling(window=1512, min_periods=10).skew().fillna(0).values
    
    sterling = evaluator._calculate_sterling_ratio(equity_curve)

    print(f"📊 【{selected_mode.upper()} 區間回測報告 ({len(final_symbols)} 支股票參與)】")
    print(f"總交易周轉次數 (Turnover): {turnover} 次")
    print(f"區間最終淨值: {equity_curve[-1]:.4f} (初始為 1.0)")
    print(f"Sterling Ratio: {sterling:.4f}\n")

    # ==========================================
    # 6. 計算單股獨立績效 (僅供個股視圖視覺化參考)
    # ==========================================
    # 全 True 訊號：期初買入並持有至期末
    bnh_signals = np.ones_like(close_prices_sliced, dtype=bool)
    
    # 經過與策略相同的狀態機，資金分配與手續費的算法一致
    bnh_total_equity, bnh_turnover = evaluator._run_state_machine(bnh_signals, close_prices_sliced, eval_prices_sliced, market_returns_sliced, active=active_sliced)
    bnh_sterling = evaluator._calculate_sterling_ratio(bnh_total_equity)

    print(f"📈 【Buy & Hold 基準績效】 最終淨值: {bnh_total_equity[-1]:.4f} | Sterling: {bnh_sterling:.4f}\n")

    # --- 計算個股獨立績效 (用於個股下拉選單顯示) ---
    pos = np.asarray(signals, dtype=float)
    pos[-1, :] = 0.0
    shifted_pos = np.roll(pos, 1, axis=0)
    shifted_pos[0, :] = 0.0
    diff = pos - shifted_pos
    
    price_shifted = np.roll(close_prices_sliced, 1, axis=0)
    price_shifted[0, :] = close_prices_sliced[0, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        asset_returns = (close_prices_sliced - price_shifted) / price_shifted
    asset_returns[np.isnan(asset_returns) | np.isinf(asset_returns)] = 0.0
    
    # 策略的單股淨值
    indiv_equity = np.cumprod(1.0 + (shifted_pos * asset_returns), axis=0)
    
    # B&H 的單股淨值 (僅價格變化，不計手續費)
    with np.errstate(divide='ignore', invalid='ignore'):
        bnh_indiv_equity = close_prices_sliced / close_prices_sliced[0, :]
    bnh_indiv_equity[np.isnan(bnh_indiv_equity) | np.isinf(bnh_indiv_equity)] = 1.0
    
    # ==========================================
    # 7. 解析 GP 語法樹圖結構
    # ==========================================
    print("🌳 正在優化 AI 邏輯樹排版...")
    nodes, edges, labels = gp.graph(best_ind)
    
    for node_id in labels:
        label = labels[node_id]
        
    symbol_map = {
        # 邏輯
        "and_": "AND", "or_": "OR",
        # Float
        "add": "+", "sub": "-", "gt": ">", "lt": "<",
        # Price
        "add_price": "+", "sub_price": "-", "gt_price": ">", "lt_price": "<",
        # Ratio
        "add_ratio": "+", "sub_ratio": "-", "gt_ratio": ">", "lt_ratio": "<",
        # Rank
        "add_rank": "+", "sub_rank": "-", "gt_rank": ">", "lt_rank": "<"
    }
    
    for node_id, current_label in labels.items():
        # 1. 替換運算子為符號
        if current_label in symbol_map:
            labels[node_id] = symbol_map[current_label]

    G = nx.DiGraph() 
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    try:
        # 用 nx_pydot 排版 (Windows 上比 pygraphviz 好安裝)
        pos_tree = nx.nx_pydot.graphviz_layout(G, prog="dot")
    except Exception as e:
        print(f"⚠️ Graphviz 呼叫失敗 ({e})，請確認是否安裝系統版 Graphviz 並加入 PATH。使用替代方案排版...")
        pos_tree = nx.spring_layout(G, k=0.5, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos_tree[edge[0]]
        x1, y1 = pos_tree[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos_tree[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(labels[node])

    # ==========================================
    # 8. 建立 Plotly 三層互動儀表板
    # ==========================================
    print("📈 正在生成互動式圖表，請稍候...")
    
    # MOO 時多一列畫 Pareto front，其他子圖往下移一列
    row_offset = 1 if pareto_front is not None else 0
    total_rows = 3 + row_offset
    
    # 動態組合子圖標題
    titles = []
    titles.extend(["Asset Price & Trading Signals", "Cumulative Equity Curve", "GP Syntax Tree"]) 
    if pareto_front is not None:
        titles.append("Pareto Front (MOO)")
    
    fig = make_subplots(
        rows=total_rows, cols=1, 
        vertical_spacing=0.08,
        shared_xaxes=False,
        subplot_titles=titles
    )
    
    
    # 繪製 Pareto Front 散佈圖
    if pareto_front is not None:
        # 收集所有策略的指標
        f1_scores = [ind.fitness.values[0] for ind in pareto_front]
        f2_scores = [ind.fitness.values[1] for ind in pareto_front]
        complexities = [len(ind) for ind in pareto_front] # 公式長度

        # 畫出所有策略 (點)
        fig.add_trace(go.Scatter(
            x=f1_scores,
            y=f2_scores,
            mode='markers',
            name='Pareto Front',
            marker=dict(
                size=12,
                color=complexities,      # 顏色深淺代表公式複雜度
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="公式節點數", 
                    x=1.05,
                    len=0.22,   # 色條長度約為一個子圖的高度
                    y=0.125    # 對齊第 4 個子圖
                ),
                line=dict(width=1, color='DarkSlateGrey')
            ),
            hovertemplate=(
                "<b>Sterling Ratio:</b> %{x:.4f}<br>"
                "<b>第二目標:</b> %{y:.4f}<br>"
                "<b>公式節點數:</b> %{marker.color}<br>"
                "<extra></extra>"
            )
        ), row=4, col=1) 

        # 以紅色星號標示目前繪製資金曲線的策略
        selected_f1 = best_ind.fitness.values[0]
        selected_f2 = best_ind.fitness.values[1]
        fig.add_trace(go.Scatter(
            x=[selected_f1],
            y=[selected_f2],
            mode='markers',
            name='Selected Strategy',
            marker=dict(symbol='star', size=18, color='red'),
            hovertemplate="<b>[當前展示策略]</b><br>Sterling: %{x:.4f}<br>第二目標: %{y:.4f}<extra></extra>"
        ), row=4, col=1)

        # 更新這張子圖的座標軸標籤
        fig.update_xaxes(title_text="Sterling Ratio (Max)", row=4, col=1)
        fig.update_yaxes(title_text=f"{multi_target} (Max/Min)", row=4, col=1)


    # --- [Trace 0] 策略總淨值曲線 ---
    fig.add_trace(go.Scatter(
        x=times_sliced, y=equity_curve, mode='lines', 
        name='Strategy Total Equity', line=dict(color='#1f77b4', width=2),
        customdata=np.stack((running_sterling, rolling_skew), axis=-1),
        hovertemplate=(
            "<b>時間:</b> %{x}<br>"
            "<b>資產淨值:</b> %{y:.4f}<br>"
            "<b>當下累積 Sterling:</b> %{customdata[0]:.4f}<br>"
            "<b>近一年報酬偏度:</b> %{customdata[1]:.4f}<br>"
            "<extra></extra>"
        )
    ), row=2, col=1)
    
    # --- [Trace 1] B&H 總淨值曲線 ---
    fig.add_trace(go.Scatter(
        x=times_sliced, y=bnh_total_equity, mode='lines', 
        name='Buy & Hold Baseline', line=dict(color='purple', width=2, dash='dot')
    ), row=2, col=1)
    
    # 如果有畫 Pareto Front，它會佔用最前面的 2 個 Trace Index
    pareto_offset = 2 if pareto_front is not None else 0
    
    # 總體績效線有 2 條 (Strategy Equity, B&H Equity)，所以個股起點要加上 pareto_offset
    stock_traces_start_idx = pareto_offset + 2 
    stock_trace_counts = 5      # 每支股票有 5 條線 (股價, 買點, 賣點, 策略淨值, B&H淨值)
    
    # --- [Traces] 個股數據 ---
    for i, symbol in enumerate(final_symbols):
        # 1. 股價
        fig.add_trace(go.Scatter(
            x=times_sliced, y=close_prices_sliced[:, i], mode='lines',
            name=f'{symbol} Price', line=dict(color='gray', width=1), visible=False
        ), row=1, col=1)
        
        # 2. 買點
        buy_idx = np.where(diff[:, i] == 1)[0]
        fig.add_trace(go.Scatter(
            x=times_sliced[buy_idx], y=close_prices_sliced[buy_idx, i], mode='markers',
            name=f'{symbol} Buy', marker=dict(symbol='triangle-up', size=10, color='green', line=dict(width=1, color='darkgreen')), visible=False
        ), row=1, col=1)
        
        # 3. 賣點
        sell_idx = np.where(diff[:, i] == -1)[0]
        fig.add_trace(go.Scatter(
            x=times_sliced[sell_idx], y=close_prices_sliced[sell_idx, i], mode='markers',
            name=f'{symbol} Sell', marker=dict(symbol='triangle-down', size=10, color='red', line=dict(width=1, color='darkred')), visible=False
        ), row=1, col=1)
        
        # 4. 策略單股獨立淨值
        fig.add_trace(go.Scatter(
            x=times_sliced, y=indiv_equity[:, i], mode='lines',
            name=f'{symbol} Strategy Equity', line=dict(color='orange', width=1.5), visible=False
        ), row=2, col=1)

        # 5. B&H 單股獨立淨值
        fig.add_trace(go.Scatter(
            x=times_sliced, y=bnh_indiv_equity[:, i], mode='lines',
            name=f'{symbol} B&H Equity', line=dict(color='purple', width=1.5, dash='dot'), visible=False
        ), row=2, col=1)

    # 樹狀圖的起點 = 個股終點
    tree_start_idx = stock_traces_start_idx + len(final_symbols) * stock_trace_counts


    # --- [Trace Tree Edges & Nodes] 繪製邏輯樹 ---
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.5, color='#888'), hoverinfo='none', showlegend=False
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text,
        textposition="middle center", hoverinfo='text', showlegend=False,
        textfont=dict(size=11, color="black", weight="bold"),
        marker=dict(color='lightblue', size=40, line=dict(width=2, color='black'))
    ), row=3, col=1)

    fig.update_xaxes(visible=False, row=3, col=1)
    fig.update_yaxes(visible=False, row=3, col=1)

    # ==========================================
    # 9. 建立 Dropdown 選單邏輯
    # ==========================================
    total_traces_count = tree_start_idx + 2
    
    # 標題同時列出 B&H 的 Sterling ratio
    dashboard_title = f"Portfolio Backtest | {selected_mode.upper()} | AI Sterling: {sterling:.2f} vs B&H: {bnh_sterling:.2f}"
    
    # 產生常駐 trace 的基礎 visibility 陣列
    def get_base_visibility():
        vis = [False] * total_traces_count
        # 1. Pareto front (trace 0, 1) 永遠顯示
        if pareto_front is not None:
            vis[0] = True 
            vis[1] = True
        # 2. 語法樹 (最後兩個 trace) 永遠顯示
        vis[tree_start_idx] = True   
        vis[tree_start_idx+1] = True
        return vis
    
    # 預設按鈕 (顯示總體績效)
    visibility_total = get_base_visibility()
    visibility_total[pareto_offset] = True     # Strategy Total Equity
    visibility_total[pareto_offset + 1] = True # B&H Baseline
    
    dropdown_buttons = [{
        'label': '🏆 Total Portfolio',
        'method': 'update',
        'args': [{'visible': visibility_total}, {'title': dashboard_title}]
    }]
    
    # 個股按鈕
    for i, symbol in enumerate(final_symbols):
        visibility_stock = get_base_visibility()
        
        start_idx = stock_traces_start_idx + i * stock_trace_counts
        visibility_stock[start_idx] = True     # Price
        visibility_stock[start_idx+1] = True   # Buy
        visibility_stock[start_idx+2] = True   # Sell
        visibility_stock[start_idx+3] = True   # Strategy Equity
        visibility_stock[start_idx+4] = True   # B&H Equity
        
        dropdown_buttons.append({
            'label': f'📈 {symbol}',
            'method': 'update',
            'args': [{'visible': visibility_stock},
                     {'title': f"{symbol} Analysis | Mode: {selected_mode.upper()}"}]
        })

    main_title = (
        f"[{opt_label} | {num_stocks} Stocks] "
        f"Target: {target_label} | Final Sterling: {sterling:.2f}"
    )

    # 全局外觀設定
    fig.update_layout(
        updatemenus=[{
            'active': 0,
            'buttons': dropdown_buttons,
            'direction': "down",
            'showactive': True,
            'x': 0.05,
            'xanchor': "left",
            'y': 1.12,
            'yanchor': "top"
        }],
        title=main_title,
        height=2000, 
        plot_bgcolor='white',
        hovermode="x unified"
    )
    
    # 加上輔助細格線
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=2, col=1)
    
    export_filename = f"Analysis_{opt_label}_{num_stocks}Stocks.html"
    fig.write_html(export_filename)
    print(f"✅ 互動式分析報告已儲存至: {export_filename}")


def select_by_topsis(front):
    """B. 理想點接近法 (TOPSIS)"""
    fits = np.array([ind.fitness.values for ind in front])
    min_vals, max_vals = fits.min(axis=0), fits.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-6
    norm_fits = (fits - min_vals) / ranges
    utopian = np.ones(fits.shape[1])
    distances = np.linalg.norm(norm_fits - utopian, axis=1)
    return front[np.argmin(distances)]

def select_by_knee_point(front):
    """A. 拐點選擇法 (Knee Point Selection)"""
    fits = np.array([ind.fitness.values for ind in front])
    min_idx, max_idx = np.argmin(fits[:, 0]), np.argmax(fits[:, 0])
    p1, p2 = fits[min_idx], fits[max_idx]
    if np.allclose(p1, p2): 
        return front[0]
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec) + 1e-6
    distances = []
    for p0 in fits:
        v0 = p0 - p1
        dist = np.abs(line_vec[0] * v0[1] - line_vec[1] * v0[0]) / line_len
        distances.append(dist)
    return front[np.argmax(distances)]

def select_by_oos(front, config):
    """C. 樣本外穩定性測試 (Out-of-Sample Selection)"""
    print("⏳ [OOS Mode] 正在驗證期 (Validation Set) 盲測 Pareto Front 的所有解...")
    
    from src.data_provider import DataProvider
    from src.evaluator import FitnessEvaluator
    from src.primitives import create_primitive_set
    from deap import gp
    import numpy as np

    # 1. 啟動 Validation 模式的 DataProvider
    val_dp = DataProvider(config=config, mode='val')
    val_evaluator = FitnessEvaluator(data_provider=val_dp, config=config)
    
    # 2. 重建 GP Primitive Set (讀取 config 中的 erc_range 設定)
    erc_range = config.get('features', {}).get('erc_range', [-100.0, 100.0])
    pset = create_primitive_set(val_dp.feature_cols, erc_range)
    
    # 3. 取出特徵陣列與價格矩陣
    feature_array = [val_dp.data[col].values for col in val_dp.feature_cols]
    close_prices = val_dp.full_close
    eval_prices = val_dp.full_eval_price
    market_returns = val_dp.full_market_returns
    
    best_ind = None
    best_val_score = -np.inf
    
    # 4. 在驗證期逐一評估 Pareto front 的個體
    for ind in front:
        try:
            # 將語法樹編譯為可執行的交易函數
            trading_func = gp.compile(ind, pset)
            signals = trading_func(*feature_array)
            
            # 公式只含常數 (如 TrueConst) 時，廣播成 (T, N)
            if not isinstance(signals, np.ndarray):
                signals = np.full(close_prices.shape, signals, dtype=bool)
                
            # 執行狀態機回測
            equity_curve, _ = val_evaluator._run_state_machine(signals, close_prices, eval_prices, market_returns, active=val_dp.full_active)
            
            # 計算驗證期的 Sterling Ratio (第一目標分數)
            score = val_evaluator._calculate_sterling_ratio(equity_curve)
            
            # 更新最高分
            if score > best_val_score:
                best_val_score = score
                best_ind = ind
                
        except Exception as e:
            print(f"⚠️ 個體驗證失敗: {e}")
            continue
            
    if best_ind is not None:
        print(f"✅ 盲測完成！選出 Validation 最佳 Sterling Ratio: {best_val_score:.4f}")
        return best_ind
    else:
        print("⚠️ 盲測全數失敗，自動退回 TOPSIS 挑選。")
        return select_by_topsis(front)



if __name__ == "__main__":
    main()
