"""
多次獨立實驗：以相同設定、不同 seed 重複執行 main.py，彙整 Train / Test 績效並產生 HTML 報告。

  - 每個 run 在獨立子進程中執行，避免 DEAP creator 的全域狀態互相干擾。
  - 子進程攔截 open("config.yaml")，改讀該 run 的設定副本；根目錄的 config.yaml 不會被修改。
  - 互動選單：Q0 模式 (一般 / 正式實驗 / 只產報告)、Q1 run 次數、Q2 單目標或多目標。
"""

import os
import sys
import yaml
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from collections import Counter
import time
from src.tree_simplifier import export_simplified_trees_to_txt, generate_llm_prompt_file

# 專案模組 (樣本外評估與畫圖用；不會觸發 main.py 的初始化)
from src.primitives import create_primitive_set
from src.data_provider import DataProvider
from src.evaluator import FitnessEvaluator
from src.moo_selection import pick_moo_individual, resolve_method
from deap import creator, base, gp

def init_deap_environment(config):
    """依本次實驗的 weights 重建 DEAP creator 類別 (樣本外評估與繪圖用)"""
    w = tuple(config.get('evolution', {}).get('weights', [1.0]))
    # 清除可能殘留的 DEAP 設定
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
        
    creator.create("FitnessMax", base.Fitness, weights=w)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def get_moo_strategy_individual(front, method, config=None):
    """多目標 Pareto Front 挑選邏輯"""
    fits = np.array([ind.fitness.values for ind in front])
    if len(fits) == 0: return None
        
    if method == "topsis":
        min_vals, max_vals = fits.min(axis=0), fits.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1e-6
        norm_fits = (fits - min_vals) / ranges
        utopian = np.ones(fits.shape[1])
        distances = np.linalg.norm(norm_fits - utopian, axis=1)
        return front[np.argmin(distances)]
        
    elif method in ("knee", "knee_point"):   # config 與 strategy_analyzer 用 knee_point，兩種寫法都接受
        min_idx, max_idx = np.argmin(fits[:, 0]), np.argmax(fits[:, 0])
        p1, p2 = fits[min_idx], fits[max_idx]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0: line_len = 1e-6
        distances = []
        for p0 in fits:
            v0 = p0 - p1
            dist = np.abs(line_vec[0] * v0[1] - line_vec[1] * v0[0]) / line_len
            distances.append(dist)
        return front[np.argmax(distances)]
        
    elif method == "oos":
        val_dp = DataProvider(config=config, mode='val')
        val_evaluator = FitnessEvaluator(data_provider=val_dp, config=config)
        pset = create_primitive_set(val_dp.feature_cols, config['features'].get('erc_range', [-100.0, 100.0]))
        
        feature_array = val_dp.full_features_list
        close_prices = val_dp.full_close
        val_eval_price = val_dp.full_eval_price
        market_returns = val_dp.full_market_returns
        
        best_ind = front[0]
        best_val_score = -np.inf
        for ind in front:
            trading_func = gp.compile(ind, pset)
            signals = trading_func(*feature_array)
            if not isinstance(signals, np.ndarray):
                signals = np.full(close_prices.shape, signals, dtype=bool)
                
            # 市場報酬沿用 val_dp.full_market_returns (eval_price 基礎，與訓練/驗證一致)
            equity_curve, _ = val_evaluator._run_state_machine(signals, close_prices, val_eval_price, market_returns, active=val_dp.full_active)
            
            score = val_evaluator._calculate_sterling_ratio(equity_curve)
            if score > best_val_score:
                best_val_score = score
                best_ind = ind
        return best_ind
    return front[0]

def trace_lineage(history, target_ind):
    """
    溯源最佳個體的演化歷史
    回傳一個列表，包含從初代到該個體的演化路徑節點資訊
    """
    if not hasattr(target_ind, 'history_index'):
        return [{"節點ID": "N/A", "操作": "未記錄歷史", "樹狀結構": str(target_ind)}]
        
    target_id = target_ind.history_index
    lineage_records = []
    visited = set()

    def dfs(current_id, depth):
        if current_id in visited:
            return
        visited.add(current_id)
        
        parents = history.genealogy_tree.get(current_id, ())
        ind = history.genealogy_history.get(current_id)
        
        if len(parents) == 0:
            operation = "初代生成 (Init)"
        elif len(parents) == 1:
            operation = f"突變 (Mutate from {parents[0]})"
        else:
            operation = f"交配 (Mate from {parents[0]} & {parents[1]})"
            
        record = {
            "深度": depth, # 與最終個體相隔的世代數 (0 = 最終個體)
            "節點ID": current_id,
            "操作 (來源)": operation,
            "樹狀結構": str(ind) if ind else "N/A"
        }
        lineage_records.append(record)
        
        for p in parents:
            dfs(p, depth + 1)

    dfs(target_id, 0)
    
    # 依深度反轉，表格由舊到新呈現
    lineage_records.sort(key=lambda x: x["深度"], reverse=True)
    return lineage_records


def main():
    project_root = Path.cwd().resolve()
    
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        print("❌ 錯誤：找不到根目錄下的 config.yaml！")
        return
        
    # 唯讀：本程式不會寫入根目錄的 config.yaml
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    print("👉 Q0. 是否要跑正式實驗？")
    print("   [1] 否 (一般測試模式)")
    print("   [2] 是 (正式實驗：自動跑單目標 30 次與多目標 30 次，並進行 Mann-Whitney U 檢定與圖表產出)")
    print("   [3] 僅生報告 (讀取已完成的 gp_experiment_studies 資料夾並產出報告)")
    exp_type = input("請選擇 (1, 2 或 3): ").strip()

    is_formal_experiment = (exp_type == "2")
    is_report_only = (exp_type == "3")
    
    
    if is_report_only:
        studies_dir = project_root / "gp_experiment_studies"
        if not studies_dir.exists():
            print("❌ 找不到 gp_experiment_studies 資料夾！")
            return
            
        studies = sorted([d for d in studies_dir.iterdir() if d.is_dir() and d.name.startswith("study_")])
        if not studies:
            print("❌ 找不到任何已完成的 study 資料夾！")
            return
            
        print("\n📂 發現以下實驗紀錄：")
        for idx, s in enumerate(studies, 1):
            print(f"   [{idx}] {s.name}")
        sel = input(f"👉 請選擇要產生報告的實驗 (1-{len(studies)}): ").strip()
        
        try:
            study_root = studies[int(sel)-1]
        except (ValueError, IndexError):
            print("❌ 輸入錯誤，結束程式。")
            return
            
        modes_to_run = [d.name for d in study_root.iterdir() if d.is_dir() and d.name in ["single", "multi"]]
        if not modes_to_run:
            print(f"❌ {study_root.name} 內找不到 single 或 multi 資料夾！")
            return
        
        moo_selection = "topsis"
        if "multi" in modes_to_run:
            print("   請選擇 Pareto Front 的最優個體挑選策略：")
            print("   [1] TOPSIS (理想點逼近法)")
            print("   [2] Knee Point (拐點/最大曲率法)")
            print("   [3] OOS (Validation 驗證集盲測得分最高法)")
            sel_input = input("   👉 請選擇挑選策略 (輸入 1, 2 或 3) [預設 1]: ").strip()
            if sel_input == "2": moo_selection = "knee_point"
            elif sel_input == "3": moo_selection = "oos"
            
        # 只產報告時，study 內同時有 single 與 multi 就視為正式實驗 (會產生 SOO vs MOO 比較)
        is_formal_experiment = ("single" in modes_to_run and "multi" in modes_to_run)
        print(f"\n📂 本次將為以下模式產生報告: {modes_to_run}")
        print("-" * 60)
    
    else:
        if is_formal_experiment:
            num_runs = 30
            modes_to_run = ["single", "multi"]
            moo_selection = "topsis"
        else:
            num_runs_input = input("👉 Q1. 請輸入要重複執行的次數 (例如 30): ").strip()
            num_runs = int(num_runs_input)

            mode_input = input("👉 Q2. 請選擇執行模式 [1] 單目標 (SOO)  [2] 多目標 (MOO) (輸入 1 或 2): ").strip()
            obj_mode = "single" if mode_input == "1" else "multi"
            modes_to_run = [obj_mode]
    
            moo_selection = "topsis"
            if obj_mode == "multi":
                print("\n   請選擇 Pareto Front 的最優個體挑選策略：               ")
                print("   [1] TOPSIS (理想點逼近法)                              ")
                print("   [2] Knee Point (拐點/最大曲率法)                        ")
                print("   [3] OOS (Validation 驗證集盲測得分最高法)              ")
                sel_input = input("   👉 請選擇挑選策略 (輸入 1, 2 或 3): ").strip()
                if sel_input == "2": moo_selection = "knee_point"
                elif sel_input == "3": moo_selection = "oos"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_root = project_root / "gp_experiment_studies" / f"study_{timestamp}"
        study_root.mkdir(parents=True, exist_ok=True)
    
    pkl_path = project_root / base_config['data']['file_path']
    if not pkl_path.exists():
        print(f"❌ 找不到特徵檔 {pkl_path}，請先執行 data_pipeline/features_pit.py (見 data/README.md)")
        return

    print(f"\n📂 本次實驗結果將存放於: {study_root.resolve()}")
    print("-" * 60)

    # ================= 每個 run 在獨立子進程中執行 =================
    formal_results = {}
    
    for current_mode in modes_to_run:
        print(f"\n" + "=" * 60)
        print(f"🚀 正在執行模式: {current_mode.upper()} 🚀")
        print("=" * 60)
        
        obj_mode = current_mode
        mode_study_root = study_root / obj_mode
        
        if is_report_only:
            # 產報告模式：從既有檔案夾讀取，不執行演化
            run_dirs = sorted([d for d in mode_study_root.iterdir() if d.is_dir() and d.name.startswith("run_")])
            num_runs = len(run_dirs)
            if num_runs == 0:
                print(f"⚠️ {current_mode} 模式下沒有找到 run_ 資料夾，跳過。")
                continue
            
            print(f"🔎 找到 {num_runs} 個 run，正在解析時間紀錄檔...")
            first_run_config_path = run_dirs[0] / "temp_config.yaml"
            with open(first_run_config_path, "r", encoding="utf-8") as f:
                run_config = yaml.safe_load(f)
                
            total_time_sec = 0
            for r_dir in run_dirs:
                logs = list(r_dir.glob("training_log_*.log"))
                if logs:
                    try:
                        with open(logs[0], "r", encoding="utf-8") as lf:
                            lines = [line for line in lf.readlines() if line.strip()]
                            if len(lines) >= 2:
                                # 解析格式 (例如: 2026-08-03 11:47:11,123 - message)
                                t1_str = lines[0].split(" - ")[0].split(",")[0]
                                t2_str = lines[-1].split(" - ")[0].split(",")[0]
                                fmt = "%Y-%m-%d %H:%M:%S"
                                t1 = datetime.strptime(t1_str, fmt)
                                t2 = datetime.strptime(t2_str, fmt)
                                dt_diff = (t2 - t1).total_seconds()
                                if dt_diff > 0:
                                    total_time_sec += dt_diff
                    except Exception:
                        pass
            
            avg_time_per_run = total_time_sec / num_runs if num_runs > 0 else 0
        
        else:
            mode_study_root.mkdir(parents=True, exist_ok=True)
            mode_start_time = time.time()
    
            for i in range(1, num_runs + 1):
                run_dir = mode_study_root / f"run_{i:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"🏃 [進度 {i:02d}/{num_runs:02d}] 正在隔離環境中啟動獨立演化進程...")

                run_config = yaml.safe_load(yaml.dump(base_config))
                run_config['system']['checkpoint_dir'] = str(run_dir.resolve())
                run_config['evolution']['objective_mode'] = obj_mode
            
                # 每個 run 的 seed = 基準 seed + run 編號
                if 'system' not in run_config:
                    run_config['system'] = {}
                base_seed = base_config.get('system', {}).get('seed', 42)
                run_config['system']['seed'] = base_seed + i
            
                if obj_mode == "single":
                    run_config['evolution']['weights'] = [1.0]
                else:
                    if len(base_config['evolution'].get('weights', [])) < 2:
                        run_config['evolution']['weights'] = [1.0, 1.0]

                temp_config_path = run_dir / "temp_config.yaml"
                with open(temp_config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(run_config, f, allow_unicode=True)

                # 子進程啟動碼：把 open("config.yaml") 導向本 run 的 temp_config.yaml
                launcher_code = f"""
import builtins
import sys
from pathlib import Path

# 把專案根目錄加入模組搜尋路徑
sys.path.insert(0, r"{project_root.resolve()}")

# 讀取 'config.yaml' 時改讀本 run 的 temp_config.yaml
original_open = builtins.open
def custom_open(file, *args, **kwargs):
    if str(file) == "config.yaml":
        return original_open(r"{temp_config_path.resolve()}", *args, **kwargs)
    return original_open(file, *args, **kwargs)
builtins.open = custom_open

# 攔截設定完成後才匯入 main.py
import main

# multiprocessing 在 Windows 以 spawn 啟動子進程，需以 __main__ 保護避免遞迴啟動
if __name__ == '__main__':
    if hasattr(main, 'main'):
        main.main()
"""
                launcher_path = run_dir / "launcher.py"
                with open(launcher_path, "w", encoding="utf-8") as f:
                    f.write(launcher_code)
            
                try:
                    subprocess.run([sys.executable, str(launcher_path)], check=True, cwd=str(project_root))
                    print(f"✨ [Run {i:02d}] 演化結束，資料安全隔離。")
                except subprocess.CalledProcessError as e:
                    print(f"❌ [Run {i:02d}] 發生錯誤，停止後續批次。")
                    return
                print("-" * 60)
            
            mode_end_time = time.time()
            total_time_sec = mode_end_time - mode_start_time
            avg_time_per_run = total_time_sec / num_runs if num_runs > 0 else 0
        
        # 讀取單一代數，以計算平均一代耗時
        max_gen = run_config.get('evolution', {}).get('max_gen', 100)   # 用該 study 自己的 temp_config (report-only 時 base_config 可能已被改過)
        avg_time_per_gen = avg_time_per_run / max_gen if max_gen > 0 else 0

        def format_time(seconds):
            h, r = divmod(seconds, 3600)
            m, s = divmod(r, 60)
            if h > 0: return f"{int(h)}時{int(m)}分{s:.1f}秒"
            if m > 0: return f"{int(m)}分{s:.1f}秒"
            return f"{s:.2f}秒"

        time_headers = [
            "總執行次數<br>(Total Runs)", 
            "實驗總耗時<br>(Total Time)", 
            "單次執行平均耗時<br>(Avg Time / Run)", 
            "單代演化平均耗時<br>(Avg Time / Gen)"
        ]
        time_values = [
            [f"{num_runs}"],
            [format_time(total_time_sec)],
            [format_time(avg_time_per_run)],
            [format_time(avg_time_per_gen)]
        ]

        # ================= 樣本外評估與畫圖 =================
        print("\n📊 演化完成！開始進行樣本外 (OOS) 動態對比...")
    
        final_config = run_config 
        
        from src.evaluator_moo import resolve_scheme, objective_labels
        multi_target = resolve_scheme(final_config) or final_config.get('evolution', {}).get('multi_target', 'skewness')
        obj1_label, obj2_label = objective_labels(final_config)
        
        init_deap_environment(final_config)
        
        train_dp = DataProvider(config=final_config, mode='train')
        train_evaluator = FitnessEvaluator(data_provider=train_dp, config=final_config)
        
        test_dp = DataProvider(config=final_config, mode='test')
        test_evaluator = FitnessEvaluator(data_provider=test_dp, config=final_config)
    
        train_feature_array = train_dp.full_features_list
        train_close = train_dp.full_close
        train_eval_price = train_dp.full_eval_price
        train_market_returns = train_dp.full_market_returns
        train_active = train_dp.full_active      # point-in-time 股票池 (舊 pkl = None)
        
        test_feature_array = test_dp.full_features_list
        test_close = test_dp.full_close
        test_eval_price = test_dp.full_eval_price
        test_market_returns = test_dp.full_market_returns
        test_active = test_dp.full_active
        
        test_times = test_dp.data['close'].index
        
        train_times = train_dp.data['close'].index
        
        pset = create_primitive_set(test_dp.feature_cols, final_config['evolution'].get('erc_range', [-100.0, 100.0]))

        curves_data = {}
        train_curves_data = {}
        
        metrics_summary = []
        max_gen = final_config['evolution']['max_gen']

        # 用來紀錄 Anytime Behavior 的矩陣，大小為 (num_runs, max_gen) (SOO使用)
        anytime_matrix = np.zeros((num_runs, max_gen))
        
        # 用來紀錄 MOO 每一 Run 在 5 的倍數世代的 Pareto Front (MOO使用)
        moo_fronts_data = {i: {} for i in range(1, num_runs + 1)}


        # 30 個 run 共用同一份 Val 資料與分數快取，不用每個 run 重載
        moo_validator = None
        if obj_mode == "multi":
            from src.moo_validation import MOOValidator
            moo_validator = MOOValidator(final_config, champion_rule=final_config['evolution'].get('champion_rule', 'val'))

        for i in range(1, num_runs + 1):
            run_dir = mode_study_root / f"run_{i:02d}"
            
            import gzip
            # 讀取每個世代的 best_model 來建立 Anytime Behavior / Pareto Front 數據
            for g in range(1, max_gen + 1):
                gen_model_path = run_dir / f"best_model_gen_{g}.pkl"
                if gen_model_path.exists():
                    with open(gen_model_path, "rb") as test_f:
                        is_gzipped = test_f.read(2) == b'\x1f\x8b'
                    
                    if is_gzipped:
                        with gzip.open(gen_model_path, "rb") as f:
                            raw_gen_model = pickle.load(f)
                    else:
                        with open(gen_model_path, "rb") as f:
                            raw_gen_model = pickle.load(f)
                    
                    # 取得該代最佳 Fitness (樣本內適應度)
                    if obj_mode == "multi" and isinstance(raw_gen_model, list):
                        # 多目標時：raw_gen_model 是 Pareto Front (個體列表)
                        # 取第一目標最大值供 Anytime 備用
                        best_fit = max([ind.fitness.values[0] for ind in raw_gen_model])
                        
                        # 每 5 代 (或最後一代) 紀錄 Pareto Front 的所有解
                        if g % 5 == 0 or g == max_gen:
                            # 依第一目標排序，以便連線
                            pts = sorted([(ind.fitness.values[0], ind.fitness.values[1]) for ind in raw_gen_model])
                            moo_fronts_data[i][g] = {
                                'x': [p[0] for p in pts],
                                'y': [p[1] for p in pts]
                            }
                    else:
                        # 單目標時：raw_gen_model 是單一最佳個體
                        best_fit = raw_gen_model.fitness.values[0]
                    
                    # 單、多目標皆在此寫入 best_fit
                    anytime_matrix[i-1, g-1] = best_fit
                else:
                    # 如果某代缺失，沿用上一代的數值
                    anytime_matrix[i-1, g-1] = anytime_matrix[i-1, g-2] if g > 1 else 0
                    if obj_mode == "multi" and (g % 5 == 0 or g == max_gen):
                        moo_fronts_data[i][g] = moo_fronts_data[i].get(g-1, {'x':[], 'y':[]})
            
            model_path = run_dir / f"best_model_final.pkl"
        
            if not model_path.exists(): 
                print(f"⚠️ 找不到模型檔案: {model_path}，跳過此 Run。")
                continue
            
            with open(model_path, "rb") as test_f:
                is_gzipped = test_f.read(2) == b'\x1f\x8b'
                
            if is_gzipped:
                with gzip.open(model_path, "rb") as f:
                    raw_model = pickle.load(f)
            else:
                with open(model_path, "rb") as f:
                    raw_model = pickle.load(f)
        
            if obj_mode == "multi" and isinstance(raw_model, list):
                ind, sel_info = pick_moo_individual(raw_model, run_dir, final_config,
                                                    menu_method=moo_selection, validator=moo_validator)
                if i == 1 or sel_info.get('fallback'):
                    print(f"   🎯 挑選方式: {sel_info['method_used']} (requested={sel_info['method_requested']}, fallback={sel_info['fallback']})")
            else:
                ind = raw_model
            
            if ind is None: continue


            # 讀取 History 並溯源 
            history_path = run_dir / "evolution_history.pkl"
            lineage_df = pd.DataFrame([{"深度": "N/A", "節點ID": "N/A", "操作 (來源)": "未找到歷史(請確認 main.py 有儲存 history)", "樹狀結構": "N/A"}])
            if history_path.exists():
                import gzip
                with open(history_path, "rb") as test_f:
                    is_gzipped = test_f.read(2) == b'\x1f\x8b'
                
                if is_gzipped:
                    with gzip.open(history_path, "rb") as f:
                        run_history = pickle.load(f)
                else:
                    with open(history_path, "rb") as f:
                        run_history = pickle.load(f)
                        
                lineage_data = trace_lineage(run_history, ind)
                lineage_df = pd.DataFrame(lineage_data)
                
                # history 很大，用完立即釋放記憶體
                del run_history
                import gc
                gc.collect()
                
            curves_data[f"Run_{i:02d}_lineage"] = lineage_df
            
            
            trading_func = gp.compile(ind, pset)
            
            # --- 2. 評估訓練集 (Train) 表現 ---
            train_signals = trading_func(*train_feature_array)
            if not isinstance(train_signals, np.ndarray):
                train_signals = np.full(train_close.shape, train_signals, dtype=bool)
            train_eq, _ = train_evaluator._run_state_machine(train_signals, train_close, train_eval_price, train_market_returns, active=train_active)
            train_sterling = train_evaluator._calculate_sterling_ratio(train_eq)
            train_ror = train_evaluator._calculate_ror(train_eq)
            
            train_curves_data[f"Run_{i:02d}"] = train_eq
            
            # 計算 Train 的報酬標準差 (年化波動率)
            bars_per_year = final_config.get('trading', {}).get('bars_per_year', 252) 
            train_returns = np.diff(train_eq) / train_eq[:-1]
            train_std = np.std(train_returns) * np.sqrt(bars_per_year)
            
            
            # --- 3. 評估測試集 (Test) 表現 ---
            test_signals = trading_func(*test_feature_array)
            if not isinstance(test_signals, np.ndarray):
                test_signals = np.full(test_close.shape, test_signals, dtype=bool)
            test_eq, test_turnover = test_evaluator._run_state_machine(test_signals, test_close, test_eval_price, test_market_returns, active=test_active)
            test_sterling = test_evaluator._calculate_sterling_ratio(test_eq)
            test_ror = test_evaluator._calculate_ror(test_eq)
            
            test_running_max = np.maximum.accumulate(test_eq)
            test_mdd = np.max((test_running_max - test_eq) / test_running_max)
            
            # 計算 Test 的報酬標準差 (年化波動率)
            test_returns = np.diff(test_eq) / test_eq[:-1]
            test_std = np.std(test_returns) * np.sqrt(bars_per_year)
            
            curves_data[f"Run_{i:02d}"] = test_eq
            
            
            # 4. 彙整 Train 與 Test 結果
            metrics_summary.append({
                "項目": f"Run_{i:02d}",
                "Train_Sterling": train_sterling,
                "Train_Return": train_ror,
                "Train_StdDev": train_std,
                "Test_Sterling": test_sterling, 
                "Test_Return": test_ror, 
                "Test_Drawdown": test_mdd, 
                "Test_StdDev": test_std,
                "總交易次數 (Turnover)": test_turnover,
                "最優公式樹結構": str(ind),
                "個體物件": ind
            })
            
        if len(metrics_summary) == 0:
            print("❌ 錯誤：未能成功讀取到任何 GP 模型。")
            continue
        
        # 1. Buy & Hold 基準 (src/benchmarks.py)
        from src.benchmarks import compute_benchmarks
        bench = compute_benchmarks(test_evaluator, test_close, test_eval_price, test_market_returns, active=test_active)
        bh = bench['true_bh']
        bh_curve, bh_turnover = bh['curve'], bh['turnover']
        bh_sterling, bh_ror, bh_mdd = bh['sterling'], bh['ror'], bh['mdd']
        curves_data["Buy & Hold 基準線"] = bh_curve
        # (選用) 同計費方式的等權每根再平衡，作為「無選股」對照
        curves_data["等權每期再平衡 (同 GP 計費)"] = bench['ew_same']['curve']
        
        
        # 2. 計算 30 次實驗的平均值與標準差
        # 各 run 的淨值曲線組成矩陣
        run_keys = [k for k in curves_data.keys() if k.startswith("Run_") and not k.endswith("_lineage")]
        all_curves = np.array([curves_data[k] for k in run_keys])
        
        avg_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        upper_bound = avg_curve + std_curve
        lower_bound = avg_curve - std_curve
        
        # 計算訓練曲線的平均與標準差
        all_train_curves = np.array([train_curves_data[k] for k in run_keys])
        avg_train_curve = np.mean(all_train_curves, axis=0)
        std_train_curve = np.std(all_train_curves, axis=0)
        upper_train_bound = avg_train_curve + std_train_curve
        lower_train_bound = avg_train_curve - std_train_curve
        
        
        # 提取 30 次 Run 的各項最終指標陣列
        all_sterlings = [m["Test_Sterling"] for m in metrics_summary]
        all_returns = [m["Test_Return"] for m in metrics_summary]
        all_drawdowns = [m["Test_Drawdown"] for m in metrics_summary]
        all_turnovers = [m["總交易次數 (Turnover)"] for m in metrics_summary]
        
        avg_sterling = np.mean(all_sterlings)
        avg_return = np.mean(all_returns)
        avg_drawdown = np.mean(all_drawdowns)
        avg_turnover = np.mean(all_turnovers)
        
        # 計算統計指標的樣本標準差 (Cross-Run StdDev)，ddof=1 代表樣本標準差
        std_sterling = np.std(all_sterlings, ddof=1)
        std_return = np.std(all_returns, ddof=1)
        std_drawdown = np.std(all_drawdowns, ddof=1)
        std_turnover = np.std(all_turnovers, ddof=1)
        
        
        # 3. 建立表格資料 (順序：平均 -> B&H -> 各 Run)
        display_metrics = []
        
        # 顯示 Mean ± Std
        display_metrics.append({
            "項目": "測試結果統計 (Mean ± Std)",
            "Sterling Ratio": f"{avg_sterling:.4f} ± {std_sterling:.4f}",
            "年化報酬率 (RoR)": f"{avg_return * 100:.2f}% ± {std_return * 100:.2f}%",
            "最大回撤 (MDD)": f"{avg_drawdown * 100:.2f}% ± {std_drawdown * 100:.2f}%",
            "年化波動率 (Std)": "N/A (見各Run波動)", 
            "總交易次數 (Turnover)": f"{avg_turnover:.1f} ± {std_turnover:.1f}",
            "最優樹": "N/A (跨30次統計)"
        })
        
        display_metrics.append({
                "項目": "Buy & Hold 基準線",
                "Sterling Ratio": round(bh_sterling, 4),
                "年化報酬率 (RoR)": f"{bh_ror * 100:.2f}%",
                "最大回撤 (MDD)": f"{bh_mdd * 100:.2f}%",
                "年化波動率 (Std)": "N/A",  # B&H 只有一條曲線，沒有跨 run 標準差
                "總交易次數 (Turnover)": bh_turnover,
                "最優樹": "N/A (基準大盤持倉)"
            })
        
        
        for m in metrics_summary:
            display_metrics.append({
                "項目": m["項目"],
                "Sterling Ratio": round(m["Test_Sterling"], 4),
                "年化報酬率 (RoR)": f"{m['Test_Return'] * 100:.2f}%",
                "最大回撤 (MDD)": f"{m['Test_Drawdown'] * 100:.2f}%",
                "年化波動率 (Std)": f"{m['Test_StdDev'] * 100:.2f}%",
                "總交易次數 (Turnover)": m["總交易次數 (Turnover)"],
                "最優樹": m["最優公式樹結構"]
            })

        # 計算 Anytime Behavior 的平均與標準差
        avg_anytime = np.mean(anytime_matrix, axis=0)
        std_anytime = np.std(anytime_matrix, axis=0)
        generations = np.arange(1, max_gen + 1)
        
             
        # 4. 繪製圖表與設定互動選單
        total_lines = len(curves_data)
        # total_lines 同時包含 *_lineage 鍵，run 數需除以 2
        real_run_count = len([k for k in curves_data.keys() if k.startswith("Run_") and not k.endswith("_lineage")])
        
        evo_cfg = final_config.get('evolution', {})
        # 設定第一橫排 (表頭)：使用 <br> 讓英文括號換行至下一行
        param_headers = [
            "族群大小<br>(pop_size)", 
            "生成子代大小<br>(offspring_size)", 
            "最大世代數<br>(max_gen)", 
            "樹最大深度<br>(max_depth)", 
            "交配機率<br>(cxpb)", 
            "突變機率<br>(mutpb)", 
            "Tournament Size<br>(tournsize)", 
            "初始樹最小<br>深度 (init_min)", 
            "初始樹最大<br>深度 (init_max)"
        ]
        
        # 設定第二橫排 (數值)：Plotly 要求每個欄位的數值被包裝成一個列表
        param_values = [
            [evo_cfg.get('pop_size', 'N/A')],
            [evo_cfg.get('offspring_size', 'N/A')],
            [evo_cfg.get('max_gen', 'N/A')],
            [evo_cfg.get('max_depth', 'N/A')],
            [evo_cfg.get('cxpb', 'N/A')],
            [evo_cfg.get('mutpb', 'N/A')],
            [evo_cfg.get('tournsize', 'N/A')],
            [evo_cfg.get('init_min', 'N/A')],
            [evo_cfg.get('init_max', 'N/A')]
        ]
        
        # 高頻子樹統計
        # a. 篩選精英個體 (以 Test_Sterling 排序，取前 20%)
        sorted_metrics = sorted(metrics_summary, key=lambda x: x["Test_Sterling"], reverse=True)
        top_k = max(1, int(len(sorted_metrics) * 0.2)) 
        elite_inds = [m["個體物件"] for m in sorted_metrics[:top_k]]
        
        subtree_counter = Counter()
        
        # b. 遞迴拆解子樹 (利用 DEAP 的 searchSubtree 與 height)
        for ind_obj in elite_inds:
            # 走訪這棵樹上的所有節點
            for node_idx in range(len(ind_obj)):
                # arity > 0 為函數節點 (如 and_、gt_rank)，否則為終端節點
                if ind_obj[node_idx].arity > 0: 
                    slice_ = ind_obj.searchSubtree(node_idx) # 切出以該節點為 root 的子樹切片
                    subtree = ind_obj[slice_]
                    prim_tree = gp.PrimitiveTree(subtree)
                    
                    # DEAP 的 height：1 為單層函數，2 為兩層巢狀
                    if 1 <= prim_tree.height <= 2:
                        subtree_counter[str(prim_tree)] += 1
        
        # c. 取得 Top 10 並準備畫長條圖
        top_10 = subtree_counter.most_common(10)
        # 反轉順序，使水平長條圖的第一名位於最上方
        top_10_labels = [item[0] for item in top_10][::-1] if top_10 else ["無"]
        top_10_counts = [item[1] for item in top_10][::-1] if top_10 else [0]
        
        # 解析樹狀結構，並在第一層級的逗號後插入 <br> 進行換行
        def format_gp_tree_label(label):
            result = ""
            depth = 0
            for char in label:
                if char == '(':
                    depth += 1
                    result += char
                elif char == ')':
                    depth -= 1
                    result += char
                elif char == ',' and depth == 1:
                    result += ",<br>"  # 第一層深度遇到逗號時，加上 Plotly 支援的換行標籤
                elif char == ' ' and result.endswith("<br>"):
                    continue           # 略過換行後的空白
                else:
                    result += char
            return result

        display_labels = [format_gp_tree_label(lbl) for lbl in top_10_labels]
        
        # 決定第五子圖的標題
        row_6_title = "Pareto Front (每5代)" if obj_mode == "multi" else "Anytime Behavior"
        
        fig = make_subplots(
            rows=8, cols=1, shared_xaxes=False, 
            vertical_spacing=0.035,   # 整體子圖間的垂直間距
            row_heights=[0.05, 0.05, 0.20, 0.12, 0.16, 0.18, 0.16, 0.08],  # 調整 8 個子圖的相對高度
            specs=[
                [{"type": "table"}], 
                [{"type": "table"}],
                [{"type": "scatter"}], 
                [{"type": "xy"}],       # Row 4: Box Plot for sterling ratio
                [{"type": "table"}],    # Row 5: 結果表
                [{"type": "scatter"}],  # Row 6: Pareto / Anytime
                [{"type": "table"}],    # Row 7: Lineage
                [{"type": "bar"}]       # Row 8: 高頻特徵
            ],
            subplot_titles=(
                "實驗參數設定", 
                "執行時間統計",
                f"結果 ({int((total_lines-1)/2)} runs)", 
                "Sterling Ratio",
                "結果表", 
                row_6_title,
                "最末代最佳解溯源 (Lineage)",
                "高頻特徵排行 (Top 20% Runs)",
            )
        )
        
        fig.add_trace(
            go.Table(
                header=dict(values=param_headers, fill_color='teal', align='center', font=dict(size=12, color='white')),
                cells=dict(values=param_values, fill_color='ghostwhite', align='center', font=dict(size=12, color='black'))
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Table(
                header=dict(values=time_headers, fill_color='steelblue', align='center', font=dict(size=12, color='white')),
                cells=dict(values=time_values, fill_color='ghostwhite', align='center', font=dict(size=12, color='black'))
            ), row=2, col=1
        )
        
        # 確保順序以決定 Trace Index (上界、下界、平均線)
        # --- 訓練結果的繪製  ---
        fig.add_trace(go.Scatter(x=train_times, y=upper_train_bound, name="訓練平均+1 Std", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=3, col=1)
        fig.add_trace(go.Scatter(x=train_times, y=lower_train_bound, name="訓練平均-1 Std", line=dict(width=0), fill='tonexty', fillcolor='rgba(144, 238, 144, 0.3)', showlegend=False, hoverinfo="skip"), row=3, col=1)
        fig.add_trace(go.Scatter(x=train_times, y=avg_train_curve, name="訓練結果平均", line=dict(color='green', width=2.5)), row=3, col=1)

        # --- 測試結果的繪製  ---
        fig.add_trace(go.Scatter(x=test_times, y=upper_bound, name="測試平均+1 Std", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=3, col=1)
        fig.add_trace(go.Scatter(x=test_times, y=lower_bound, name="測試平均-1 Std", line=dict(width=0), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.3)', showlegend=False, hoverinfo="skip"), row=3, col=1)
        fig.add_trace(go.Scatter(x=test_times, y=avg_curve, name="測試結果平均", line=dict(color='red', width=2.5)), row=3, col=1)

        # B&H
        fig.add_trace(go.Scatter(x=test_times, y=bh_curve, name="Buy & Hold 基準線", line=dict(color='black', width=3, dash='dash')), row=3, col=1)

        # 各 Run (預設為隱藏，透過選單叫出)
        for name in run_keys:
            fig.add_trace(go.Scatter(x=test_times, y=curves_data[name], name=name, opacity=0.7, line=dict(width=1.5), visible=False), row=3, col=1)

        
        # Row 4：各 run Sterling ratio 的盒狀圖
        fig.add_trace(go.Box(
            x=all_sterlings,
            name="GP Runs",
            boxpoints='all',      # 顯示所有 30 runs 的散佈點
            jitter=0.5,           # 讓點稍微散開不重疊
            pointpos=-1.5,        # 點顯示在箱體下方
            marker_color='teal',
            showlegend=False
        ), row=4, col=1)
        
        # 1. 繪製 Buy & Hold (紅色虛線)
        fig.add_shape(
            type="line", x0=bh_sterling, x1=bh_sterling, y0=0, y1=1,
            yref="y domain", # y domain：以子圖高度為座標範圍
            line=dict(color="red", width=2, dash="dash"),
            row=4, col=1
        )
        fig.add_annotation(
            x=bh_sterling, y=1.0, yref="y domain",
            text=f"Buy & Hold: {bh_sterling:.4f}",
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(color="red", size=11),
            row=4, col=1
        )

        # 2. 繪製 GP 平均值 (藍色虛線)
        fig.add_shape(
            type="line", x0=avg_sterling, x1=avg_sterling, y0=0, y1=1,
            yref="y domain",
            line=dict(color="blue", width=2, dash="dash"),
            row=4, col=1
        )
        fig.add_annotation(
            x=avg_sterling, y=0.0, yref="y domain",
            text=f"GP Mean: {avg_sterling:.4f}",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(color="blue", size=11),
            row=4, col=1
        )

        # ==================================================================
        
        
        # 表格 (一直顯示)
        df_display = pd.DataFrame(display_metrics)
        fig.add_trace(
            go.Table(
                header=dict(values=list(df_display.columns), fill_color='darkslateblue', align='left', font=dict(size=12, color='white')),
                cells=dict(values=[df_display[col] for col in df_display.columns], fill_color='ghostwhite', align='left', font=dict(size=11, color='black'))
            ), row=5, col=1
        )
        
        # 繪製第 5 列的圖形 (Anytime Behavior 或 Pareto Front)
        if obj_mode == "single":
            fig.add_trace(go.Scatter(
                x=generations, y=avg_anytime, name="平均最佳適應度 (Mean Best Fitness)", 
                line=dict(color='blue', width=2.5), 
                hovertemplate="<b>Gen %{x}</b><br>Fitness: %{y:.4f}<extra></extra>"
            ), row=6, col=1)
        else:
            # MOO 模式：為每一 Run 建立一個隱藏的 Pareto Scatter Trace (初始顯示最大代數的資料)
            def get_pareto_front(pts):
                if not pts: return [], []
                # 雙目標最大化：先用 X 排序，再過濾 Y
                sorted_pts = sorted(pts, key=lambda p: (p[0], p[1]), reverse=True)
                pareto_x, pareto_y = [], []
                max_y = -float('inf')
                for p in sorted_pts:
                    if p[1] > max_y:
                        pareto_x.append(p[0])
                        pareto_y.append(p[1])
                        max_y = p[1]
                # 依 X 遞增排序後回傳，以便連線
                sorted_idx = np.argsort(pareto_x)
                return [pareto_x[i] for i in sorted_idx], [pareto_y[i] for i in sorted_idx]

            global_moo_fronts_data = {}
            gen_steps = [g for g in range(5, max_gen + 1, 5)]
            if max_gen not in gen_steps: gen_steps.append(max_gen)

            # 計算每個世代的 Global Pareto Front
            for g in gen_steps:
                all_pts = []
                for i in range(1, num_runs + 1):
                    pts = moo_fronts_data[i].get(g, {'x':[], 'y':[]})
                    for x, y in zip(pts['x'], pts['y']):
                        all_pts.append((x, y))
                px, py = get_pareto_front(all_pts)
                global_moo_fronts_data[g] = {'x': px, 'y': py}

            # 找出計算 Hypervolume 的基準參考點 Reference Point (全域最低點再往下減去一點緩衝)
            min_x_val = min([min(moo_fronts_data[r][g]['x']) for r in moo_fronts_data for g in moo_fronts_data[r] if moo_fronts_data[r][g]['x']] + [0])
            min_y_val = min([min(moo_fronts_data[r][g]['y']) for r in moo_fronts_data for g in moo_fronts_data[r] if moo_fronts_data[r][g]['y']] + [0])
            ref_x = min_x_val - abs(min_x_val) * 0.05 - 0.01
            ref_y = min_y_val - abs(min_y_val) * 0.05 - 0.01

            # 計算各世代的 Hypervolume
            global_hv_data = {}
            run_hv_data = {i: {} for i in range(1, num_runs + 1)}
            
            for g in gen_steps:
                # 計算 Global Hypervolume
                hv = 0.0
                curr_x = ref_x
                px = global_moo_fronts_data[g]['x']
                py = global_moo_fronts_data[g]['y']
                for x, y in zip(px, py):
                    if x <= ref_x or y <= ref_y: continue
                    hv += (x - curr_x) * (y - ref_y)
                    curr_x = x
                global_hv_data[g] = hv
                
                # 計算每個獨立 Run 的 Hypervolume
                for i in range(1, num_runs + 1):
                    hv_run = 0.0
                    curr_x_run = ref_x
                    pts = moo_fronts_data[i].get(g, {'x':[], 'y':[]})
                    # 必須確保按照 X 遞增排序，計算才會正確
                    sorted_pts = sorted(zip(pts['x'], pts['y']), key=lambda p: p[0])
                    px_run = [p[0] for p in sorted_pts]
                    py_run = [p[1] for p in sorted_pts]
                    
                    for x, y in zip(px_run, py_run):
                        if x <= ref_x or y <= ref_y: continue
                        hv_run += (x - curr_x_run) * (y - ref_y)
                        curr_x_run = x
                    run_hv_data[i][g] = hv_run

            # 計算所有世代與所有 Run 的 Pareto Front 的 X、Y 範圍，用來固定坐標軸 (目標值可能為負)
            all_x = [v for r in moo_fronts_data.values() for g in r.values() for v in g['x']]
            all_y = [v for r in moo_fronts_data.values() for g in r.values() for v in g['y']]
            x_lo, x_hi = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
            y_lo, y_hi = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)
            pad_x = (x_hi - x_lo) * 0.05 or 0.05      # 兩側各留 5% 緩衝；範圍為 0 時給固定寬度
            pad_y = (y_hi - y_lo) * 0.05 or 0.05
            x_lo, x_hi = x_lo - pad_x, x_hi + pad_x
            y_lo, y_hi = y_lo - pad_y, y_hi + pad_y

            # 1. 新增總體 Pareto Front 軌跡 (給「統計結果」畫面預設顯示用)
            g_max = gen_steps[-1]
            fig.add_trace(go.Scatter(
                x=global_moo_fronts_data[g_max]['x'], 
                y=global_moo_fronts_data[g_max]['y'],
                mode='markers+lines',
                name=f"Global Pareto Front<br>(HV: {global_hv_data[g_max]:.4f})",
                marker=dict(size=10, color='gold', symbol='diamond', line=dict(width=1, color='darkgoldenrod')),
                line=dict(color='gold', width=2),
                visible=True, # 預設在 stats 顯示
                hovertemplate=f"<b>第一目標 ({obj1_label}):</b> %{x:.4f}<br><b>第二目標:</b> %{y:.4f}<extra>%{name}</extra>"
            ), row=6, col=1)

            # 2. MOO 模式：為每一 Run 建立一個隱藏的 Pareto Scatter Trace
            for i, name in enumerate(run_keys):
                run_idx = i + 1
                gen_data = moo_fronts_data[run_idx].get(max_gen, {'x':[], 'y':[]})
                fig.add_trace(go.Scatter(
                    x=gen_data['x'], y=gen_data['y'], 
                    mode='markers+lines', # 顯示點與連線
                    name=f"Pareto Front ({name})<br>(HV: {run_hv_data[run_idx][g_max]:.4f})",
                    marker=dict(size=8, color='crimson'),
                    line=dict(color='crimson', width=1, dash='dot'),
                    visible=False, # 預設隱藏，由下拉選單控制
                    hovertemplate=f"<b>第一目標 ({obj1_label}):</b> %{x:.4f}<br><b>第二目標:</b> %{y:.4f}<extra>%{name}</extra>"
                ), row=6, col=1)
                
            # 3. 新增純文字軌跡 (Text Traces)，將對應的 HV 顯示在圖表右下角
            text_x = x_hi - 0.05 * (x_hi - x_lo)
            text_y = y_lo + 0.05 * (y_hi - y_lo)

            # Global 文字軌跡
            fig.add_trace(go.Scatter(
                x=[text_x], y=[text_y],
                mode='text',
                text=[f"Hypervolume: {global_hv_data[g_max]:.4f}"],
                textposition='bottom left',
                textfont=dict(size=14, color='darkgoldenrod'),
                showlegend=False,
                hoverinfo='skip',
                visible=True
            ), row=6, col=1)

            # 各 Run 文字軌跡
            for i, name in enumerate(run_keys):
                run_idx = i + 1
                fig.add_trace(go.Scatter(
                    x=[text_x], y=[text_y],
                    mode='text',
                    text=[f"Hypervolume: {run_hv_data[run_idx][g_max]:.4f}"],
                    textposition='bottom left',
                    textfont=dict(size=14, color='crimson'),
                    showlegend=False,
                    hoverinfo='skip',
                    visible=False
                ), row=6, col=1)

            # MOO 的 X, Y 軸標籤
            fig.update_xaxes(title_text=f"第一目標 ({obj1_label})", range=[x_lo, x_hi], row=5, col=1)
            fig.update_yaxes(title_text=f"第二目標 ({obj2_label})", range=[y_lo, y_hi], row=5, col=1)
            
        # 繪製預設空溯源表 (為了在 "統計結果" 時顯示提示)
        empty_lineage = pd.DataFrame([{"說明": "👉 請從右上方下拉選單選擇特定 Run 以查看最佳解溯源"}])
        fig.add_trace(
            go.Table(
                header=dict(values=list(empty_lineage.columns), fill_color='lightgrey', align='left'),
                cells=dict(values=[empty_lineage[col] for col in empty_lineage.columns], align='left')
            ), row=7, col=1
        )
        
        # 把每個 Run 的溯源表加進去 (預設設為隱藏 visible=False)
        for name in run_keys:
            df_lin = curves_data[f"{name}_lineage"]
            fig.add_trace(
                go.Table(
                    header=dict(values=list(df_lin.columns), fill_color='darkcyan', align='left', font=dict(color='white')),
                    cells=dict(values=[df_lin[col] for col in df_lin.columns], fill_color='ghostwhite', align='left')
                ), row=7, col=1
            )
            fig.data[-1].visible = False # 隱藏
        
        # 高頻特徵子樹排行
        fig.add_trace(go.Bar(
            x=top_10_counts, y=display_labels, orientation='h',
            marker=dict(color='teal'), text=top_10_counts, textposition='auto', name="高頻積木", 
            hovertext=top_10_labels, hoverinfo="text+x",
            width=0.4  # 長條寬度
        ), row=8, col=1)
        
        
        # 5. 管理 Trace 索引與下拉互動選單
        N = len(run_keys)
        # trace 索引必須與上面加入圖層的順序一致 (Train 與 Test 分開)
        trace_map = {
            'param': 0, 'time': 1,
            'train_upper': 2, 'train_lower': 3, 'train_avg': 4,
            'test_upper': 5, 'test_lower': 6, 'test_avg': 7,
            'bh': 8,
            'runs': list(range(9, 9 + N)),
            'box_plot': 9 + N,
            'result_table': 10 + N,
        }
        curr_idx = 11 + N
        
        # Row 5 traces
        if obj_mode == "single":
            trace_map['anytime'] = [curr_idx]
            curr_idx += 1
        else:
            trace_map['global_pareto'] = curr_idx  # 註冊 Global Pareto
            curr_idx += 1
            trace_map['moo_paretos'] = list(range(curr_idx, curr_idx + N))
            curr_idx += N
            
            trace_map['global_hv_text'] = curr_idx
            curr_idx += 1
            trace_map['moo_hv_texts'] = list(range(curr_idx, curr_idx + N))
            curr_idx += N
            
        trace_map['empty_lin'] = curr_idx
        curr_idx += 1
        trace_map['run_lins'] = list(range(curr_idx, curr_idx + N))
        curr_idx += N
        trace_map['feature'] = curr_idx


        def get_vis_state(mode, run_index=None):
            """依畫面模式產生各 trace 的 visible 布林陣列"""
            vis = [False] * len(fig.data)
            # 永遠顯示的項目
            vis[trace_map['param']] = vis[trace_map['time']] = True
            vis[trace_map['box_plot']] = vis[trace_map['result_table']] = vis[trace_map['feature']] = True
            
            # 訓練結果顯示邏輯
            if mode == 'train_stats':
                vis[trace_map['train_upper']] = vis[trace_map['train_lower']] = vis[trace_map['train_avg']] = True
                vis[trace_map['bh']] = vis[trace_map['empty_lin']] = True
                if obj_mode == "single": 
                    vis[trace_map['anytime'][0]] = True
                else:
                    vis[trace_map['global_pareto']] = True 
                    vis[trace_map['global_hv_text']] = True 
            
            # 測試結果顯示邏輯
            elif mode == 'test_stats':
                vis[trace_map['test_upper']] = vis[trace_map['test_lower']] = vis[trace_map['test_avg']] = True
                vis[trace_map['bh']] = vis[trace_map['empty_lin']] = True
                if obj_mode == "single": 
                    vis[trace_map['anytime'][0]] = True
                else:
                    vis[trace_map['global_pareto']] = True 
                    vis[trace_map['global_hv_text']] = True 
            
            elif mode == 'bh':
                vis[trace_map['bh']] = vis[trace_map['empty_lin']] = True
                if obj_mode == "single": 
                    vis[trace_map['anytime'][0]] = True
                else:
                    vis[trace_map['global_pareto']] = True
                    vis[trace_map['global_hv_text']] = True 
                    
            elif mode == 'run' and run_index is not None:
                vis[trace_map['bh']] = True
                vis[trace_map['runs'][run_index]] = True
                vis[trace_map['run_lins'][run_index]] = True
                if obj_mode == "single":
                    vis[trace_map['anytime'][0]] = True
                else:
                    vis[trace_map['moo_paretos'][run_index]] = True 
                    vis[trace_map['moo_hv_texts'][run_index]] = True 
            return vis

        dropdown_buttons = [
            {'label': '訓練結果', 'method': 'update', 'args': [{'visible': get_vis_state('train_stats')}]},
            {'label': '測試結果', 'method': 'update', 'args': [{'visible': get_vis_state('test_stats')}]},
            {'label': 'Buy & Hold 基準線', 'method': 'update', 'args': [{'visible': get_vis_state('bh')}]}
        ]
        # 把後續的各 Run 選單加回去
        for i, name in enumerate(run_keys):
            dropdown_buttons.append({'label': f'{name}', 'method': 'update', 'args': [{'visible': get_vis_state('run', i)}]})
            
        # 設定主下拉選單
        updatemenus = [{
            'active': 0, 'buttons': dropdown_buttons, 'direction': "down",
            'showactive': True, 'x': 1.02, 'xanchor': "left", 'y': 1.0, 'yanchor': "top"
        }]

        # 6. MOO 專屬：拉桿控制面板 (連動軌跡更新)
        if obj_mode == "multi":
            gen_steps = [g for g in range(5, max_gen + 1, 5)]
            if max_gen not in gen_steps: gen_steps.append(max_gen)
            
            slider_steps = []
            for g in gen_steps:
                # 準備更新資料的陣列
                step_x = [global_moo_fronts_data[g]['x']]
                step_y = [global_moo_fronts_data[g]['y']]
                step_names = [f"Global Pareto Front<br>(HV: {global_hv_data[g]:.4f})"]
                
                text_x_list = [[text_x]]
                text_y_list = [[text_y]]
                text_content = [[f"Hypervolume: {global_hv_data[g]:.4f}"]]
                
                # 接著放入各 Run 的資料
                for i in range(1, num_runs + 1):
                    pts = moo_fronts_data[i].get(g, {'x':[], 'y':[]})
                    step_x.append(pts['x'])
                    step_y.append(pts['y'])
                    run_name = run_keys[i-1]
                    step_names.append(f"Pareto Front ({run_name})<br>(HV: {run_hv_data[i][g]:.4f})")
                    
                    text_x_list.append([text_x])
                    text_y_list.append([text_y])
                    text_content.append([f"Hypervolume: {run_hv_data[i][g]:.4f}"])
                
                step = {
                    'args': [
                        {   # 同時更新散點圖(前面) 與 純文字(後面)
                            'x': step_x + text_x_list, 
                            'y': step_y + text_y_list, 
                            'name': step_names + step_names, 
                            'text': [None]*(len(step_x)) + text_content
                        }, 
                        {'xaxis3.title.text': f"第一目標 ({obj1_label})"},  
                        [trace_map['global_pareto']] + trace_map['moo_paretos'] + [trace_map['global_hv_text']] + trace_map['moo_hv_texts']
                    ],
                    'label': str(g), 
                    'method': 'update'
                }
                slider_steps.append(step)

            # 加入拉桿
            sliders = [{
                'active': len(gen_steps) - 1, 
                'currentvalue': {'prefix': '當前演化進度: 世代 '},
                'pad': {'t': 40, 'b': 25},      
                'x': 0.1,        
                'y': 0.52,       
                'xanchor': 'left',
                'yanchor': 'top',
                'len': 0.9,      
                'steps': slider_steps
            }]
            
            fig.update_layout(sliders=sliders)

        fig.update_layout(
            updatemenus=updatemenus,
            showlegend=False,
            margin=dict(r=160, b=50) 
        )
        

        if obj_mode == "single":
            fig.update_layout(title=" GP 結果表 (single, 1: sterling ratio)", height=2600, template="plotly_white")
            output_html_path = mode_study_root / f"{obj_mode}_report.html"
            fig.write_html(str(output_html_path), include_plotlyjs='cdn')
        elif obj_mode == "multi":
            fig.update_layout(title=f" GP 結果表 (multi, 1: sterling ratio, 2: {multi_target}, selection_method: {resolve_method(final_config, moo_selection)})", height=2600, template="plotly_white")
            output_html_path = mode_study_root / f"{obj_mode}_{multi_target}_report.html"
            fig.write_html(str(output_html_path), include_plotlyjs='cdn')
            
        export_simplified_trees_to_txt(metrics_summary, mode_study_root)
        
        baseline_metrics = {
            "bh_sterling": bh_sterling,
            "bh_ror": bh_ror,
            "bh_mdd": bh_mdd
        }
        
        generate_llm_prompt_file(metrics_summary, test_times, mode_study_root, baseline_metrics)
        
        print("=" * 60)
        print(f"🎉 成功！動態交互對比網頁已生成。位於: {output_html_path.resolve()}")

        if is_formal_experiment:
            formal_results[current_mode] = {
                'metrics': pd.DataFrame(metrics_summary), 
                'curves': curves_data
            }
    
    # ================= 正式實驗：SOO 與 MOO 的統計比較與圖表 =================
    if is_formal_experiment:
        generate_formal_analysis(formal_results, study_root)

def generate_formal_analysis(formal_results, study_root):
    if "single" not in formal_results or "multi" not in formal_results:
        print("❌ 錯誤: 缺少單目標或多目標的正式實驗數據，無法進行對比分析。")
        return
    
    df_so = formal_results["single"]["metrics"]
    df_mo = formal_results["multi"]["metrics"]
    
    # =========================================================
    # 圖表 1：Train vs Test 分組長條圖
    # =========================================================
    # 單目標 (SOO) 的平均值
    so_train_sterling = df_so["Train_Sterling"].mean()
    so_test_sterling = df_so["Test_Sterling"].mean()
    so_train_return = df_so["Train_Return"].mean()
    so_test_return = df_so["Test_Return"].mean()
    # 以 Return / StdDev 當作簡化版 Sharpe ratio
    so_train_sharpe = (df_so["Train_Return"] / (df_so["Train_StdDev"] + 1e-6)).mean()
    so_test_sharpe = (df_so["Test_Return"] / (df_so["Test_StdDev"] + 1e-6)).mean()

    # 多目標 (MOO) 的平均值
    mo_train_sterling = df_mo["Train_Sterling"].mean()
    mo_test_sterling = df_mo["Test_Sterling"].mean()
    mo_train_return = df_mo["Train_Return"].mean()
    mo_test_return = df_mo["Test_Return"].mean()
    mo_train_sharpe = (df_mo["Train_Return"] / (df_mo["Train_StdDev"] + 1e-6)).mean()
    mo_test_sharpe = (df_mo["Test_Return"] / (df_mo["Test_StdDev"] + 1e-6)).mean()

    single_bars_y = [so_train_sterling, so_test_sterling, so_train_return, so_test_return, so_train_sharpe, so_test_sharpe]
    multi_bars_y = [mo_train_sterling, mo_test_sterling, mo_train_return, mo_test_return, mo_train_sharpe, mo_test_sharpe]

    x_axes_structure = [
        ['Sterling Ratio', 'Sterling Ratio', 'Total Return', 'Total Return', 'Sharpe Ratio', 'Sharpe Ratio'],
        ['Train', 'Test', 'Train', 'Test', 'Train', 'Test']
    ]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=x_axes_structure, y=single_bars_y, name='單目標 (SOO)', marker_color='rgb(55, 83, 109)'
    ))
    fig_bar.add_trace(go.Bar(
        x=x_axes_structure, y=multi_bars_y, name='多目標 (MOO)', marker_color='rgb(26, 118, 141)'
    ))
    fig_bar.update_layout(
        title='Figure 1: Mean results obtained for training and testing datasets',
        xaxis=dict(title='指標與數據集劃分', tickfont=dict(size=12)),
        yaxis=dict(title='平均數值 (Mean Value)', gridcolor='gainsboro'),
        barmode='group', template='plotly_white', height=500
    )
    html_fig1 = fig_bar.to_html(full_html=False, include_plotlyjs='cdn')

    # =========================================================
    # Mann-Whitney U 統計檢定 (測試集樣本外表現；SOO 與 MOO 為獨立樣本)
    # =========================================================
    stats_data = []
    # 比較 Test 階段的指標
    cols_to_test = [("Test_Sterling", "Sterling Ratio"), ("Test_Return", "Annual Return"), ("Test_Drawdown", "Max Drawdown")]
    
    for col_key, col_name in cols_to_test:
        so_vals = df_so[col_key].values
        mo_vals = df_mo[col_key].values
        
        if len(so_vals) >= 5 and not np.array_equal(so_vals, mo_vals):
            stat, p_val = stats.mannwhitneyu(so_vals, mo_vals, alternative="two-sided")   # SOO/MOO 各 run 種子不同 = 獨立樣本，不能用配對的 Wilcoxon signed-rank
            sig = "🟢 顯著差異" if p_val < 0.05 else "⚪ 無顯著差異"
        else:
            stat, p_val, sig = 0.0, 1.0, "N/A"
            
        stats_data.append({
            "比較指標": col_name,
            "SOO 平均值": f"{so_vals.mean():.4f}",
            "MOO 平均值": f"{mo_vals.mean():.4f}",
            "U 統計量": f"{stat:.2f}",
            "P 顯著性值 (P-Value)": f"{p_val:.5f}",
            "統計結論 (α=0.05)": sig
        })
    
    df_stats = pd.DataFrame(stats_data)
    html_stats_table = df_stats.to_html(classes="table table-striped table-hover text-center", index=False)

    # =========================================================
    # 圖表 2：Risk vs Return 散佈圖
    # =========================================================
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_so["Test_Return"], y=df_so["Test_StdDev"],
        mode='markers', name='單目標策略 (SOO)',
        marker=dict(color='blue', size=10, opacity=0.7)
    ))
    fig_scatter.add_trace(go.Scatter(
        x=df_mo["Test_Return"], y=df_mo["Test_StdDev"],
        mode='markers', name='多目標策略 (MOO)',
        marker=dict(color='red', size=10, opacity=0.7)
    ))
    fig_scatter.update_layout(
        title="Figure 8: 投資組合報酬與波動率比較 (Return vs Risk)",
        xaxis_title="報酬指標 (Mean Return)", yaxis_title="風險指標 (Standard Deviation / Volatility)",
        template="plotly_white", height=500
    )
    html_fig8 = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

    # =========================================================
    # 圖表 3：累積績效曲線
    # =========================================================
    fig_curves = go.Figure()
    for mode_name, color in zip(["single", "multi"], ["blue", "red"]):
        curves = formal_results[mode_name]["curves"]
        for idx, (curve_name, curve_vals) in enumerate(curves.items()):
            if curve_name == "Buy & Hold 基準線":
                if mode_name == "single": 
                    fig_curves.add_trace(go.Scatter(
                        y=curve_vals, mode='lines', line=dict(color='black', width=2, dash='dash'), name='Buy & Hold'
                    ))
            else:
                fig_curves.add_trace(go.Scatter(
                    y=curve_vals, mode='lines', line=dict(color=color, width=1), opacity=0.25,
                    showlegend=(idx == 0), name=f'{mode_name.upper()} 歷次執行'
                ))
    fig_curves.update_layout(
        title="Figure 9: 測試期累積績效曲線 (Equity Curves)",
        xaxis_title="交易時間點", yaxis_title="淨值 (Equity)",
        template="plotly_white", height=500
    )
    html_fig9 = fig_curves.to_html(full_html=False, include_plotlyjs=False)

    # =========================================================
    # 以 Bootstrap Tabs 把所有內容打包成單一 HTML 報告，存於 study_root
    # =========================================================
    integrated_report_html = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <title>深度穩健性檢定與學術論文整合報告</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 25px; background-color: #f8f9fa; }}
            .card {{ box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 20px; border: none; }}
            .nav-tabs .nav-link.active {{ font-weight: bold; }}
            th {{ background-color: #4A5568 !important; color: white !important; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h2 class="mb-2 text-dark"> Comprehensive Report </h2>
            <ul class="nav nav-tabs" id="reportTabs" role="tablist">
                <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#fig1-panel">Figure 1 (Train/Test 對比)</button></li>
                <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#stats-panel">Mann-Whitney U 統計檢定表</button></li>
                <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#fig8-panel">Figure 8 (風險與報酬)</button></li>
                <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#fig9-panel">Figure 9 (資產淨值曲線)</button></li>
            </ul>
            <div class="tab-content text-dark">
                <div class="tab-pane fade show active card bg-white p-3" id="fig1-panel">{html_fig1}</div>
                <div class="tab-pane fade card bg-white p-4" id="stats-panel">{html_stats_table}</div>
                <div class="tab-pane fade card bg-white p-3" id="fig8-panel">{html_fig8}</div>
                <div class="tab-pane fade card bg-white p-3" id="fig9-panel">{html_fig9}</div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

    output_report_path = study_root / "Comprehensive_Report.html"
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(integrated_report_html)
    
    print("📈 統計與圖表產出完成！已儲存為 HTML 檔案。")


if __name__ == "__main__":
    main()
