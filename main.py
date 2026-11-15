"""
單次演化的進入點。
讀取 config.yaml，必要時先產生特徵矩陣，建立 DataProvider / Evaluator / GPManager，
執行演化迴圈 (含 Validation early stopping)，並把每代冠軍存成 checkpoint。
"""
import os
os.environ['PYTHONHASHSEED'] = '0'

import yaml
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

from src.data_provider import DataProvider
from src.evaluator_moo import MOOFitnessEvaluator
from src.gp_manager_moo import MOOGPManager as GPManager

from deap import creator, base, gp


def init_deap_from_config():
    """依 config 的 evolution.weights 建立 DEAP 的 Fitness 與 Individual (單目標 [1.0]、多目標如 [1.0, 1.0])。"""
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
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"❌ 找不到設定檔 {config_path}，請確認檔案路徑。")
        exit(1)

def setup_logging_and_backup(config, config_path="config.yaml"):
    """備份本次使用的 config，並同時輸出 log 到檔案與終端機。"""
    checkpoint_dir = Path(config.get('system', {}).get('checkpoint_dir', 'checkpoints_multi'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    shutil.copy(config_path, checkpoint_dir / f"config_backup_{timestamp}.yaml")

    logging.getLogger().handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(checkpoint_dir / f"training_log_{timestamp}.log", encoding='utf-8')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

def main():
    config = load_config()

    import random
    import numpy as np
    seed = config.get('system', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)

    setup_logging_and_backup(config)

    logging.info("🚀 [系統啟動] 量化遺傳規劃 (GP) 演化引擎準備就緒...")
    start_time = time.time()

    # Phase 1：建立核心元件 (單/多目標由 config 的 objective_mode 決定，類別固定用 MOO 版)
    logging.info("⚙️ [Phase 1] 實例化核心演化組件...")
    dp = DataProvider(config=config)
    evaluator = MOOFitnessEvaluator(data_provider=dp, config=config)
    manager = GPManager(config=config, evaluator=evaluator)

    # Phase 2：演化迴圈
    max_gen = config['evolution']['max_gen']
    logging.info(f"🧬 [Phase 2] 啟動演化迴圈！預定執行世代數: {max_gen} 代...")
    print("-" * 50)

    checkpoint_path = config.get('system', {}).get('checkpoint_dir', 'checkpoints_multi')
    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(exist_ok=True)

    try:
        last_gen = 0
        for gen in range(max_gen):
            last_gen = gen + 1
            logging.info(f"🔄 --- Generation {gen + 1}/{max_gen} ---")

            manager.run_generation()
            is_stop, _ = manager.check_early_stopping()

            save_path = checkpoint_dir / f"best_model_gen_{last_gen}.pkl"
            manager.save_best_model(save_path)

            if is_stop:
                logging.warning(f"🛑 觸發 Early Stopping！連續 {manager.patience_limit} 代無進步，強制中止於第 {last_gen} 代。")
                break
    finally:
        # 不論正常結束或中途中斷，都把最後一代的冠軍複製為 best_model_final.pkl
        last_model_path = checkpoint_dir / f"best_model_gen_{last_gen}.pkl"
        final_model_path = checkpoint_dir / "best_model_final.pkl"
        if last_model_path.exists():
            shutil.copy(last_model_path, final_model_path)

        history_path = checkpoint_dir / "evolution_history.pkl"
        manager.save_history(history_path)

        manager.close()
        logging.info("🧹 已徹底關閉平行運算資源。")

    print("-" * 50)
    logging.info(f"✅ 演化任務完成！總耗時: {time.time() - start_time:.2f} 秒")
    logging.info(f"🏆 歷代最佳策略模型已序列化至 {checkpoint_dir}/ 目錄。")

if __name__ == "__main__":
    main()
