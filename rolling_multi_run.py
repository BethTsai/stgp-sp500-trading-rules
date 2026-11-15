"""
rolling_multi_run.py
==========================================================================
滾動時間窗口批次實驗：對每一組 (Train 4 年 / Val 1 年 / Test 1 年) 時間區段，
各自完整執行一次 multi_run_manager.py (單目標 SOO，每組 N 個 run)。

  組 1  Train 2014-2017  Val 2018  Test 2019
  組 2  Train 2015-2018  Val 2019  Test 2020
  ...
  組 7  Train 2020-2023  Val 2024  Test 2025

流程 (根目錄的 config.yaml 只讀不寫)：
  1. 讀取根目錄 config.yaml 當基準 (唯讀)，只改寫 data.oos_split 的 6 個日期，
     存成 gp_experiment_studies/rolling_<時間戳>/configs/test_<年>.yaml。
  2. 以子進程執行 multi_run_manager.main()：
     - 攔截 open()，使其讀取根目錄 config.yaml 時改讀該組設定 (與 multi_run_manager 的子進程相同做法)。
     - 由 stdin 自動回答互動選單：Q0=[1] 一般模式、Q1=run 次數、Q2=[1] 單目標。
  3. 比對執行前後的 study_* 資料夾，記錄每組對應哪個 study，寫入 rolling_index.csv，
     並把該組 config 複製一份到 study 資料夾內 (rolling_window_config.yaml)。
  4. 每組跑完後把 study 資料夾搬進 gp_experiment_studies/rolling/rolling_<時間戳>_<標籤>/，
     標籤由 config 自動產生 (資料檔_窗口模式_lr<min_long_ratio>)，可用 --tag 再附加說明；
     全部結束 (或中途失敗) 時在 gp_experiment_studies/EXPERIMENTS.md 追加一列紀錄。

用法：
    python rolling_multi_run.py --dry-run               # 只列出各組日期，不執行
    python rolling_multi_run.py                         # 跑全部 7 組，每組 30 run
    python rolling_multi_run.py --runs 1                # 每組只跑 1 run (快速檢查)
    python rolling_multi_run.py --only 2019 2020        # 只跑 Test 年為 2019、2020 的組
    python rolling_multi_run.py --val-years 2 --first-test-year 2020   # Val 改 2 年 (資料 2014 起，Test 最早只能 2020)
    python rolling_multi_run.py --tag note            # 資料夾名稱附加說明，如 rolling_..._features_stage27_calendar_half_lr0.2_val2y
    python rolling_multi_run.py --resume gp_experiment_studies/rolling/rolling_XXXX   # 中斷後續跑，跳過已完成的組
"""
import argparse
import copy
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
STUDIES_DIR = PROJECT_ROOT / "gp_experiment_studies"
ROLLING_DIR = STUDIES_DIR / "rolling"
EXPERIMENTS_MD = STUDIES_DIR / "EXPERIMENTS.md"

# 子進程啟動碼：攔截 config.yaml 的讀取後執行 multi_run_manager.main()
LAUNCHER = r'''
import builtins, os, sys
sys.path.insert(0, r"{root}")
os.chdir(r"{root}")
_ROOT_CFG = os.path.realpath(r"{root_cfg}")
_original_open = builtins.open
def _custom_open(file, *args, **kwargs):
    if isinstance(file, (str, os.PathLike)) and (
        str(file) == "config.yaml" or os.path.realpath(os.fspath(file)) == _ROOT_CFG):
        return _original_open(r"{cfg}", *args, **kwargs)
    return _original_open(file, *args, **kwargs)
builtins.open = _custom_open

import multi_run_manager
multi_run_manager.main()
'''


def build_windows(first_test_year, last_test_year, train_years, val_years=1):
    """回傳 [(test_year, {train_start, train_end, val_start, val_end, test_start, test_end}), ...]"""
    windows = []
    for test_year in range(first_test_year, last_test_year + 1):
        val_first = test_year - val_years     # Val 為 Test 前 val_years 個完整年度
        windows.append((test_year, {
            'train_start': f"{val_first - train_years}-01-01",
            'train_end': f"{val_first - 1}-12-31",
            'val_start': f"{val_first}-01-01",
            'val_end': f"{test_year - 1}-12-31",
            'test_start': f"{test_year}-01-01",
            'test_end': f"{test_year}-12-31",
        }))
    return windows


def auto_tag(cfg):
    """由 config 產生資料夾標籤，例如 features_stage27_calendar_half_lr0.2"""
    data = Path(cfg['data']['file_path']).stem
    mode = cfg.get('rsfgp', {}).get('window_mode', 'bars')
    lr = cfg.get('fitness', {}).get('min_long_ratio')
    return f"{data}_{mode}_lr{lr}"


def append_experiment_log(root, cfg, rows, args):
    """在 EXPERIMENTS.md 追加一列：資料夾、設定、各組狀態 (結果摘要留空，分析後再補)"""
    if not EXPERIMENTS_MD.exists():
        EXPERIMENTS_MD.write_text(
            "# 實驗紀錄\n\n"
            "| 資料夾 | 日期 | 資料 | 窗口 | min_long_ratio | 特徵數 | Train/Val | Test 年 | runs | 狀態 | 結果摘要 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    years = sorted(rows)
    ok = sum(rows[y].get('status') == 'ok' for y in years)
    feats = cfg.get('features', {}).get('selected_features_exact') or []
    line = (f"| {root.name} | {datetime.now():%Y-%m-%d} | {Path(cfg['data']['file_path']).name} "
            f"| {cfg.get('rsfgp', {}).get('window_mode', 'bars')} | {cfg.get('fitness', {}).get('min_long_ratio')} "
            f"| {len(feats) or 'all'} | {args.train_years}y/{args.val_years}y |{years[0]}–{years[-1]} | {args.runs} "
            f"| {ok}/{len(years)} ok | |\n")
    with open(EXPERIMENTS_MD, "a", encoding="utf-8") as f:
        f.write(line)


def list_studies():
    return {d.name for d in STUDIES_DIR.glob("study_*") if d.is_dir()} if STUDIES_DIR.exists() else set()


def load_index(index_path):
    if not index_path.exists():
        return {}
    with open(index_path, encoding="utf-8-sig") as f:
        return {int(r['test_year']): r for r in csv.DictReader(f)}


def write_index(index_path, rows):
    fields = ['test_year', 'train_start', 'train_end', 'val_start', 'val_end',
              'test_start', 'test_end', 'runs', 'status', 'study_dir', 'started', 'finished']
    with open(index_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({fld: rows[k].get(fld, '') for fld in fields})


def run_window(test_year, cfg_path, runs, log_path):
    """執行一組 multi_run_manager，回傳 (returncode, 新產生的 study 資料夾)"""
    launcher_path = cfg_path.with_suffix(".launcher.py")
    launcher_path.write_text(LAUNCHER.format(root=PROJECT_ROOT, root_cfg=PROJECT_ROOT / "config.yaml",
                                             cfg=cfg_path.resolve()), encoding="utf-8")
    answers = f"1\n{runs}\n1\n"   # Q0=[1] 一般測試模式, Q1=run 次數, Q2=[1] 單目標 (SOO)

    before = list_studies()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(launcher_path)], cwd=str(PROJECT_ROOT),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        proc.stdin.write(answers)
        proc.stdin.close()
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()

    new = sorted(list_studies() - before)
    study_dir = STUDIES_DIR / new[-1] if new else None
    return proc.returncode, study_dir


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=30, help="每組 run 次數 (預設 30)")
    ap.add_argument('--first-test-year', type=int, default=2019)
    ap.add_argument('--last-test-year', type=int, default=2025)
    ap.add_argument('--train-years', type=int, default=4, help="Train 期長度 (年，固定長度滑動)")
    ap.add_argument('--val-years', type=int, default=1, help="Val 期長度 (年)；2 年時 Val 依日曆半年切成 4 個窗口")
    ap.add_argument('--only', type=int, nargs='*', default=None, help="只跑指定 Test 年")
    ap.add_argument('--resume', default=None, help="續跑既有 rolling_* 資料夾，跳過 status=ok 的組")
    ap.add_argument('--tag', default='', help="附加在 rolling 資料夾名稱後的說明 (自動標籤之後)")
    ap.add_argument('--dry-run', action='store_true', help="只列出各組日期，不執行")
    a = ap.parse_args()

    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:   # 唯讀
        base_cfg = yaml.safe_load(f)

    soo_mode = base_cfg.get('evolution', {}).get('soo_training_mode')
    if soo_mode != 'oos':
        sys.exit(f"❌ config.yaml 的 evolution.soo_training_mode = {soo_mode!r}，"
                 f"必須是 'oos' 才會使用 oos_split 的日期，否則每組會跑出相同的結果。")

    windows = build_windows(a.first_test_year, a.last_test_year, a.train_years, a.val_years)
    if a.only:
        windows = [w for w in windows if w[0] in a.only]

    print(f"📅 共 {len(windows)} 組，每組 {a.runs} run (單目標 SOO)：")
    for test_year, d in windows:
        print(f"   Test {test_year}:  Train {d['train_start']} ~ {d['train_end']}  "
              f"Val {d['val_start']} ~ {d['val_end']}  Test {d['test_start']} ~ {d['test_end']}")
    if a.dry_run:
        return

    if a.resume:
        root = Path(a.resume)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
    else:
        tag = auto_tag(base_cfg) + (f"_val{a.val_years}y" if a.val_years != 1 else "") + (f"_{a.tag}" if a.tag else "")
        root = ROLLING_DIR / f"rolling_{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    shutil.copy(PROJECT_ROOT / "config.yaml", root / "base_config_backup.yaml")
    index_path = root / "rolling_index.csv"
    rows = load_index(index_path)
    print(f"📂 總索引存放於: {root}")

    for test_year, dates in windows:
        if rows.get(test_year, {}).get('status') == 'ok':
            print(f"⏭️  Test {test_year} 已完成 ({rows[test_year]['study_dir']})，跳過。")
            continue

        cfg = copy.deepcopy(base_cfg)
        cfg['data']['oos_split'].update(dates)
        cfg_path = root / "configs" / f"test_{test_year}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

        print("\n" + "=" * 60)
        print(f"🚀 [Test {test_year}] Train {dates['train_start']} ~ {dates['train_end']} | "
              f"Val {dates['val_start'][:4]}~{dates['val_end'][:4]} | Test {test_year}")
        print("=" * 60)
        row = {'test_year': test_year, **dates, 'runs': a.runs,
               'status': 'running', 'started': f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
        rows[test_year] = row
        write_index(index_path, rows)

        rc, study_dir = run_window(test_year, cfg_path, a.runs, root / "logs" / f"test_{test_year}.log")

        row['finished'] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        row['study_dir'] = study_dir.name if study_dir else ''
        if study_dir:
            shutil.copy(cfg_path, study_dir / "rolling_window_config.yaml")
            study_dir = Path(shutil.move(str(study_dir), str(root / study_dir.name)))   # 收進本次 rolling 資料夾
        # multi_run_manager 中途某個 run 失敗時會直接 return (rc 仍為 0)，所以另外確認每個 run 都有產出冠軍
        n_done = len(list(study_dir.glob("single/run_*/best_model_final.pkl"))) if study_dir else 0
        if rc != 0 or not study_dir:
            row['status'] = f'failed(rc={rc})'
        elif n_done < a.runs:
            row['status'] = f'incomplete({n_done}/{a.runs})'
        else:
            row['status'] = 'ok'
        write_index(index_path, rows)

        if row['status'] != 'ok':
            append_experiment_log(root, base_cfg, rows, a)
            sys.exit(f"❌ [Test {test_year}] 失敗，停止後續組。log: {root / 'logs' / f'test_{test_year}.log'}\n"
                     f"   修正後可用 --resume {root.relative_to(PROJECT_ROOT)} 續跑。")
        print(f"✅ [Test {test_year}] 完成 → {study_dir.name}")

    append_experiment_log(root, base_cfg, rows, a)
    print(f"\n🎉 全部完成，索引: {index_path}")


if __name__ == "__main__":
    main()
