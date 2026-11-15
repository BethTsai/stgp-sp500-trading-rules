"""
MOO 代表策略挑選：從一個 run 的 Pareto 前緣中選出一個代表策略。

挑選方法由 config 的 moo_validation.selection 指定；未設定時使用 multi_run_manager 選單的選擇。

  方法              說明                                                        適用
  ----------------  ----------------------------------------------------------  ---------------------------------
  val_champion      演化過程中追蹤的 Val 冠軍 (moo_champion.pkl)                 啟用 MOO 驗證的 run
  val_front         以 Val 重新評估最後一代前緣，取冠軍分數最高者                 所有 MOO run
  topsis            TOPSIS；可先排除無效點 (moo_validation.filter_degenerate)     -
  knee_point        拐點法；可先排除無效點                                       -
  oos_sterling      整段 Val 的年化 Sterling ratio 最高者                         -
  (別名) oos = oos_sterling；knee = knee_point

冠軍分數規則沿用 evolution.champion_rule ('val' / 'min_train_val')。

呼叫方式 (multi_run_manager.py)：
    ind, info = pick_moo_individual(raw_model, run_dir, final_config, menu_method=moo_selection)
回傳 (個體, info)；info 包含 method_used、fallback 與各項分數。
"""
import gzip
import logging
import pickle
from pathlib import Path

import numpy as np

from src.moo_validation import MOOValidator, get_mv_cfg

METHOD_ALIASES = {
    'val_champion': 'val_champion', 'champion': 'val_champion',
    'val_front': 'val_front',
    'topsis': 'topsis',
    'knee_point': 'knee_point', 'knee': 'knee_point',
    'oos_sterling': 'oos_sterling', 'oos': 'oos_sterling',
}


def resolve_method(config: dict, menu_method: str = None) -> str:
    """config.moo_validation.selection 優先；沒寫才用選單的值；都沒有 -> topsis"""
    mv = config.get('moo_validation') or {}
    raw = mv.get('selection') or menu_method or 'topsis'
    name = str(raw).strip().lower()
    if name not in METHOD_ALIASES:
        raise ValueError(f"moo_validation.selection 不支援: {raw} (可用: {sorted(set(METHOD_ALIASES.values()))})")
    return METHOD_ALIASES[name]


# ----------------------------------------------------------------------
# 前緣過濾與 TOPSIS / 拐點法 (另處理無效點與單點前緣)
# ----------------------------------------------------------------------
def filter_degenerate_front(front, clip: float):
    """排除無效點 (第一目標 <= -clip)；若全部無效則回傳原前緣。"""
    kept = [i for i in front if i.fitness.values[0] > -clip + 1e-9]
    return kept if kept else list(front)


def _topsis(front):
    fits = np.array([i.fitness.values for i in front], dtype=float)
    lo, hi = fits.min(axis=0), fits.max(axis=0)
    rng = hi - lo
    rng[rng == 0] = 1e-6                           # 某個目標全部相同時避免除以 0
    dist = np.linalg.norm((fits - lo) / rng - 1.0, axis=1)   # 兩目標皆最大化，理想點 = (1, 1)
    return front[int(np.argmin(dist))]


def _knee(front):
    fits = np.array([i.fitness.values for i in front], dtype=float)
    # 用前緣內 min-max 正規化後再算距離，避免兩個目標尺度差異大時 (例如 ±1 對 ±10) 偏向其中一軸
    lo, hi = fits.min(axis=0), fits.max(axis=0)
    rng = hi - lo
    rng[rng == 0] = 1e-6
    f = (fits - lo) / rng
    p1, p2 = f[np.argmin(f[:, 0])], f[np.argmax(f[:, 0])]
    line = p2 - p1
    L = np.linalg.norm(line)
    if L < 1e-9:                                   # 前緣縮成一點
        return front[0]
    d = np.abs(line[0] * (f[:, 1] - p1[1]) - line[1] * (f[:, 0] - p1[0])) / L
    return front[int(np.argmax(d))]


# ----------------------------------------------------------------------
# 讀冠軍檔
# ----------------------------------------------------------------------
def load_champion(run_dir):
    """讀 run_dir/moo_champion.pkl；沒有檔案回傳 None。
    注意：unpickle 個體需要 deap.creator 已經建好 Individual / FitnessMax
    (multi_run_manager 在讀模型之前就呼叫過 init_deap_environment，所以這裡不用處理)。"""
    p = Path(run_dir) / 'moo_champion.pkl'
    if not p.exists():
        return None
    with gzip.open(p, 'rb') as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def pick_moo_individual(front, run_dir, config: dict, menu_method: str = None, validator: MOOValidator = None):
    """
    front       最後一代的 Pareto 前緣 (best_model_final.pkl 讀出的 list)
    run_dir     該 run 的資料夾 (val_champion 要在裡面找 moo_champion.pkl)
    config      該次實驗的 config (multi_run_manager 的 final_config)
    menu_method 選單選的方法；config 有寫 moo_validation.selection 時會被覆蓋
    validator   選填：傳入共用的 MOOValidator，多個 run 可共用 Val 資料與快取

    回傳 (individual, info)。
    """
    if not front:
        return None, {'method_used': None, 'note': '空前緣'}
    method = resolve_method(config, menu_method)
    mv = get_mv_cfg(config)
    clip = float((config.get('fitness') or {}).get('score_clip', 10.0))
    rule = (config.get('evolution') or {}).get('champion_rule', 'val')
    info = {'method_requested': method, 'method_used': method, 'fallback': False}

    def _validator():
        return validator if validator is not None else MOOValidator(config, champion_rule=rule)

    if method == 'val_champion':
        rec = load_champion(run_dir)
        if rec is not None:
            info.update(gen=rec['gen'], champion_score=rec['champion_score'], val_score=rec['val_score'],
                        train_score=rec['train_score'], rule_at_train=rec['rule'])
            if rec['rule'] != rule:
                logging.warning(f"⚠️ 冠軍是用 champion_rule='{rec['rule']}' 演化出來的，但目前 config 是 '{rule}'，以檔案為準。")
            return rec['individual'], info
        # 找不到冠軍檔 (未啟用 MOO 驗證的 run)：改為以 Val 重新評估前緣，並記錄於 info
        logging.warning(f"⚠️ {run_dir} 沒有 moo_champion.pkl，退回 val_front (事後在最後一代前緣上挑 Val 冠軍)。")
        method = 'val_front'
        info.update(method_used='val_front', fallback=True)

    if method == 'val_front':
        v = _validator()
        cands = v.pick_candidates(front)
        ind, score = v.best_of(cands)
        if ind is None:                            # 前緣全是無效點
            info.update(note='前緣沒有有效個體，退回 front[0]')
            return front[0], info
        rec = v.scores(ind)
        info.update(champion_score=score, val_score=rec['val'], train_score=rec['train'], n_candidates=len(cands))
        return ind, info

    if method == 'oos_sterling':
        v = _validator()
        pool = filter_degenerate_front(front, clip) if mv['filter_degenerate'] else list(front)
        best = max(pool, key=v.val_sterling)
        info.update(val_sterling=v.val_sterling(best))
        return best, info

    pool = filter_degenerate_front(front, clip) if mv['filter_degenerate'] else list(front)
    return (_topsis(pool) if method == 'topsis' else _knee(pool)), info
