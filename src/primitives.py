"""
STGP 算子集 (Primitive Set)。
依特徵名稱決定型別，只替資料中實際出現的型別註冊比較 / 加減算子與隨機常數 (ERC)，
確保樹只會比較同一尺度的量。根節點輸出 bool (True = 下一根 K 棒持有)。
"""
import operator
import random
import deap.gp

class PriceType(float): pass
class RatioType(float): pass
class RankType(float): pass    # 落在 [0, 1] 的排名或 IBS

def gen_ratio_erc(): return RatioType(random.uniform(-1.0, 1.0))
def gen_rank_erc(): return RankType(random.uniform(0.0, 1.0))

def get_feature_type(col_name: str):
    """依欄位名稱判斷特徵型別；新增特徵時必須遵守這個命名規則。"""
    if col_name.startswith('Rank_') or col_name.startswith('IBS'):
        return RankType

    if col_name in ['close', 'open', 'high', 'low'] or \
       col_name.startswith('SMA') or \
       col_name.startswith('EMA') or \
       (col_name.startswith('BB') and ('Basis' in col_name or 'Upper' in col_name or 'Lower' in col_name)):
        return PriceType

    if col_name.startswith(('ROC', 'HV', 'alpha', 'beta', 'sharpe', 'RSI')):
        return RatioType

    return float

def create_primitive_set(feature_cols: list, erc_range: list):
    """建立 PrimitiveSetTyped：輸入為各特徵 (依名稱決定型別)，輸出為 bool。"""
    in_types = [get_feature_type(col) for col in feature_cols]
    pset = deap.gp.PrimitiveSetTyped("MAIN", in_types, bool)

    present_types = set(in_types)

    pset.addPrimitive(operator.and_, [bool, bool], bool, name="and_")
    pset.addPrimitive(operator.or_, [bool, bool], bool, name="or_")

    # 每種型別只能和同型別比較 / 加減
    if PriceType in present_types:
        pset.addPrimitive(operator.gt, [PriceType, PriceType], bool, name="gt_price")
        pset.addPrimitive(operator.lt, [PriceType, PriceType], bool, name="lt_price")
        pset.addPrimitive(operator.add, [PriceType, PriceType], PriceType, name="add_price")
        pset.addPrimitive(operator.sub, [PriceType, PriceType], PriceType, name="sub_price")

    if RatioType in present_types:
        pset.addPrimitive(operator.gt, [RatioType, RatioType], bool, name="gt_ratio")
        pset.addPrimitive(operator.lt, [RatioType, RatioType], bool, name="lt_ratio")
        pset.addPrimitive(operator.add, [RatioType, RatioType], RatioType, name="add_ratio")
        pset.addPrimitive(operator.sub, [RatioType, RatioType], RatioType, name="sub_ratio")

        # 同一種 ERC 註冊 5 次，提高常數在樹中被選到的機率；範圍 U(-1, 1)
        for i in range(5):
            pset.addEphemeralConstant(
                name=f'c_ratio_{i}',
                ephemeral=gen_ratio_erc,
                ret_type=RatioType
            )

        # 固定門檻常數：0.0215 ~ 0.1215 間 5 個等距值
        import numpy as np
        micro_thresholds = np.linspace(0.0215, 0.1215, 5)
        for i, val in enumerate(micro_thresholds):
            pset.addTerminal(RatioType(val), RatioType, name=f"c_micro_{i}")

    if RankType in present_types:
        pset.addPrimitive(operator.gt, [RankType, RankType], bool, name="gt_rank")
        pset.addPrimitive(operator.lt, [RankType, RankType], bool, name="lt_rank")
        pset.addPrimitive(operator.add, [RankType, RankType], RankType, name="add_rank")
        pset.addPrimitive(operator.sub, [RankType, RankType], RankType, name="sub_rank")

        # 排名常數 U(0, 1)，同樣註冊 5 次
        for i in range(5):
            pset.addEphemeralConstant(
                name=f'c_rank_{i}',
                ephemeral=gen_rank_erc,
                ret_type=RankType
            )

    # 未分類的特徵 (如成交量) 用一般 float 算子
    if float in present_types:
        pset.addPrimitive(operator.gt, [float, float], bool, name="gt")
        pset.addPrimitive(operator.lt, [float, float], bool, name="lt")
        pset.addPrimitive(operator.add, [float, float], float, name="add")
        pset.addPrimitive(operator.sub, [float, float], float, name="sub")

    # 把 DEAP 預設的 ARG0, ARG1... 改名為實際特徵名稱
    arg_names = {f"ARG{i}": col for i, col in enumerate(feature_cols)}
    pset.renameArguments(**arg_names)

    pset.addTerminal(True, bool, name="TrueConst")
    pset.addTerminal(False, bool, name="FalseConst")

    return pset
