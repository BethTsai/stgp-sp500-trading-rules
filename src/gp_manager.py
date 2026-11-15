"""
GPManager：封裝 DEAP 的 (μ+λ) 演化流程。
負責建立算子集與 toolbox、平行評估、去重存活選擇、Validation 選冠軍與 early stopping。
所有亂數都由固定種子推導 (每代 gen_seed、排序過的型別)，同一個 seed 可完全重現。
"""
import logging
import pickle
import re
import operator
import random
import numpy as np
import gc
from multiprocessing import Pool
from collections import defaultdict
from deap import base, creator, tools, gp
from src.primitives import create_primitive_set

# 每個子進程各自持有的 Evaluator 與算子集 (由 init_worker 建立；Windows 的 spawn 模式也適用)
WORKER_EVALUATOR = None
WORKER_PSET = None

_FLOAT = re.compile(r'-?\d+\.\d+(?:e[-+]?\d+)?')

def init_worker(config, seed):
    """子進程啟動時自行載入訓練資料，避免主進程把大型資料序列化傳過去。"""
    global WORKER_EVALUATOR, WORKER_PSET
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    from src.data_provider import DataProvider
    from src.evaluator_moo import MOOFitnessEvaluator
    from src.primitives import create_primitive_set

    dp = DataProvider(config=config, mode='train')
    WORKER_EVALUATOR = MOOFitnessEvaluator(data_provider=dp, config=config)
    WORKER_PSET = create_primitive_set(dp.feature_cols, config['features']['erc_range'])
    gc.collect()

def worker_evaluate(args):
    """評估一個個體。args = (個體, gen_seed)；同一代所有個體用同一個種子，結果不受進程分配順序影響。"""
    individual, gen_seed = args
    global WORKER_EVALUATOR, WORKER_PSET
    import random
    import numpy as np

    random.seed(gen_seed)
    np.random.seed(gen_seed)
    
    trading_func = gp.compile(expr=individual, pset=WORKER_PSET)
    return WORKER_EVALUATOR.evaluate_tree(trading_func, individual)

def custom_cxOnePoint(ind1, ind2):
    """
    STGP 單點交配。取代 DEAP 內建的 cxOnePoint：內建版從 set 隨機選型別，
    set 的順序取決於型別類別的記憶體位址，跨次執行無法重現；這裡先依字串排序再選。
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:], 1):
        types2[node.ret].append(idx)

    common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        sorted_common_types = sorted(list(common_types), key=lambda x: str(x))
        type_ = random.choice(sorted_common_types)

        node1 = random.choice(types1[type_])
        node2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(node1)
        slice2 = ind2.searchSubtree(node2)

        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2

def custom_mixed_mutation(individual, pset, expr_mut):
    """50% 節點替換突變、50% 子樹替換突變"""
    if random.random() < 0.5:
        return gp.mutNodeReplacement(individual, pset=pset)
    else:
        return gp.mutUniform(individual, expr=expr_mut, pset=pset)

def custom_varOr(mode, population, toolbox, lambda_, cxpb, mutpb):
    """
    產生 lambda_ 個子代：每個子代以機率 cxpb 由交配產生、以機率 mutpb 由突變產生，
    其餘 (cxpb + mutpb < 1 時) 直接複製。
    """

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    if mode == 'multi':
        # selTournamentDCD 需要先算好擁擠距離 (crowding distance)
        if not hasattr(population[0].fitness, "crowding_dist"):
            population = toolbox.select(population, len(population))

    offspring = []
    i = 0
    while i < lambda_:
        op_choice =  random.random()
        if op_choice < cxpb:
            selected = toolbox.select_mating(population, k=2)[:2]
            ind1, ind2 = [toolbox.clone(j) for j in selected]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
            offspring.append(ind1)
            offspring.append(ind2)
            i = i + 2
        elif op_choice < cxpb + mutpb:
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
            i = i + 1
        else:
            offspring.append(random.choice(population))

    return offspring


class GPManager:
    def __init__(self, config: dict, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.pset = None
        self.toolbox = None
        self.population = None

        self.base_seed = self.config.get('system', {}).get('seed', 42)
        n_workers = self.config.get('system', {}).get('n_workers', 4)

        self.current_gen = 0  # 用來推導每代的 gen_seed

        self._setup_deap()
        self.pool = Pool(processes=n_workers, initializer=init_worker, initargs=(self.config, self.base_seed))

        evo_cfg = self.config.get('evolution', {})
        obj_mode = evo_cfg.get('objective_mode', 'single')
        soo_mode = evo_cfg.get('soo_training_mode', 'regular')

        self.use_validation = (obj_mode == 'single' and soo_mode == 'oos')

        if self.use_validation:
            import copy
            from src.data_provider import DataProvider
            self.val_dp = DataProvider(config=self.config, mode='val')
            # Validation 用與訓練同一種適應度函數，Train / Val 分數才可比較
            self.val_evaluator = type(self.evaluator)(data_provider=self.val_dp, config=self.config)
            # window_mode='bars' 時，把 Val 窗口長度調成能剛好切滿 Val 期間、且最接近訓練窗口的長度
            # (否則約 1 年的 Val 只會切出 1 個窗口、尾巴被丟掉)；calendar_half 直接依日曆半年切
            wb = self.config['rsfgp'].get('window_bars', 756)
            val_cfg = copy.deepcopy(self.config)
            val_cfg['rsfgp']['num_samples'] = None   # Val 一律全窗口評估，結果才可重現
            calendar = val_cfg['rsfgp'].get('window_mode', 'bars') == 'calendar_half'
            if not calendar:
                n_win = max(1, round(self.val_dp.total_length / wb))   # 四捨五入，讓 val 窗口長度最接近訓練窗口
                val_cfg['rsfgp']['window_bars'] = self.val_dp.total_length // n_win
            self.val_dp.config = val_cfg
            self.val_evaluator.config = val_cfg
            if calendar:
                logging.info(f"🧪 Validation: {self.val_dp.total_length} 根 K 棒 -> 依日曆半年切成 {len(self.val_dp.get_rsfgp_windows())} 個窗口")
            else:
                logging.info(f"🧪 Validation: {self.val_dp.total_length} 根 K 棒 -> {n_win} 個窗口 x {val_cfg['rsfgp']['window_bars']} 根 (訓練窗口 {wb} 根)")
        else:
            self.val_dp = None
            self.val_evaluator = None

        self.patience_limit = evo_cfg.get('early_stopping_patience', 20)
        self.val_top_n = evo_cfg.get('val_top_n', 20)
        # 選用：改善幅度需超過 min_delta 才重置耐受度；前 warmup 代不累計耐受度
        self.es_min_delta = evo_cfg.get('early_stopping_min_delta', 0.0)
        self.es_warmup = evo_cfg.get('early_stopping_warmup', 0)
        # 冠軍規則：
        #   'val'           : 冠軍分數 = Val 適應度 (預設)
        #   'min_train_val' : 冠軍分數 = min(Train, Val)，以降低選擇偏誤 (winner's curse)
        self.champion_rule = evo_cfg.get('champion_rule', 'val')
        if self.champion_rule not in ('val', 'min_train_val'):
            raise ValueError(f"evolution.champion_rule 不支援: {self.champion_rule}")
        self.val_cache = {}   # str(個體) -> val 分數 (Val 是確定性的，菁英每代重複出現不必重算)

        # 存活選擇去重 (單目標)：避免重複個體佔滿族群、降低多樣性
        self.mode = obj_mode
        self.dedup_survival = evo_cfg.get('dedup_survival', True)
        self.const_round = int(evo_cfg.get('dedup_const_round', 2))
        self.immigrant_rate = float(evo_cfg.get('immigrant_rate', 0.0))
        self.dedup_fitness = bool(evo_cfg.get('dedup_by_fitness', False))

        self.patience_counter = 0
        self.best_val_score = -float('inf')
        self.best_val_ind = None

        self.mu = evo_cfg.get('pop_size', 100)
        self.lambda_ = evo_cfg.get('offspring_size', 100)

        self.population = self.toolbox.population(n=self.mu)
        self.history.update(self.population)  # 初代登錄進族譜

    def _setup_deap(self):
        # 註冊算子前先固定主進程亂數
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)

        evo_cfg = self.config.get('evolution', {})
        mode = evo_cfg.get('objective_mode', 'single')
        max_depth = evo_cfg.get('max_depth', 7)
        init_min = evo_cfg.get('init_min', 2)
        init_max = evo_cfg.get('init_max', 4)

        feature_cols = self.evaluator.dp.feature_cols
        erc_range = self.config['features'].get('erc_range', [-100.0, 100.0])
        self.pset = create_primitive_set(feature_cols, erc_range)

        weight_list = evo_cfg.get('weights', [1.0])
        weight_tuple = tuple(weight_list)

        # creator 是全域的，已建立過就不重複建立
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=weight_tuple)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=init_min, max_=init_max)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        mode = self.config['evolution'].get('objective_mode', 'single')
        if mode == 'single':
            self.toolbox.register("select_mating", tools.selTournament, tournsize=self.config['evolution'].get('tournsize', 3))
        else:
            self.toolbox.register("select_mating", tools.selTournamentDCD)
            self.toolbox.register("select", tools.selNSGA2)

        self.toolbox.register("mate", custom_cxOnePoint)

        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", custom_mixed_mutation, pset=self.pset, expr_mut=self.toolbox.expr_mut)

        # 族譜記錄 (trace_lineage 用) 與樹深度上限
        self.history = tools.History()
        self.toolbox.decorate("mate", self.history.decorator)
        self.toolbox.decorate("mutate", self.history.decorator)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))


    # ------------------------------------------------------------------
    # 存活選擇去重
    # ------------------------------------------------------------------
    def _key(self, ind) -> str:
        """結構鍵：樹字串，常數四捨五入到 dedup_const_round 位"""
        r = self.const_round
        return _FLOAT.sub(lambda m: f"{float(m.group()):.{r}f}", str(ind))

    def _unique_best(self, individuals, k):
        """依 fitness 由高到低，回傳最多 k 個不重複的個體 (同分取較短的樹)。
        dedup_by_fitness=True 時以「適應度(小數6位)」判重，否則以結構鍵判重。"""
        seen, out = set(), []
        for ind in sorted(individuals, key=lambda i: (-i.fitness.values[0], len(i))):
            key = f"{ind.fitness.values[0]:.6f}" if self.dedup_fitness else self._key(ind)
            if key in seen:
                continue
            seen.add(key)
            out.append(ind)
            if len(out) >= k:
                break
        return out

    def _make_immigrants(self, n, gen_seed):
        """產生 n 個全新隨機個體並評估 (登錄進族譜，trace_lineage 才找得到)"""
        imm = self.toolbox.population(n=n)
        self.history.update(imm)
        fits = self.pool.map(worker_evaluate, [(ind, gen_seed) for ind in imm])
        for ind, fit in zip(imm, fits):
            ind.fitness.values = fit
        return imm

    # ------------------------------------------------------------------
    # 演化一代
    # ------------------------------------------------------------------
    def run_generation(self):
        cxpb = self.config['evolution'].get('cxpb', 0.9)
        mutpb = self.config['evolution'].get('mutpb', 0.1)

        # 每代一個固定種子：同代所有個體在相同條件下評估
        self.current_gen += 1
        gen_seed = self.base_seed + (self.current_gen * 10000)

        invalid_pop = [ind for ind in self.population if not ind.fitness.valid]
        if invalid_pop:
            fits = self.pool.map(worker_evaluate, [(ind, gen_seed) for ind in invalid_pop])
            for ind, fit in zip(invalid_pop, fits):
                ind.fitness.values = fit

        # 產生 λ 個子代並評估
        offspring = custom_varOr(self.mode, self.population, self.toolbox, self.lambda_, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fits = self.pool.map(worker_evaluate, [(ind, gen_seed) for ind in invalid_ind])
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit

        # 存活選擇 (μ + λ)
        combined = self.population + offspring
        if self.mode == 'multi':
            # 多目標：NSGA-II 非支配排序 + 擁擠距離
            self.population = self.toolbox.select(combined, self.mu)
            return

        if not self.dedup_survival:
            self.population = tools.selBest(combined, self.mu)
            return

        n_imm = int(round(self.immigrant_rate * self.mu))
        if n_imm > 0:
            combined = combined + self._make_immigrants(n_imm, gen_seed)
        survivors = self._unique_best(combined, self.mu)
        if len(survivors) < self.mu:                  # 去重後不足 μ 個時，補入隨機新個體
            need = self.mu - len(survivors)
            logging.info(f"   🧬 [去重] 不重複個體僅 {len(survivors)} 個，補入 {need} 個隨機新個體")
            survivors = survivors + self._make_immigrants(need, gen_seed)
        self.population = sorted(survivors, key=lambda i: i.fitness.values[0], reverse=True)

        n_unique = len({self._key(i) for i in self.population})
        n_pheno = len({round(i.fitness.values[0], 6) for i in self.population})
        logging.info(f"   📊 族群不重複個體: {n_unique}/{self.mu} (不同適應度 {n_pheno}) | Train 最佳: {self.population[0].fitness.values[0]:.4f}")

    def save_best_model(self, filepath: str):
        import gzip
        mode = self.config['evolution'].get('objective_mode', 'single')

        if self.use_validation:
            # 單目標 + Validation：存 Validation 歷史最佳 (冠軍)
            if self.best_val_ind is not None:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(self.best_val_ind, f)
                logging.info(f"已儲存 Validation 歷史最佳個體，盲測分數: {self.best_val_score:.4f}")
        elif mode == 'single':
            # 單目標、無 Validation：存 Train 第一名
            best_ind = tools.selBest(self.population, 1)[0]
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(best_ind, f)
            logging.info(f"已儲存最佳個體，其適應度為: {best_ind.fitness.values[0]:.4f}")

        else:
            # 多目標：存第一層 Pareto front
            front = tools.sortNondominated(self.population, len(self.population), first_front_only=True)[0]
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(front, f)
            logging.info(f"已儲存 Pareto Front，包含 {len(front)} 個非支配策略！")

    def close(self):
        self.pool.close()
        self.pool.join()

    def check_early_stopping(self) -> tuple[bool, float]:
        """Validation 評估與 early stopping。候選為 Train 前 val_top_n 名的不重複個體。
        多目標 (MOO) 不支援 Early Stopping。"""
        if not self.use_validation:
            return False, 0.0

        cur_best, cur_ind = -float('inf'), None
        for ind in self._unique_best(self.population, self.val_top_n):
            key = str(ind)
            if key not in self.val_cache:
                score = float(self.val_evaluator.evaluate_tree(self.toolbox.compile(expr=ind), ind)[0])
                self.val_cache[key] = score if np.isfinite(score) else -float('inf')
            score = self.val_cache[key]
            if self.champion_rule == 'min_train_val':
                score = min(score, float(ind.fitness.values[0]))
            if score > cur_best:
                cur_best, cur_ind = score, ind

        improved = cur_best > self.best_val_score + self.es_min_delta
        if cur_ind is not None and cur_best > self.best_val_score:
            # 分數高於歷史最佳即更新冠軍；min_delta 只決定耐受度是否歸零
            self.best_val_score = cur_best
            self.best_val_ind = self.toolbox.clone(cur_ind)

        if improved:
            self.patience_counter = 0
            logging.info(f"   🏆 [Val 盲測破紀錄] 新高: {self.best_val_score:.4f} (耐受度歸零)")
        elif self.current_gen <= self.es_warmup:
            logging.info(f"   ⏳ [Val 暖機中 {self.current_gen}/{self.es_warmup}] 當代最佳: {cur_best:.4f} | 歷史高點: {self.best_val_score:.4f} (不累計耐受度)")
        else:
            self.patience_counter += 1
            logging.info(f"   ⚠️ [Val 盲測未突破] 當代最佳: {cur_best:.4f} | 歷史高點: {self.best_val_score:.4f} | 耐受度: {self.patience_counter}/{self.patience_limit}")

        return (self.patience_counter >= self.patience_limit), self.best_val_score

    def save_history(self, filepath: str):
        """只保留最後一代存活者及其所有祖先的族譜節點，以 gzip 存檔。"""
        survivors = set()
        for ind in self.population:
            if hasattr(ind, "history_index"):
                survivors.add(ind.history_index)

        active_nodes = set(survivors)
        nodes_to_check = list(survivors)

        while nodes_to_check:
            current = nodes_to_check.pop()
            parents = self.history.genealogy_tree.get(current, ())
            for p in parents:
                if p not in active_nodes:
                    active_nodes.add(p)
                    nodes_to_check.append(p)

        all_nodes = list(self.history.genealogy_history.keys())
        for node in all_nodes:
            if node not in active_nodes:
                del self.history.genealogy_history[node]
                if node in self.history.genealogy_tree:
                    del self.history.genealogy_tree[node]

        import gzip
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(self.history, f)

        logging.info(f"📜 已儲存修剪並壓縮後之演化系譜 (History) 至: {filepath} (保留 {len(active_nodes)} 個關鍵節點)")
