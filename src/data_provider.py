"""
DataProvider：載入特徵 pkl，依模式 (train / val / test) 切出時間區段與股票，
並把訓練期切成 RSFGP 窗口，以 NumPy 陣列交給 Evaluator 做向量化運算。
"""
import pandas as pd
import numpy as np
import random
import logging
import pickle

class DataProvider:
    def __init__(self, config: dict, mode='train'):
        self.config = config
        self.mode = mode

        data_path = config['data']['file_path']
        logging.info(f"💾 DataProvider: 準備載入 {data_path} ...")

        import gzip
        try:
            # 以檔頭 magic number 判斷是否為 gzip 壓縮的 pickle
            with open(data_path, "rb") as test_f:
                is_gzipped = test_f.read(2) == b'\x1f\x8b'

            if is_gzipped:
                with gzip.open(data_path, "rb") as f:
                    full_dataset = pickle.load(f)
            else:
                with open(data_path, "rb") as f:
                    full_dataset = pickle.load(f)

        except FileNotFoundError:
            raise FileNotFoundError(f"找不到檔案 {data_path}，請先準備資料並執行 python data_pipeline/features_pit.py (見 README 的「資料」一節)")

        data_conf = config['data']
        evo_conf = config.get('evolution', {})

        obj_mode = evo_conf.get('objective_mode', 'single')
        soo_mode = evo_conf.get('soo_training_mode', 'regular')
        moo_selection = evo_conf.get('moo_selection_method', 'topsis')

        # 只有 oos 模式 (有 Validation) 才用 data.oos_split 的日期，否則用 data.train_* / test_*
        has_oos_block = 'oos_split' in data_conf
        use_oos_dates = False

        if has_oos_block:
            if obj_mode == 'single' and soo_mode == 'oos':
                use_oos_dates = True
            elif obj_mode == 'multi' and moo_selection == 'oos':
                use_oos_dates = True

        # train_symbols 可寫 'all'：使用 pkl 內全部股票 (股票加入 / 剔除由 pkl 的 'active' 矩陣處理)
        def _expand(syms):
            return list(full_dataset['close'].columns) if syms == 'all' else list(syms)

        if mode == 'train':
            start_date = data_conf['oos_split']['train_start'] if use_oos_dates else data_conf['train_start']
            end_date = data_conf['oos_split']['train_end'] if use_oos_dates else data_conf['train_end']
            target_symbols = _expand(data_conf['train_symbols'])

        elif mode == 'val':
            if not use_oos_dates:
                raise ValueError("當前設定為 regular 模式或未設定 oos_split，無法讀取 Validation 資料！")
            start_date = data_conf['oos_split']['val_start']
            end_date = data_conf['oos_split']['val_end']
            target_symbols = _expand(data_conf['train_symbols'])

        elif mode == 'test':
            start_date = data_conf['oos_split']['test_start'] if use_oos_dates else data_conf['test_start']
            end_date = data_conf['oos_split']['test_end'] if use_oos_dates else data_conf['test_end']
            target_symbols = _expand(data_conf['train_symbols']) + list(data_conf.get('test_only_symbols', []))

        # 結束日只寫日期 (例如 "2019-12-31") 時，要包含當天所有 K 棒，而不是停在當天 00:00
        end_ts = pd.Timestamp(end_date)
        if end_ts == end_ts.normalize():
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        self.data = {}
        for feat_name, df in full_dataset.items():
            available_symbols = [s for s in target_symbols if s in df.columns]
            self.data[feat_name] = df.loc[pd.Timestamp(start_date):end_ts, available_symbols]

        # 只保留切片後的資料，釋放完整資料集的記憶體
        del full_dataset
        import gc
        gc.collect()

        # 'active'、'filled' 是遮罩，'close'、'eval_price' 是價格，都不是特徵
        exclude_cols = ['close', 'eval_price', 'active', 'filled']
        all_features = sorted([k for k in self.data.keys() if k not in exclude_cols])

        # 依 config.features.selected_features_exact 精確挑選特徵 (空 = 使用全部特徵)
        exact_names = config.get('features', {}).get('selected_features_exact', [])
        if exact_names:
            from src.feature_utils import select_features_exact, check_feature_scales
            self.feature_cols = select_features_exact(all_features, exact_names)
            if config.get('features', {}).get('enforce_unit_scale', True):
                check_feature_scales(self.feature_cols, self.data)
            logging.info(f"🎯 精確名稱比對：使用 {len(self.feature_cols)} 個特徵: {self.feature_cols}")
        else:
            self.feature_cols = all_features
            logging.info(f"🔍 未啟用特徵篩選，載入全部 {len(self.feature_cols)} 個特徵。")

        first_feat = self.feature_cols[0]
        self.total_length = len(self.data[first_feat])
        self.num_assets = len(self.data[first_feat].columns)

        self.full_close = self.data['close'].values
        self.full_eval_price = self.data['eval_price'].values

        # point-in-time 股票池 (T, N) bool；舊 pkl 沒有 'active' -> None (所有股票一直在池內)
        self.full_active = self.data['active'].values.astype(bool) if 'active' in self.data else None

        # 各股以 eval_price 計算的逐根報酬 (第 0 根為 0)
        self.full_market_returns = np.zeros_like(self.full_close)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.full_market_returns[1:, :] = (self.full_eval_price[1:, :] - self.full_eval_price[:-1, :]) / self.full_eval_price[:-1, :]
        self.full_market_returns[np.isnan(self.full_market_returns) | np.isinf(self.full_market_returns)] = 0.0

        # 特徵依 GP 函數的參數順序排成 list
        self.full_features_list = [self.data[col].values for col in self.feature_cols]

        self._cached_windows = None
        self._cached_seed_state = None
        self._warned_short = False   # 日曆窗口「不足半年」的警告只印一次

        logging.info(f"✅ DataProvider 初始化完成！區間: {start_date} ~ {end_date}")
        logging.info(f"📊 資料維度: {self.total_length} 根 K 棒 x {self.num_assets} 支股票")

    def get_rsfgp_windows(self) -> list:
        """把資料切成互不重疊的 RSFGP 窗口，回傳 Evaluator 使用的 dict list。"""
        # 以亂數狀態當快取鍵：同一代內同一個 worker 不必重複切片
        current_state = random.getstate()
        if self._cached_seed_state == current_state and self._cached_windows is not None:
            return self._cached_windows

        window_bars = self.config['rsfgp'].get('window_bars', 756)   # 65 分鐘 K 棒，半年約 756 根
        num_samples = self.config['rsfgp'].get('num_samples', None)  # None = 使用全部窗口
        total_length = self.total_length

        window_mode = self.config['rsfgp'].get('window_mode', 'bars')
        if window_mode == 'calendar_half':
            # 依日曆半年 (1~6 月、7~12 月) 切窗口：長度不必相同，但每根 K 棒都會被用到
            idx = self.data['close'].index
            half = (idx.year * 2 + (idx.month > 6)).to_numpy()
            bounds = [0, *(np.flatnonzero(np.diff(half)) + 1), total_length]
            spans = list(zip(bounds[:-1], bounds[1:]))
            short = [(str(idx[s].date()), e - s) for s, e in spans if e - s < 0.8 * window_bars]
            if short and not self._warned_short:      # 期間沒對齊 1/1、7/1 時才會出現
                logging.warning(f"⚠️ 有不足半年的窗口 (起始日, K 棒數): {short}")
                self._warned_short = True
        elif window_mode == 'bars':
            if total_length < window_bars:
                raise ValueError(
                    f"🚨 資料長度嚴重不足！當前資料僅有 {total_length} 根 K 棒，"
                    f"但 config 要求的單一窗口大小為 {window_bars} 根。請縮小窗口或擴大回測區間！"
                )
            # 每 window_bars 根切一刀，除不盡的尾巴會被捨棄
            spans = [(s, s + window_bars) for s in range(0, total_length - window_bars + 1, window_bars)]
        else:
            raise ValueError(f"rsfgp.window_mode 不支援: {window_mode}")

        if num_samples is not None and num_samples < len(spans):
            # RSFGP 隨機抽樣 (不放回)
            selected_spans = random.sample(spans, num_samples)
            logging.debug(f"🎲 隨機抽樣模式：抽出 {num_samples} 個窗口進行評估。")
        else:
            selected_spans = spans

        windows = []

        for start_idx, end_idx in selected_spans:
            close_chunk = self.full_close[start_idx:end_idx, :]
            eval_price_chunk = self.full_eval_price[start_idx:end_idx, :]
            market_chunk = self.full_market_returns[start_idx:end_idx, :]
            feat_chunk = [f[start_idx:end_idx, :] for f in self.full_features_list]
            active_chunk = self.full_active[start_idx:end_idx, :] if self.full_active is not None else None

            windows.append({
                'close': close_chunk,
                'eval_price': eval_price_chunk,
                'feature_array': feat_chunk,
                'market_returns': market_chunk,
                'active': active_chunk
            })

        self._cached_windows = windows
        self._cached_seed_state = current_state

        return windows
