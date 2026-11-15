# stgp-sp500-trading-rules

以強型別遺傳規劃 (Strongly Typed Genetic Programming, STGP；DEAP 實作) 演化 S&P 500 大型股的交易規則，
並於 2019–2025 年逐年滾動的樣本外測試年，與等權重買進持有 (Buy-and-Hold, B&H) 比較。

- **交易規則**：布林語法樹，對每檔股票、每根 65 分鐘 K 棒輸出 True (下一根持有) 或 False (空手)；只做多，持有的股票等權重。
- **適應度**：將 4 年訓練期依日曆切為 8 個半年窗口，
  每個窗口的分數為 報酬 / (MDD + 0.1) 減去同窗口 B&H 的分數 (限制於 ±10)，
  再組合為 `mean(s_w) − 0.5 · downside_dev − 0.002 · 節點數`。
- **模型選擇**：(μ+λ) = 2,000 + 2,000；每代將訓練期前 20 名送至 1 年驗證期評估，
  驗證分數最高者為冠軍；連續 100 代未改善即停止演化。
- **成分股進出 (point-in-time)**：股票僅在 S&P 500 成分股期間可被持有，排名特徵亦只在當時的成分股之間計算。
- **交易成本**：每邊 0.1%，以 K 棒 (最高價 + 最低價) / 2 成交；僅在持股組合改變時調整權重。

## 專案結構

```text
.
├── main.py                  # 單次演化：讀取 config.yaml，建立 DataProvider / Evaluator / GPManager 並執行
├── multi_run_manager.py     # 互動式選單：以不同 seed 重複執行 N 次，彙整 Train / Test 績效並產生 HTML 報告
├── rolling_multi_run.py     # 滾動實驗：Train 4 年 / Val 1 年 / Test 1 年，Test 年 2019–2025 各執行一次 multi_run_manager
├── config.yaml              # 超參數、資料路徑與特徵清單 (預設值為報告的主設定)
├── environment.yml          # Conda 環境
├── data/                    # 資料 (不隨 repo 提供；格式見 data/README.md)
├── data_pipeline/
│   ├── universe_data_build.py   # 原始 CSV -> 對齊後的價格矩陣與成分股期間矩陣 (universe_raw.pkl)
│   └── features_pit.py          # 計算 point-in-time 特徵 -> features_stage27.pkl / features_stage55.pkl
├── src/
│   ├── primitives.py        # STGP 算子集：依特徵名稱決定型別 (Rank_* / IBS* -> RankType)
│   ├── data_provider.py     # 載入特徵檔，切分 Train / Val / Test 期間與 RSFGP 窗口
│   ├── evaluator.py         # 向量化交易狀態機與適應度
│   ├── evaluator_moo.py     # 多目標適應度 (單目標時沿用 evaluator.py)
│   ├── gp_manager.py        # DEAP (μ+λ) 演化、去重存活選擇、Validation 與 early stopping
│   ├── gp_manager_moo.py    # 多目標版的 Validation 與 early stopping (單目標時等同 gp_manager.py)
│   ├── benchmarks.py        # B&H 基準 (含成分股進出的版本)
│   ├── feature_utils.py     # 特徵名稱與 [0, 1] 尺度檢查、特徵診斷報告
│   ├── moo_selection.py / moo_validation.py   # 多目標實驗的代表策略挑選與驗證
│   └── tree_simplifier.py   # 公式化簡、產生供 LLM 解讀公式的 prompt
├── analysis/
│   └── strategy_analyzer.py # 互動式：讀取單一策略 (.pkl)，輸出公式並繪製資金曲線與語法樹
└── tests/
    └── test_membership.py   # 成分股加入 / 剔除規則的正確性測試
```

## 環境

```bash
conda env create -f environment.yml
conda activate trading_gp
```

Windows 使用者需另行安裝 [Graphviz](https://graphviz.org/download/) 並加入 PATH (`strategy_analyzer.py` 繪製語法樹時使用)。

## 資料

本 repo **不提供任何資料**。研究使用 TradingView (BATS) 的 65 分鐘與日線 OHLCV (2013–2025，經分割與股利調整)、
FRED 三個月期國庫券利率 (DTB3)，以及自行整理的 S&P 500 成分股期間名單。
所需檔案、資料夾結構與欄位格式見 [`data/README.md`](data/README.md)。備齊後依序執行：

```bash
python data_pipeline/universe_data_build.py                # -> data/processed/universe_raw.pkl
python data_pipeline/features_pit.py --stage 27 --market   # -> data/processed/features_stage27.pkl
python data_pipeline/features_pit.py --stage 55 --market   # -> data/processed/features_stage55.pkl
```

實驗使用的特徵檔由 `config.yaml` 的 `data.file_path` 指定。

### 特徵 (15 個，數值皆落於 [0, 1])

| 類別 | 特徵 | 定義 |
| :--- | :--- | :--- |
| 市場模型 | `Rank_alpha20`, `Rank_alpha60`, `Rank_beta60` | 日報酬對其他成分股等權報酬的滾動迴歸 (20 / 60 日)，延遲一日 |
| 短期反轉 | `Rank_ROC1` | 前一根 K 棒的報酬 |
| | `IBS3` | 收盤價於最近 3 根 K 棒高低區間中的相對位置 (本身即在 [0, 1]，不排名) |
| | `Rank_Bias_EMA6` | 收盤價相對 6 根 EMA 的乖離 |
| 動能 | `Rank_ROC30`, `Rank_ROC240`, `Rank_ROC720` | 最近 30 / 240 / 720 根 K 棒的報酬 (約 1 週 / 2 個月 / 6 個月) |
| 波動度 | `Rank_HV30`, `Rank_HV240` | 最近 30 / 240 根 K 棒對數報酬的標準差 |
| 成交量 | `Rank_RelVol20` | 成交量 / 最近 20 根平均成交量 |
| 大盤狀態 | `Rank_Mkt_Trend`, `Rank_Mkt_Vol` | 成分股等權指數的 240 根報酬 / 30 根波動度，於過去 1,512 根中的百分位 |
| | `Rank_Mkt_Breadth` | 收盤價高於自身 240 根 EMA 的成分股比例 |

除 `IBS3` 與大盤狀態特徵外，`Rank_*` 皆為每根 K 棒在當時成分股之間的百分位排名。
1 個交易日為 6 根 65 分鐘 K 棒，1 年約 1,512 根。

### 股票池

兩個 S&P 500 大型股子集合：27 檔 (涵蓋 11 個 GICS 產業) 與包含前者的 55 檔。
股票僅在成分股期間內可被持有；「全期」表示 2013–2025 年間皆為 S&P 500 成分股。

**27 檔股票池**

| # | 代號 | 公司 | GICS 產業 | 成分股期間 |
| ---: | :--- | :--- | :--- | :--- |
| 1 | AAPL | Apple | Information Technology | 全期 |
| 2 | INTC | Intel | Information Technology | 全期 |
| 3 | GOOGL | Alphabet | Communication Services | 全期 |
| 4 | NFLX | Netflix | Communication Services | 全期 |
| 5 | AMZN | Amazon.com | Consumer Discretionary | 全期 |
| 6 | TSLA | Tesla | Consumer Discretionary | 2020-12-21 起 |
| 7 | NKE | Nike | Consumer Discretionary | 全期 |
| 8 | HD | Home Depot | Consumer Discretionary | 全期 |
| 9 | KO | Coca-Cola | Consumer Staples | 全期 |
| 10 | PG | Procter & Gamble | Consumer Staples | 全期 |
| 11 | PEP | PepsiCo | Consumer Staples | 全期 |
| 12 | JNJ | Johnson & Johnson | Health Care | 全期 |
| 13 | UNH | UnitedHealth Group | Health Care | 全期 |
| 14 | PFE | Pfizer | Health Care | 全期 |
| 15 | ABT | Abbott Laboratories | Health Care | 全期 |
| 16 | JPM | JPMorgan Chase | Financials | 全期 |
| 17 | GS | Goldman Sachs | Financials | 全期 |
| 18 | V | Visa | Financials | 全期 |
| 19 | BAC | Bank of America | Financials | 全期 |
| 20 | CVX | Chevron | Energy | 全期 |
| 21 | XOM | Exxon Mobil | Energy | 全期 |
| 22 | BA | Boeing | Industrials | 全期 |
| 23 | CAT | Caterpillar | Industrials | 全期 |
| 24 | UPS | United Parcel Service | Industrials | 全期 |
| 25 | DUK | Duke Energy | Utilities | 全期 |
| 26 | PLD | Prologis | Real Estate | 全期 |
| 27 | LIN | Linde | Materials | 全期 |

**55 檔股票池 = 上述 27 檔 + 以下 28 檔**

| # | 代號 | 公司 | GICS 產業 | 成分股期間 |
| ---: | :--- | :--- | :--- | :--- |
| 1 | MSFT | Microsoft | Information Technology | 全期 |
| 2 | NVDA | NVIDIA | Information Technology | 全期 |
| 3 | CSCO | Cisco Systems | Information Technology | 全期 |
| 4 | AVGO | Broadcom | Information Technology | 2014-05-08 起 |
| 5 | ENPH | Enphase Energy | Information Technology | 2021-01-07 ~ 2025-09-19 |
| 6 | META | Meta Platforms | Communication Services | 全期 |
| 7 | LUMN | Lumen Technologies | Communication Services | 至 2023-03-17 |
| 8 | MCD | McDonald's | Consumer Discretionary | 全期 |
| 9 | WHR | Whirlpool | Consumer Discretionary | 至 2024-03-15 |
| 10 | WMT | Walmart | Consumer Staples | 全期 |
| 11 | LLY | Eli Lilly | Health Care | 全期 |
| 12 | DXCM | DexCom | Health Care | 2020-05-12 起 |
| 13 | ILMN | Illumina | Health Care | 至 2024-06-21 |
| 14 | WFC | Wells Fargo | Financials | 全期 |
| 15 | MS | Morgan Stanley | Financials | 全期 |
| 16 | ZION | Zions Bancorporation | Financials | 至 2024-03-15 |
| 17 | COP | ConocoPhillips | Energy | 全期 |
| 18 | NOV | NOV Inc. | Energy | 至 2021-09-17 |
| 19 | HON | Honeywell | Industrials | 全期 |
| 20 | UNP | Union Pacific | Industrials | 全期 |
| 21 | LMT | Lockheed Martin | Industrials | 全期 |
| 22 | ODFL | Old Dominion Freight Line | Industrials | 2019-12-09 起 |
| 23 | NEE | NextEra Energy | Utilities | 全期 |
| 24 | PCG | PG&E | Utilities | 至 2019-01-17；2022-10-03 起再次納入 |
| 25 | AMT | American Tower | Real Estate | 全期 |
| 26 | EQIX | Equinix | Real Estate | 2015-03-23 起 |
| 27 | APD | Air Products and Chemicals | Materials | 全期 |
| 28 | FMC | FMC Corporation | Materials | 至 2025-03-21 |

## 執行

```bash
python main.py                                   # 單次演化 (依 config.yaml)
python multi_run_manager.py                      # 以相同設定重複執行 N 次並產生報告 (互動式選單)
python rolling_multi_run.py --dry-run            # 列出 7 組滾動窗口的日期
python rolling_multi_run.py --runs 1 --only 2019 # 快速檢查：僅執行 Test 2019 一次
python rolling_multi_run.py                      # 完整滾動實驗：7 個 Test 年 × 30 次
python analysis/strategy_analyzer.py             # 檢視單一策略 (互動式選單)
```

- 每個 run 的族群規模為 2,000 + 2,000，完整實驗耗時較長。平行程序數由 `system.n_workers` 設定 (預設 4)。
- 單次執行 `main.py` 的輸出存於 `checkpoints/`；`multi_run_manager.py` 的結果存於 `gp_experiment_studies/study_<時間>/`，
  滾動實驗整理於 `gp_experiment_studies/rolling/`，並在 `gp_experiment_studies/EXPERIMENTS.md` 新增一列紀錄。
- `multi_run_manager.py` 與 `rolling_multi_run.py` 不會修改根目錄的 `config.yaml`，每個 run 使用各自的設定副本。

## 測試

```bash
python tests/test_membership.py   # 以逐根 K 棒的參考實作，驗證向量化狀態機的成分股進出規則
```

## 備註

- 研究報告中的統計檢定 (Wilcoxon signed-rank、Mann-Whitney U、Holm 校正、bootstrap 信賴區間) 程式未包含於本 repo。
- 本專案僅供學術研究，不構成任何投資建議。
