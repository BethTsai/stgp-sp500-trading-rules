# data/

本 repo 不提供任何資料。請依下列結構自行準備原始資料，再依序執行：

```bash
python data_pipeline/universe_data_build.py                # -> data/processed/universe_raw.pkl
python data_pipeline/features_pit.py --stage 27 --market   # -> data/processed/features_stage27.pkl
python data_pipeline/features_pit.py --stage 55 --market   # -> data/processed/features_stage55.pkl
```

## 資料夾結構

```text
data/
├── raw/                          # 原始資料 (自行準備)
│   ├── prices_65min/             # 每檔股票一個 65 分鐘 K 棒 CSV
│   │   ├── AAPL.csv
│   │   └── ...
│   ├── prices_daily/             # 每檔股票一個日 K CSV (用於 CAPM 特徵)
│   │   ├── AAPL.csv
│   │   └── ...
│   ├── spy_daily.csv             # SPY 日 K，僅用於檢查 CAPM 計算
│   ├── fred_dtb3.csv             # 無風險利率
│   └── sp500_membership.csv      # S&P 500 成分股期間
└── processed/                    # 程式輸出 (自動建立)
    ├── universe_raw.pkl
    ├── build_report.txt
    ├── features_stage27.pkl
    └── features_stage55.pkl
```

股票代號取自檔名。可接受 `AAPL.csv`、`AAPL_65min.csv` 或 TradingView 的預設檔名 `BATS_AAPL, 65.csv`；
同一資料夾內不可有兩個檔案對應同一個代號。路徑設定位於 `data_pipeline/universe_data_build.py` 開頭。

## 檔案格式

### `prices_65min/`、`prices_daily/`、`spy_daily.csv` (TradingView 匯出格式)

| 欄位 | 說明 |
| :--- | :--- |
| `time` | 65 分鐘：含時區的時間，例如 `2013-01-02T09:30:00-05:00` (時區不限，程式轉換為美東時間)；日 K：日期，例如 `2013-01-02` |
| `open`, `high`, `low`, `close` | 經分割與股利調整的價格 |
| `Volume` | 成交量 |

其他欄位不會被讀取。使用期間為 2013-01-02 至 2025-12-31。
正常交易日有 6 根 65 分鐘 K 棒，起始時間為美東 09:30、10:35、11:40、12:45、13:50、14:55。

### `fred_dtb3.csv` (FRED 三個月期國庫券利率，序列代碼 DTB3)

| 欄位 | 說明 |
| :--- | :--- |
| `observation_date` | 日期 |
| `DTB3` | 年化利率 (%)，例如 `5.25` |

### `sp500_membership.csv` (成分股期間)

| 欄位 | 說明 |
| :--- | :--- |
| `symbol` | 股票代號，需與價格檔的代號一致 |
| `stage` | 所屬股票池：`27` 屬於 27 檔池 (同時包含於 55 檔池)；`55` 僅屬於 55 檔池 |
| `start_date` | 納入 S&P 500 的日期 |
| `end_date` | 剔除日期；仍為成分股則留空 |

曾多次進出指數的股票，每段期間各占一列。格式範例：

```csv
symbol,stage,start_date,end_date
AAPL,27,1982-11-30,
TSLA,27,2020-12-21,
ENPH,55,2021-01-07,2025-09-19
PCG,55,1957-03-04,2019-01-17
PCG,55,2022-10-03,
```
