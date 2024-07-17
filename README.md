# 適用於多任務的迴歸模型
設計通用任務的迴歸模型，以自動化方式完成一系列資料分析流程。

## Pipeline
```
Regression Analysis
    ├── viz
    │    ├── viz_pre.py (資料視覺化)
    │    └── viz_model.py (特徵重要性, 模型預測密度圖)
    ├── prep
    │   ├── load.py (資料讀取)
    │   ├── split.py (劃分資料集)
    │   ├── cat.py (處理類別變數)
    │   ├── miss.py (處理遺失值)
    │   ├── norm.py (正規化)
    │   └── feat_sel.py (特徵擷取)
    └── model
        ├── train.py (模型訓練)
        ├── eval.py (模型評估)
        └── time.py (時間序列模型sklearn接口)

```
## How to inference on your data
```python
python main.py --target_column target --input_file data.csv 
```

- `--base_folder` (str, 選填): 用於存儲結果的根目錄。預設值為空字串。
- `--sheet` (str, 選填): 要載入的工作表名稱。預設值為 `None`。
- `--load_model` (str, 選填): 要載入的模型名稱，使用 `None` 來訓練新模型。預設值為 `None`。
- `--input_file` (str, 選填): 輸入檔案(csv, xlsx)
- `--test_file` (str, 選填): 測試(csv, xlsx)檔案的路徑。
- `--target_column` (str, 必填): 目標欄位名稱。
- `--preserve_vars` (str, 選填): 一定需要留下的重要變數。預設值為 `None`。
- `--plot_col` (str, 選填): 指定欄位畫圖。選項包括 `all` (每個欄位都畫), `no_draw` (不畫), 以及以`,`分隔的欄位名稱。預設值為 `all`。
- `--time_column` (str, 選填): 時間欄位。預設值為 `None`。
- `--test_size` (float, 選填): 資料集劃分中包含在測試集中的比例。預設值為 `0.2`。
- `--impute_method` (str, 選填): 填補遺失值的方法。選項包括 `auto`, `mice`, `knn`, `mean`。預設值為 `auto`。
- `--normalize_method` (str, 選填): 正規化方法。選項包括 `auto`, `minmax`, `standard`。預設值為 `auto`。
- `--feature_method` (str, 選填): 特徵選擇方法。選項包括 `auto`, `model`, `backward`, `rf`, `lasso`。預設值為 `auto`。
- `--sig` (float, 選填): 向後選擇的顯著性水平。預設值為 `0.4`。
- `--rf_thr` (float, 選填): 隨機森林選擇的最大比例。預設值為 `0.2`。
- `--models` (str, 選填): 要訓練的模型。選項包括 `all`, `LinearRegression`, `KNN`, `SVM`, `DecisionTree`, `RandomForest`, `XGBoost`, `LightGBM`, `MLP`，以`,`分隔的欄位名稱。預設值為 `all`。
- `--top_n` (int, 選填): 特徵重要度圖中顯示的特徵數量。預設值為 `10`。
- `--seed` (int, 選填): 設置隨機種子以獲得一致的結果。預設值為 `None`。


```
python main.py --input_file 'data.csv' --target_column '學期成績(原始)'  --plot_col all --time_column Name --preserve_vars HW1 --base_folder example --seed 123
```
```
Example
├── analysis.log (資料分析日誌)
├── fig (資料視覺化)
├── result
│   ├── density_plot (模型預測密度圖)
│   ├── feature_importance (特徵重要性)
│   └── model_evaluation.txt (模型表現一覽)
└── saved_model (訓練完成的模型)
```

