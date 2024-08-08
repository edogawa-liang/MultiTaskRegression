import pandas as pd
import joblib
import os

# 載入資料做預處理，並使用已訓練的模型進行預測。
def load_and_predict(base_folder, target_column, load_model, data_file_path, sheet_name=None):

    # 嘗試讀取CSV文件，如果失敗則讀取Excel文件
    try:
        test_new = pd.read_csv(data_file_path)
    except FileNotFoundError:
        if sheet_name:
            test_new = pd.read_excel(data_file_path, sheet_name=sheet_name)
        else:
            raise FileNotFoundError("指定的文件不存在，並且未提供Excel的sheet名稱。")
    
    # 確認目標欄位是否存在於數據中，若不存在，預處理只使用特徵
    X_test = test_new.drop(columns=[target_column], errors='ignore')

    # 從指定路徑加載模型
    model_pipeline = joblib.load(os.path.join(base_folder, 'saved_model', f"{load_model}.pkl"))

    # 加載模型組件
    preprocessor = model_pipeline.named_steps['preprocessor']
    model = model_pipeline.named_steps['model']

    # 對測試數據進行預處理
    X_test_preprocessed = preprocessor.transform(X_test)

    # 使用模型進行預測
    y_pred = model.predict(X_test_preprocessed)

    # 將預測結果合併回原始數據
    colname = f"{target_column}_pred"
    X_test[colname] = y_pred

    # 將含預測結果的數據框存儲為CSV文件
    output_path = os.path.join(base_folder, "result/new_pred.csv")
    X_test.to_csv(output_path, index=False)

    print(f"Model {load_model} loaded and predictions saved successfully at {output_path}.")
