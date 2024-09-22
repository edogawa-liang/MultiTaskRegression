from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import urllib.parse
from typing import Optional, Dict, List
import os
import joblib
import pandas as pd
import random
from sklearn.pipeline import Pipeline
from prep.load import load_data, create_folders
from prep.load import download_from_ftp, zip_files, get_files_in_folder
from prep.cat import CategoricalEncoder
from prep.split import split_dataset
from prep.miss import CustomImputer
from prep.norm import CustomNormalizer
from prep.feat_sel import FeatureSelector
from model.train import ModelManager
from model.eval import performance
from viz.viz_pre import plot_Xy
from viz.viz_model import plot_feature, plot_true_vs_pred
import logging
import socket
import traceback


hostname = socket.gethostname()
server_ip = socket.gethostbyname(hostname)
print(server_ip)
port = 8008

app = FastAPI(
    title="Regression Analysis API",
    description="""
    This API can automatically complete a series of data analysis workflow. 
    Users can upload FTP links pointing to Excel or CSV files. It can automatically perform data visualization, preprocessing, build regression models, evaluate and select the best model to provide to the user. """,
    version="0.1",
    servers=[
        {"url": f"http://{server_ip}:{port}", "description": "dev server"}
    ]
)

@app.post("/Regression/",
          tags=["Regression Analysis API"],
          summary="Run Regression Analysis",
          description="This endpoint runs regression analysis on the provided excel/csv data from an FTP link.")
async def regression(
    url_input: Optional[str],
    target_column: Optional[str],
    time_column: Optional[str] = None,
    preserve_vars: Optional[str] = None,
    plot_col: Optional[str] = 'all',
    sheet: Optional[str] = None,
    seed: Optional[int] = 0,
    test_size: Optional[float] = 0.2,
    test_file: Optional[str] = None,
    impute_method: Optional[str] = 'auto',
    normalize_method: Optional[str] = 'auto',
    feature_method: Optional[str] = 'auto',
    sig: Optional[float] = 0.4,#
    rf_thr: Optional[float] = 0.2,#
    models: Optional[str] = 'auto',
    top_n: Optional[int] = 10, #
    base_folder: Optional[str] = 'test2',#
    ):
    try:
        # 若沒有設定隨機種子 則結果隨機
        if seed == 0:
            print('無指定隨機種子')
            seed = random.randint(0, 9999)
        else:
            print('指定隨機種子')


        # 建立資料夾
        base_folder = base_folder if base_folder else 'test2'
        create_folders(base_folder)

        # 將從 FTP 鏈接獲取的 URL 進行解碼，以便獲取原始格式的 URL
        ftp_url = urllib.parse.unquote(url_input)
        print(ftp_url)
        # 下載了 FTP 鏈接指向的 Excel 文件到伺服器的本地資料夾中。 (後續再讀取了下載在server的文件)
        local_path = os.path.join(base_folder, "data", os.path.basename(ftp_url))
        download_from_ftp(ftp_url, local_path)
        
        # 額外的 test 資料
        if test_file is not None and test_file != '' :
            ftp_url_test = urllib.parse.unquote(test_file)
            local_path_test = os.path.join(base_folder, "data", os.path.basename(ftp_url_test))
            download_from_ftp(ftp_url_test, local_path_test)
        else:
            local_path_test = None

        # 設定日誌記錄
        log_file = os.path.join(base_folder, 'analysis.log')
        print(log_file)

        # 關掉之前沒關掉的log
        logger = logging.getLogger()
        for handler in logger.handlers:
            handler.close()         # 關閉所有處理器
            logger.removeHandler(handler)
        # if os.path.exists(log_file):
        #     os.remove(log_file)

        # 開始新的log
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w', encoding='utf-8')

        
        # 進行資料前處理與模型訓練
        try:
            df = load_data(local_path, sheet_name=sheet)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Failed to load data: {e}")
            return JSONResponse(content={"error": f"Failed to load data: {e}"}, status_code=400)
        
        # 若 y 為 NA，則移除該筆資料
        df = df.dropna(subset=[target_column])
        
        # 若time_column為空，則為None
        if time_column is None or time_column == '':
            time_column = None

        # 0. 資料視覺化
        logging.info("開始進行資料視覺化")
        plot_Xy(df, target_column, plot_col=plot_col, time_column=time_column, base_folder=base_folder)

        # 1. 去除重複值
        df = df.drop_duplicates()
        
        # 2. 劃分資料集
        try:    
            X_train, X_test, y_train, y_test = split_dataset(df, target_column, test_size, random_state=seed, test_file=local_path_test, sheet_name=sheet, time_column=time_column)
        except Exception as e:
            return f"Failed to split dataset: {e}"
        
        
        # 3. 資料預處理流程
        logging.info("開始進行資料預處理")
        preprocessor = Pipeline(steps=[
            # 處理類別變數
            ('categorical_encoder', CategoricalEncoder(time_column=time_column)),
            # 填補遺失值
            ('imputer', CustomImputer(method=impute_method, preserve_vars=preserve_vars, seed=seed)),
            # 正規化
            ('normalizer', CustomNormalizer(method=normalize_method)),
            # 特徵選擇
            ('feature_selector', FeatureSelector(method=feature_method, significance_level=sig, max_percentage=rf_thr, preserve_vars=preserve_vars, seed=seed))
        ])

        # 4. 預處理
        X_train = preprocessor.fit_transform(X_train, y_train)
        
        # 5. 交叉驗證並訓練模型
        logging.info("開始進行模型訓練與超參數選擇")
        model_manager = ModelManager(models, seed=seed, time_column=time_column) 
        best_models = model_manager.cv_fit(X_train, y_train)

        # 6. 保存每個模型與其對應的預處理步驟
        for name, model in best_models.items():
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model_training', model)
            ])
            joblib.dump(model_pipeline, os.path.join(base_folder, 'saved_model', f"{name}.pkl"))

        # 7. 將測試資料過預處理
        logging.info("使用測試資料評估模型")
        # 保留一份X_test (後續找time_col用)
        X_test_saved = X_test.copy()
        X_test = preprocessor.transform(X_test)
        trained_models = best_models

        # 8. 評估模型，並保存結果到txt文件
        results = []
        best_model = None
        best_model_score = float('inf')
        
        for name, model in trained_models.items():
            print(f"Evaluating {name}")
            y_pred, metrics = performance(model, X_test, y_test)
            results.append(f"Model: {name}\n{metrics}\n")
            logging.info(f"Model: {name}, Metrics: {metrics}")
            print(metrics)

            if metrics['MAPE'] < best_model_score:
                best_model_score = metrics['MAPE']
                best_model = name
            
            # 9. 視覺化
            plot_true_vs_pred(y_test, y_pred, name, target_column, base_folder=base_folder)
        
            # 10. 存下 y_pred
            y_pred_path = os.path.join(base_folder, f'result/prediction/{name}_predictions.csv')
            # 合併 X_test 和 y_pred
            y_pred_df = pd.concat([X_test_saved.reset_index(drop=True), 
                                y_test.reset_index(drop=True).rename(target_column), 
                                pd.Series(y_pred, name='Predicted').reset_index(drop=True)], axis=1)

            y_pred_df.to_csv(y_pred_path, index=False, encoding='utf-8-sig')
            # logging.info(f"Predictions for {name} saved to {y_pred_path}")

        if len(trained_models) > 1:
            logging.info(f"測試集表現最佳模型為: {best_model}")
            print(f"測試集表現最佳模型為: {best_model}")

        with open(os.path.join(base_folder, 'result/model_evaluation.txt'), 'w') as f:
            for result in results:
                f.write(result)

        # 10. 特徵重要性
        feature_importance_models = {
            'LinearRegression': 'coef_',
            'RandomForest': 'feature_importances_',
            'DecisionTree': 'feature_importances_',
            'XGBoost': 'feature_importances_',
            'LightGBM': 'feature_importances_'
        }
        
        for model_name in trained_models.keys():
            if model_name in feature_importance_models:
                attribute = feature_importance_models[model_name]
                model = trained_models[model_name]
                feature_importances = getattr(model, attribute)
                plot_feature(feature_importances, top_n, X_train.columns, f'{model_name}', base_folder)

                # 若為 Linear Regression 印出迴歸係數
                if model_name == 'LinearRegression':
                    intercept = model.intercept_
                    coefficients = pd.Series(feature_importances, index=X_train.columns)
                    # 建立數學式
                    equation = f"{target_column} = {intercept:.2f}"
                    for feature, coef in coefficients.items():
                        equation += f" + ({coef:.2f}) * {feature}"
                
                    print(f"線性迴歸方程式: {equation}")
                    logging.info(f"Linear Regression equation: {equation}")
                    
        # print(f"預測結果已儲存至 {base_folder}/result 資料夾。")
                
        links = {}
        file_list = get_files_in_folder(os.path.join(base_folder, "fig"))
        zip_files(file_list, os.path.join(base_folder, "fig.zip"))    
        m = os.path.join(base_folder, "saved_model", f"{best_model}.pkl")
        if os.path.exists(m):
            links["model_link"] = f"http://{server_ip}:{port}/models/?model_path={m}"

        v = os.path.join(base_folder, "fig.zip")
        if os.path.exists(v):
            links["viz_link"] = f"http://{server_ip}:{port}/visualization/?viz_path={v}"

        f = os.path.join(base_folder, "result", "feature_importance", f"{best_model}_featimp.png")
        if os.path.exists(f):
            links["featimp_link"] = f"http://{server_ip}:{port}/feature_importance/?featimp_path={f}"

        d = os.path.join(base_folder, "result", "density_plot", f"{best_model}_densplot.png")
        if os.path.exists(d):
            links["densplot_link"] = f"http://{server_ip}:{port}/density_plot/?densplot_path={d}"

        p = os.path.join(base_folder, "result", "prediction", f"{best_model}_predictions.csv")
        if os.path.exists(p):
            links["pred_link"] = f"http://{server_ip}:{port}/prediction/?pred_path={p}"

        l = os.path.join(base_folder, "analysis.log")
        if os.path.exists(l):
            links["log_link"] = f"http://{server_ip}:{port}/log/?log_path={l}"

        # 提供執行的壓縮檔
        if os.path.exists('pred.zip'):
            links["pred_new_link"] = f"http://{server_ip}:{port}/pred_new/?pred_new_path={"pred.zip"}"

        return JSONResponse(content=links)
    
    except PermissionError as e:
        error_trace = traceback.format_exc()
        return JSONResponse(content={"error": str(e), "details": error_trace}, status_code=500)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        return JSONResponse(content={"error": str(e), "details": error_trace}, status_code=500)
    
    # 生成 Python 代碼

# 下載模型
@app.get("/models/",
         tags=["Regression Analysis API"],
         summary="Download Trained Model",
         description="This endpoint allows you to download the trained regression model.",
         response_description="The response is the trained model file."
         )
async def models(model_path: str):
    return FileResponse(model_path, media_type='application/octet-stream', filename=os.path.basename(model_path))


# 下載視覺化結果
@app.get("/visualization/",
         tags=["Regression Analysis API"],
         summary="Download Data Visualization",
         description="This endpoint allows you to download data visualization, such as scatter plot, barplot and so on.",
         response_description="The response is the data visualization zip file.")
async def visualization(viz_path: str):
    return FileResponse(viz_path, media_type='image/zip', filename=os.path.basename(viz_path))

# 下載 feature importance
@app.get("/feature_importance/",
         tags=["Regression Analysis API"],
         summary="Download Feature Importance Plot",
         description="This endpoint allows you to download the feature importance plot, which helps you visualize the most influential features for the model's predictions.",
         response_description="The response is feature importance of the best model.")
async def feature_importance(featimp_path: str):
    return FileResponse(featimp_path, media_type='image/png', filename=os.path.basename(featimp_path))

# 下載 density plot
@app.get("/density_plot/",
         tags=["Regression Analysis API"],
         summary="Download Density Plot",
         description="This endpoint allows you to download the predict and actual value density plot.",
         response_description="The response includes the density plot of actual and predicted values.")
async def density_plot(densplot_path: str):
    return FileResponse(densplot_path, media_type='image/png', filename=os.path.basename(densplot_path))

# 下載 Prediction
@app.get("/prediction/",
         tags=["Regression Analysis API"],
         summary="Download Predicted value",
         description="This endpoint allows you to download the prediction",
         response_description="The response is the prediction of test ")
async def prediction(pred_path: str):
    return FileResponse(pred_path, media_type='text/csv', filename=os.path.basename(pred_path))

# 下載 log
@app.get("/log/",
         tags=["Regression Analysis API"],
         summary="Download log",
         description="This log endpoint to generate a report that details the actions taken during the data analysis workflow. The log contains a record of the operations performed, including data processing steps, training results and other relevant information. You can extract this information from the log to generate report outlining the analysis process and its outcomes.",
         response_description="The response is the record what analysis do.")
async def log(log_path: str):
    return FileResponse(log_path, media_type='text/plain', filename=os.path.basename((log_path)))

# 下載 如何執行
@app.get("/pred_new/",
         tags=["Regression Analysis API"],
         summary="Download Python code for predicting new data",
         description="This endpoint provides a ZIP file containing the Python code that can be used to predict new data using a trained model.",
         response_description="The response is a ZIP file containing the Python code for prediction.")
async def pred_new(pred_new_path: str):
    return FileResponse(pred_new_path, media_type='application/zip', filename=os.path.basename(pred_new_path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=server_ip, port=port)
