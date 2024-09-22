import argparse
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from prep.load import load_data, create_folders
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


def main(args):
    # 建立資料夾
    create_folders(args.base_folder)

    # 設定日誌記錄
    base_folder = args.base_folder if args.base_folder else '.'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f'{base_folder}/analysis.log', filemode='w', encoding='utf-8')

    # 若有指定模型，則載入模型
    if args.load_model is not None:
        if args.test_file is None:
            raise ValueError("必須提供測試檔案")
        model_pipeline = joblib.load(os.path.join(base_folder, 'saved_model', f"{args.load_model}.pkl"))
        model = model_pipeline.named_steps['model_training']
        print(f"Model {args.load_model} loaded successfully.")
        logging.info("成功載入模型")
    
        # 載入測試資料
        test_data = load_data(args.test_file, sheet_name=args.sheet)
        # 若 y 為 NA，則移除那筆資料
        test_data = test_data.dropna(subset=[args.target_column])
        X_test = test_data.drop(columns=[args.target_column])
        y_test = test_data[args.target_column]

        # 將測試資料過預處理
        preprocessor = model_pipeline.named_steps['preprocessor']
        X_test = preprocessor.transform(X_test)
        trained_models = {args.load_model: model}
        logging.info("測試資料預處理完成")
    

    # 否則進行資料前處理與模型訓練
    else:
        try:
            df = load_data(args.input_file, sheet_name=args.sheet)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Failed to load data: {e}")
            return
        
        # 若 y 為 NA，則移除該筆資料
        df = df.dropna(subset=[args.target_column])
        
        # 0. 資料視覺化
        logging.info("開始進行資料視覺化")
        plot_Xy(df, args.target_column, plot_col=args.plot_col, time_column=args.time_column, base_folder=base_folder)

        # 1. 去除重複值
        df = df.drop_duplicates()
        
        # 2. 劃分資料集
        try:    
            X_train, X_test, y_train, y_test = split_dataset(df, args.target_column, args.test_size, random_state=args.seed, test_file=args.test_file, sheet_name=args.sheet, time_column=args.time_column)
        except Exception as e:
            return f"Failed to split dataset: {e}"
        
        # 3. 資料預處理流程
        logging.info("開始進行資料預處理")
        preprocessor = Pipeline(steps=[
            # 處理類別變數
            ('categorical_encoder', CategoricalEncoder(time_column=args.time_column)),
            # 填補遺失值
            ('imputer', CustomImputer(method=args.impute_method, preserve_vars=args.preserve_vars, seed=args.seed)),
            # 正規化
            ('normalizer', CustomNormalizer(method=args.normalize_method)),
            # 特徵選擇
            ('feature_selector', FeatureSelector(method=args.feature_method, significance_level=args.sig, max_percentage=args.rf_thr, preserve_vars=args.preserve_vars, seed=args.seed))
        ])

        # 4. 預處理
        X_train = preprocessor.fit_transform(X_train, y_train)

        # 5. 交叉驗證並訓練模型
        logging.info("開始進行模型訓練與超參數選擇")
        model_manager = ModelManager(args.models, seed=args.seed, time_column=args.time_column) 
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
        plot_true_vs_pred(y_test, y_pred, name, args.target_column, base_folder=base_folder)
    
        # 10. 存下 y_pred
        y_pred_path = os.path.join(base_folder, f'result/prediction/{name}_predictions.csv')
        y_pred_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
        y_pred_df.to_csv(y_pred_path, index=False)
        # logging.info(f"Predictions for {name} saved to {y_pred_path}")

    if len(trained_models) > 1:
        logging.info(f"測試集表現最佳模型為: {best_model}")
        print(f"測試集表現最佳模型為: {best_model}")

    with open(os.path.join(base_folder, 'result/model_evaluation.txt'), 'w') as f:
        for result in results:
            f.write(result)

    # 10. 特徵重要性 (訓練階段才需要看)
    if args.load_model is None:
        feature_importance_models = {
            'LinearRegression': 'coef_',
            'RandomForest': 'feature_importances_',
            'DecisionTree': 'feature_importances_',
            'XGBoost': 'feature_importances_',
            'LightGBM': 'feature_importances_'
        }

        for model_name, attribute in feature_importance_models.items():
            if model_name in trained_models:
                model = trained_models[model_name]
                feature_importances = getattr(model, attribute)
                plot_feature(feature_importances, args.top_n, X_train.columns, f'{model_name}', base_folder)

    print(f"預測結果已儲存至 {base_folder}/result/feature_importance 資料夾。")


# 主程式
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing and model training pipeline.')
    # 隨機種子(選填)
    parser.add_argument('--seed', type=int, default=None, help='set seed for stable results.')
    # 根目錄(選填)
    parser.add_argument('--base_folder', type=str, default='', help='Root folder for storing results.')
    # 工作表
    parser.add_argument('--sheet', type=str, default=None, help='Name of the sheet to load.')
    # 載入模型(選填)
    parser.add_argument('--load_model', type=str, default= None, help='Model name to load, use "None" to train new models.')
    # 輸入檔案(選填)
    parser.add_argument('--input_file', type=str, help='Path to the input CSV file.')
    # 測試檔案(選填)
    parser.add_argument('--test_file', type=str, help='Path to the test CSV file.')
    # 標籤欄位(必填)
    parser.add_argument('--target_column', type=str, required=True, help='Name of the target column.')
    # 必須保留變數(處理遺失值與變數篩選時一定會保留)
    parser.add_argument('--preserve_vars', type=str, default=None, help='一定需要留下的重要變數 以,分隔')
    # 指定欄位畫圖(選填)
    parser.add_argument('--plot_col', type=str, default='all', help='每個欄位都畫:all, 不畫:no_draw, 指定欄位畫: 以,分隔欄位名稱')
    # 時間欄位(選填)
    parser.add_argument('--time_column', type=str, default=None, help='時間欄位名(資料必須按照時間順序排)')
    # 資料劃分方法與劃分比例(選填)
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    # 填補遺失值方法(選填)
    parser.add_argument('--impute_method', type=str, choices=['auto', 'mice', 'knn', 'mean'], default='auto', help='Method for imputing missing values.')
    # 正規化方法(選填)
    parser.add_argument('--normalize_method', type=str, choices=['auto', 'minmax', 'standard'], default='auto', help='Method for normalizing the data.')
    # 特徵選擇方法(選填)
    parser.add_argument('--feature_method', type=str, choices=['auto', 'backward+rf', 'backward', 'rf', 'lasso'], default='auto', help='Method for feature selection.')
    parser.add_argument('--sig', type=float, default=0.4, help='Backward selection significance level.')
    parser.add_argument('--rf_thr', type=float, default=0.2, help='Random Forest selection max percentage.')
    # 模型訓練(選填)
    parser.add_argument('--models', type=str, default='all', help='可選: all, LinearRegression, KNN, SVM, DecisionTree, RandomForest, XGBoost, LightGBM, MLP， 以,分隔欄位名稱')
    # 特徵重要度圖的變數數量(選填)
    parser.add_argument('--top_n', type=int, default=10, help='Number of features to show in feature importance plot.')
    
    args = parser.parse_args()
    main(args)
 