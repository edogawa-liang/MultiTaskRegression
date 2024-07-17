from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
import logging

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='auto', preserve_vars=None, seed=None):
        self.method = method
        self.imputer = None
        self.mean_values = None
        self.chosen_method = None
        self.seed = seed
        # 將字串轉換為列表
        if isinstance(preserve_vars, str):
            preserve_vars = [var.strip() for var in preserve_vars.split(',')]
        self.preserve_vars = preserve_vars if preserve_vars is not None else [] # 必須保留的變數

    def _handle_missing(self, df):
        """
        根據資料特性自動選擇適當的補值方法來處理遺失值。
        
        Returns: DataFrame
        """
        # 判斷每個變數和樣本的遺失比例
        missing_var_ratio = df.isnull().mean()
        missing_sample_ratio = df.isnull().mean(axis=1)
        
        # 如果變數超過60%的樣本遺失，且不在保留列表中，刪除變數
        vars_to_drop = [var for var in missing_var_ratio[missing_var_ratio > 0.6].index if var not in self.preserve_vars]
        if vars_to_drop:
            df = df.drop(columns=vars_to_drop)
            logging.info(f"變數超過60%的樣本遺失，刪除變數: {vars_to_drop}")
        
        # 如果樣本超過60%的變數遺失，刪除樣本
        if any(missing_sample_ratio > 0.6):
            df = df.drop(index=missing_sample_ratio[missing_sample_ratio > 0.6].index)
            logging.info(f"樣本超過60%的變數遺失，刪除樣本")

        # 使用適當的方法填補缺失值
        if df.shape[0] > 200:
            self.imputer = IterativeImputer(random_state=self.seed)
            df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
            self.chosen_method = 'mice'

        else:
            # 計算變數之間的平均相關性(排除NaN值)
            cor = 0
            n = 0  # 實際配對數
            corr_matrix = df.corr(min_periods=1).abs()
            for i in range(corr_matrix.shape[0]):
                for j in range(i):
                    if not pd.isna(corr_matrix.iloc[i, j]):
                        cor += corr_matrix.iloc[i, j]
                        n += 1
            # 計算平均相關性
            mean_cor = cor / n if n != 0 else 0
            
            # 若變數間彼此有點相關性，使用KNN補值 (平均相關性>0.15)
            if mean_cor > 0.15:
                self.imputer = KNNImputer(n_neighbors=5)
                df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
                self.chosen_method = 'knn'

            else:
                self.mean_values = df.mean(skipna=True)
                df_imputed = df.fillna(self.mean_values)
                self.imputer = None
                self.chosen_method = 'mean'

        return df_imputed

    def fit(self, X, y=None):
        if self.method == 'auto':
            self._handle_missing(X)
        else:
            if self.method == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
                self.chosen_method = 'knn'
            elif self.method == 'mice':
                self.imputer = IterativeImputer(random_state=self.seed)
                self.chosen_method = 'mice'
            elif self.method == 'mean':
                self.mean_values = X.mean()
                self.imputer = None
                self.chosen_method = 'mean'
            else:
                raise ValueError("無效的補值方法。可選方法為 'knn', 'mice', 'mean', 'auto'。")

            if self.imputer:
                self.imputer.fit(X)

        return self

    
    def transform(self, X):
        logging.info(f"使用 {self.chosen_method} 補遺失值")
        if self.method == 'mean' and self.imputer is None:
            return X.fillna(self.mean_values)
        elif self.imputer:
            return pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        else:
            return self._handle_missing(X)