from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import logging

class CustomNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, method='auto'):
        self.method = method
        self.scaler = None

    def fit(self, X, y=None):
        # 檢查是否存在nominal變數（one-hot編碼）
        contains_categorical = any(X.dtypes == 'uint8')

        # 選擇正規化方法
        if self.method == 'auto':
            if contains_categorical:
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError("無效的正規化方法。可選方法為 'auto', 'minmax', 'standard'。")

        # 擬合scaler
        self.scaler.fit(X)
        return self

    def transform(self, X):
        logging.info(f"使用 {self.scaler} 做正規化")
        # 應用擬合好的scaler進行轉換
        X_transformed = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        return X_transformed
