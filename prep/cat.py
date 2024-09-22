from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, time_column=None):
        self.ordinal_cols = []
        self.nominal_cols = []
        self.bool_cols = []
        self.one_hot_encoders = {}
        self.time_column = time_column

    def _categorize_columns(self, df):
        """
        辨別類別型變數和連續型變數。
        對於連續型變數中種類不超過7的變數，當作ordinal處理。
        對nominal資料進行one-hot編碼。
        """  
        if self.time_column is not None:
            df = df.drop(columns=self.time_column, errors='ignore')
        
        cont_cols = []
        ordinal_cols = []
        nominal_cols = []
        bool_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # nominal
                nominal_cols.append(col)
            elif df[col].nunique() <= 7 and df[col].dtype != 'bool':
                # ordinal
                ordinal_cols.append(col)
            elif df[col].dtype == 'bool':
                # boolean
                bool_cols.append(col)
            else:
                # continuous
                cont_cols.append(col)
        
        return ordinal_cols, nominal_cols, bool_cols

    def fit(self, X, y=None):
        self.ordinal_cols, self.nominal_cols, self.bool_cols = self._categorize_columns(X)
        
        for col in self.nominal_cols:
            ohe = OneHotEncoder(handle_unknown='ignore')
            self.one_hot_encoders[col] = ohe.fit(X[[col]])
        
        return self

    def transform(self, X):
        X = X.copy()
        # 對ordinal資料進行處理 (變成類別無法進行encoder, 暫當成連續型
        # for col in self.ordinal_cols:
        #     X[col] = pd.Categorical(X[col], ordered=True)
        
        # 刪除時間欄位
        if self.time_column is not None:
            X = X.drop(columns=self.time_column, errors='ignore')

        # 對bool特徵進行處理
        for col in self.bool_cols:
            X[col] = X[col].astype(int)
        
        for col in self.nominal_cols:
            ohe = self.one_hot_encoders[col]
            ohe_result = ohe.transform(X[[col]])
            ohe_result =  ohe_result.toarray()
            ohe_df = pd.DataFrame(ohe_result, columns=ohe.get_feature_names_out([col]))
            X = X.drop(columns=[col])
            X = pd.concat([X.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        
        if len(self.nominal_cols) == 0:
            logging.info(f"此資料無類別型變數，不需進行類別變數處理")
        else:
            logging.info(f"處理類別型變數: 將名目變數 {self.nominal_cols} 做 One-Hot Encoding")
        return X
