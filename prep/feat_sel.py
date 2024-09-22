import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import logging

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method='auto', significance_level=0.4, max_percentage=0.2, preserve_vars=None, seed=None):
        self.method = method
        self.significance_level = significance_level
        self.max_percentage = max_percentage
        self.selected_features = None
        self.seed = seed
        
        # 將字串轉換為列表
        if isinstance(preserve_vars, str):
            preserve_vars = [var.strip() for var in preserve_vars.split(',') if var.strip()] 
        self.preserve_vars = preserve_vars if preserve_vars is not None else [] # 必須保留的變數

    def _backward_selection(self, X, y):
        features = list(X.columns)
        y = y.reindex(X.index)  # 確保因變量的索引和自變量一致
        while len(features) > 0:
            X_with_const = sm.add_constant(X[features])
            model = sm.OLS(y, X_with_const).fit()
            p_values = model.pvalues[1:]  # 排除常數項的 p 值
            max_p_value = p_values.max()
            if max_p_value > self.significance_level:
                excluded_feature = p_values.idxmax()
                features.remove(excluded_feature)
            else:
                break
        return features

    def _rf_selection(self, X, y):
        rf = RandomForestRegressor(random_state=self.seed)
        rf.fit(X, y)
        importances = rf.feature_importances_
        max_importance = importances.max()
        threshold = max_importance * self.max_percentage
        # 若 importance 小於最大的0.2倍，則捨棄
        selected_features = X.columns[importances >= threshold].tolist()
        return selected_features

    def _lasso_selection(self, X, y):
        lasso = LassoCV(cv=5, random_state=self.seed)
        lasso.fit(X, y)
        selected_features = X.columns[lasso.coef_ != 0].tolist()
        return selected_features

    def fit(self, X, y=None):
        if self.method == 'auto':
            if X.shape[1] > 50:
                self.method = 'lasso'
            else:
                self.method = 'backward+rf'

        if self.method == 'backward+rf':
            backward_features = self._backward_selection(X, y)
            forest_features = self._rf_selection(X, y)
            self.selected_features = list(set(backward_features).union(set(forest_features)))
            logging.info(f"特徵擷取: 使用 Linear Backward Selection 與 Random Forest 的重要變數聯集選擇特徵")

        elif self.method == 'backward':
            self.selected_features = self._backward_selection(X, y)
            logging.info(f"特徵擷取: 使用 Linear Backward Selection 選擇特徵")
        
        elif self.method == 'rf':
            self.selected_features = self._rf_selection(X, y)
            logging.info(f"特徵擷取: 使用 Random Forest 選擇特徵")

        elif self.method == 'lasso':
            self.selected_features = self._lasso_selection(X, y)
            logging.info(f"特徵擷取: 使用 LASSO 選擇特徵")

        else:
            raise ValueError("Invalid method. Choose from 'auto', 'backward+rf', 'backward', 'rf', or 'lasso'.")

        return self

    def transform(self, X):
        # 確保 preserve_vars 在 selected_features 中
        for var in self.preserve_vars:
            if var not in self.selected_features:
                self.selected_features.append(var)
                logging.info(f"重要變數 '{var}' 不在選擇的特徵中，已加入。")
        logging.info(f"最終選擇特徵: {self.selected_features}")

        # return X[self.selected_features]
        return X[self.selected_features].reindex(X.index)
