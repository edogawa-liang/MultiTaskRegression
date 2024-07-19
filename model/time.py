from sklearn.base import BaseEstimator
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging

class ARIMAWrapper(BaseEstimator):
    def __init__(self, order=None):
        self.order = order
        self.model_ = None
    
    def fit(self, X, y):
        logging.getLogger('statsmodels').setLevel(logging.WARNING)
        self.model_ = ARIMA(y, order=self.order).fit()
        return self
    
    def predict(self, X):
        return self.model_.predict(start=0, end=len(X)-1).tolist()


class SARIMAXWrapper(BaseEstimator):
    def __init__(self, order=None, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_ = None
    
    def fit(self, X, y):
        logging.getLogger('statsmodels').setLevel(logging.WARNING)
        self.model_ = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order).fit()
        return self
    
    def predict(self, X):
        return self.model_.predict(start=0, end=len(X)-1).tolist()
