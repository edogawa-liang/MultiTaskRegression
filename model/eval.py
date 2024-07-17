from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# MAPE
def _mape(y_true, y_pred):
    return (abs((y_true - y_pred) / (y_true + 1e-10)).mean()) 


# Adjusted R-squared
def _adjusted_r2(y_true, y_pred, X):
    n = len(y_true)
    p = X.shape[1]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1 + 1e-10)


# 評估指標
def _calculate_metrics(y_true, y_pred, X):
    r2 = r2_score(y_true, y_pred)
    adj_r2 = _adjusted_r2(y_true, y_pred, X)
    rmse = mean_squared_error(y_true, y_pred)**0.5
    mape_value = _mape(y_true, y_pred)

    return {'R-squared': r2, 'Adjusted R-squared': adj_r2, 'RMSE': rmse, 'MAPE': mape_value}


# 計算模型表現
def performance(model, X, y):
    """
    計算模型表現
    """
    y_pred = model.predict(X)
    metrics = _calculate_metrics(y, y_pred, X)
    
    rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}

    return y_pred, rounded_metrics