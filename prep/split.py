from sklearn.model_selection import train_test_split
from prep.load import load_data
import logging

def split_dataset(data, target_column, test_size, random_state=None, test_file=None, sheet_name=None, time_column=None):
    '''
    劃分訓練集與測試集
    
    Parameters:
    data (DataFrame)
    target_column (str): 標籤名
    method (str): 劃分方法 ('random', 'time', ) (default: 'random')
    test_size (float): 測試集比例 (default: 0.2)
    random_state (int): 隨機種子 (default: None)
    test_file (str, optional): 測試集檔案路徑 (若method為'additional_test'必須提供)
    
    Returns:
    X_train, X_test, y_train, y_test 
    '''
    try:
        if target_column not in data.columns:
            raise ValueError(f"標籤 '{target_column}' 不在資料集中。")
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # 若有提供測試集檔案，則讀取
        if test_file is not None and test_file != '' :
            print("提供測試集檔案")
            logging.info(f"Loading additional test file: {test_file}")
            test_data = load_data(test_file, sheet_name=sheet_name)
            if target_column not in test_data.columns:
                raise ValueError(f"標籤 '{target_column}' 不在測試資料集中。")
            X_train = X
            y_train = y
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            logging.info("Additional test dataset loaded and split.")

        # 否則進行劃分
        else:
            print("由提供的資料劃分")
            if time_column is None:
                # 隨機劃分
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                logging.info(f"劃分資料集: 隨機劃分資料集，測試集比例為 {test_size} ")
            
            elif time_column is not None:
                # 按時間劃分 (假設資料已按時間排序)
                if time_column not in data.columns:
                    raise ValueError(f"時間欄位 '{time_column}' 不在資料集中。")
                
                split_index = int(len(data) * (1 - test_size))
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]
                logging.info(f"劃分資料集: 按時間劃分資料集，測試集比例為 {test_size}")
            
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        raise RuntimeError(f"Failed to split dataset: {str(e)}")