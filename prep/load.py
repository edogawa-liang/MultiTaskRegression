import os
import pandas as pd

def create_folders(base_folder):
    """
    建立資料夾。
    """
    for folder in ['fig', 'saved_model', 'result', 'result/feature_importance', 'result/density_plot', 'result/prediction']:
        folder_path = os.path.join(base_folder, folder)
        os.makedirs(folder_path, exist_ok=True)


def load_data(file_path, sheet_name=None):
    """
    根據文件擴展名讀取 csv 或 xlsx 文件。
    """
    ext = os.path.splitext(file_path)[1]
    print(ext)
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.xlsx':
        return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")