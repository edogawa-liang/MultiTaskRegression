import os
import pandas as pd
import zipfile
from ftplib import FTP

def create_folders(base_folder):
    """
    建立資料夾。
    """
    for folder in ['data', 'fig', 'saved_model', 'result', 'result/feature_importance', 'result/density_plot', 'result/prediction']:
        folder_path = os.path.join(base_folder, folder)
        os.makedirs(folder_path, exist_ok=True)


def load_data(file_path, sheet_name=None):
    """
    根據文件擴展名讀取 csv 或 xlsx 文件。
    """
    ext = os.path.splitext(file_path)[1]
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.xlsx':
        if sheet_name is not None and sheet_name != '':
            return pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    
    
# 由FTP讀取檔案
def download_from_ftp(ftp_url: str, local_path: str):
    # 解析 FTP 連結
    url_parts = ftp_url.split('/')
    ftp_host = url_parts[2]
    ftp_file_path = '/'.join(url_parts[3:])
    # 連接 FTP 並下載文件
    ftp = FTP(ftp_host)
    ftp.login()
    ftp.sendcmd('OPTS UTF8 ON')
    with open(local_path, 'wb') as f:
        ftp.retrbinary(f'RETR {ftp_file_path}', f.write)
    ftp.quit()

# 包成壓縮檔
def zip_files(file_list, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_list:
            zipf.write(file, os.path.relpath(file, os.path.dirname(file)))

# 讀取資料夾中的檔案
def get_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list
