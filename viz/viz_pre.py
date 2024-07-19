import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from matplotlib.font_manager import FontProperties  
import os
import logging

fontset = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=12)  # 微軟標準

# for data visualization
def _corr_heatmap(df, base_folder):
    """ 
    繪製相關性熱圖
    """
    # 只取數值型變數
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=[8, 6])
    corr = numeric_df.corr(min_periods=1)
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm')
    # plt.title('變數相關性熱圖', fontproperties=fontset)
    plt.title('Correlation Heatmap', fontproperties=fontset)
    plt.xticks(fontproperties=fontset)
    plt.yticks(fontproperties=fontset)
    plt.tight_layout()
    output_path = os.path.join(base_folder, 'fig', 'corr_heatmap.png')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved correlation heatmap to {output_path}")


def _plot_cont(X, y, column, target, base_folder):
    """
    continuous 變數與 y 的關係
    """
    plt.figure(figsize=(6, 4.5))
    sns.scatterplot(x=X[column], y=y)
    plt.title(f'{column} vs {target} Scatter Plot', fontproperties=fontset)
    plt.ylabel(target, fontproperties=fontset)
    plt.xlabel(column, fontproperties=fontset) 
    plt.tight_layout()
    output_path = os.path.join(base_folder, 'fig', f'{column}_{target}.png')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved {column} vs {target} scatter plot to {output_path}")


def _plot_cat(X, y, column, target, base_folder):
    """
    nominal, ordinal 變數與y的關係
    """
    plt.figure(figsize=(6, 4.5))
    sns.barplot(x=X[column], y=y)
    plt.title(f'{column} vs {target} Bar Plot', fontproperties=fontset)
    plt.xlabel(column, fontproperties=fontset)
    plt.ylabel(target, fontproperties=fontset)
    plt.tight_layout()
    output_path = os.path.join(base_folder, 'fig', f'{column}_{target}.png')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved {column} vs {target} bar plot to {output_path}")


def _plot_time(X, y, time_column, target, base_folder):
    """
    繪製時間變數與 y 的關係
    """
    plt.figure(figsize=(8, 4))
    plt.plot_date(y, '-')
    plt.title(f'Time vs {target} Line Plot', fontproperties=fontset)
    plt.xlabel(time_column, fontproperties=fontset)
    plt.ylabel(target, fontproperties=fontset)
    
    # 設定 xtick 每 6 個一個
    show_index = [i for i in range(0, y.shape[0], 6)]
    show_col = X[time_column].iloc[show_index]
    plt.xticks(show_index, show_col)
    
    plt.tight_layout()
    output_path = os.path.join(base_folder, 'fig', f'{time_column}_{target}.png')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved time vs {target} line plot to {output_path}")


def _auto_plot(X, y, column, target, base_folder):
    """
    自動選擇並繪製變數與 y 的關係
    """
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('seaborn').setLevel(logging.WARNING)
    try:
        if is_numeric_dtype(X[column]):
            if X[column].nunique() > 7:
                _plot_cont(X, y, column, target, base_folder)
            else:
                _plot_cat(X, y, column, target, base_folder)
        elif is_categorical_dtype(X[column]):
            _plot_cat(X, y, column, target, base_folder)
    except Exception as e:
        return f"錯誤訊息: {e}"
    
    return None


def plot_Xy(df, target, plot_col='all', time_column=None, base_folder='.'):
    """
    根據變數類型繪製每個變數與目標變數 y 之間的關係
    """
    X = df.drop(columns=[target])
    y = df[target]

    if plot_col == 'nodraw':
        return None

    else:
        # 若只挑選幾個欄位，將 plot_col 轉為list
        if isinstance(plot_col, str):
            if plot_col == 'all':
                _corr_heatmap(df, base_folder)
                columns = X.columns
            else:
                columns = [col.strip() for col in plot_col.split(',')]
        else:
            columns = plot_col

        error_messages = []
        if time_column is not None and time_column in X.columns:
            try:    
                _plot_time(X, y, time_column, target, base_folder)
                # remove time_column from columns
                columns = [col for col in columns if col != time_column]
            except Exception as e:
                error_messages.append(f"Error: {e}")
        
        for column in columns:
            if column in X.columns:
                error_message = _auto_plot(X, y, column, target, base_folder)
                if error_message:
                    error_messages.append(error_message)
            else:
                error_messages.append(f'Column {column} not in dataset')
        
        if error_messages:
            return error_messages
        
        return None
