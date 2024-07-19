import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties  
import os

fontset = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=12)  # 微軟標準

# for model visualization
def plot_feature(importance, k, X_colname, name, base_folder):
    '''
    變數重要度(By Linear, Tree-based)
    '''
    k = min(k, len(X_colname))
    indices_of_largest = np.argsort(np.abs(importance), )[-k:][::-1]
    k_importance = [X_colname[i] for i in indices_of_largest]
    
    plt.figure(figsize=(8, 6))
    bars = plt.barh(k_importance, np.abs(importance)[indices_of_largest])
    plt.xlabel('Importance', size=12)
    plt.title(f'{name} Feature Importance', size=13, fontproperties=fontset)
    plt.yticks(range(k), labels=k_importance, fontproperties=fontset)
    plt.gca().invert_yaxis()  
    plt.tight_layout()
    output_path = os.path.join(base_folder, 'result/feature_importance', f'{name}_featimp.png')
    plt.savefig(output_path)
    plt.close()
    # plt.show()


def plot_true_vs_pred(y_true, y_pred, name, target_column, base_folder):
    """
    實際值和預測值的密度圖。
    """
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    # 設置調色板
    sns.set_palette("pastel")

    # 繪製密度圖
    plt.figure(figsize=(6, 4.5))
    sns.kdeplot(df['Actual'], shade=True, color="#3498db", label="Actual")  # Blue
    sns.kdeplot(df['Predicted'], shade=True, color="#e74c3c", label="Predicted")  # Red
    plt.legend(prop=fontset)
    plt.xlabel(target_column, fontproperties=fontset)
    plt.ylabel("Density")
    plt.title('Density Plot of Actual vs Predicted', fontproperties=fontset)
    plt.tight_layout()
    output_path = os.path.join(base_folder, 'result/density_plot', f'{name}_densplot.png')
    plt.savefig(output_path)
    plt.close()
    # plt.show()
