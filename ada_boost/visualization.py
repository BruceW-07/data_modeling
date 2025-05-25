import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def plot_performance_comparison(performance_df, save_path='bin'):
    """
    绘制各模型性能比较图
    
    参数:
    performance_df: 包含性能比较的DataFrame
    save_path: 保存图片的路径
    """

    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_labels = performance_df['Model'].tolist()
    ticks_pos = np.arange(len(model_labels))

    # 绘制准确率比较
    sns.barplot(x='Model', y='Accuracy', data=performance_df, ax=axes[0])
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(ticks_pos)
    axes[0].set_xticklabels(model_labels, rotation=45, ha='right')

    # 绘制F1值比较
    sns.barplot(x='Model', y='F1 Score', data=performance_df, ax=axes[1])
    axes[1].set_title('Model F1 Score')
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(ticks_pos)
    axes[1].set_xticklabels(model_labels, rotation=45, ha='right')

    # 绘制训练时间比较
    sns.barplot(x='Model', y='Training Time (s)', data=performance_df, ax=axes[2])
    axes[2].set_title('Model Training Time')
    axes[2].set_xticks(ticks_pos)
    axes[2].set_xticklabels(model_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path + '/model_performance_comparison.png', dpi=300)
    plt.close()