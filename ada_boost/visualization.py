import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

def plot_performance_comparison(performance_df, save_path='bin/model_performance_comparison.png'):
    """
    绘制各模型性能比较图
    
    参数:
    performance_df: 包含性能比较的DataFrame
    save_path: 保存图片的路径
    """
    # 设置图形风格
    # sns.set(style="whitegrid")
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
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
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(conf_matrix, model_name, save_path=None):
    """
    绘制混淆矩阵热图
    
    参数:
    conf_matrix: 混淆矩阵
    model_name (str): 模型名称
    save_path: 保存图片的路径
    """
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    
    # 绘制热图
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # 添加标签和标题
    plt.title(f'混淆矩阵 - {model_name}')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 设置刻度标签（0-9数字）
    class_labels = range(10)
    plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels)
    plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_learning_curves(error_rates, model_name, save_path=None):
    """
    绘制AdaBoost学习曲线
    
    参数:
    error_rates: 错误率列表
    model_name (str): 模型名称
    save_path: 保存图片的路径
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.plot(range(1, len(error_rates) + 1), error_rates, marker='o')
    plt.title(f'AdaBoost学习曲线 - {model_name}')
    plt.xlabel('迭代次数')
    plt.ylabel('错误率')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
