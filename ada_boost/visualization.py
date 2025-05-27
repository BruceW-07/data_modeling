import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def plot_performance_comparison(performance_df, timestamp, save_path='bin'):
    """
    绘制各模型性能比较图
    
    参数:
    performance_df: 包含性能比较的DataFrame
    save_path: 保存图片的路径
    """
    save_path = save_path + f'/{timestamp}'
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
    plt.savefig(save_path + f'/model_performance_comparison_{timestamp}.png', dpi=300)
    plt.close()

def plot_adaboost_performance(adaboost_result, timestamp, save_path='bin'):
    """
    绘制AdaBoost模型在不同n_estimators下的性能变化图
    
    参数:
    adaboost_result: 包含不同AdaBoost模型结果的字典
    timestamp: 时间戳，用于文件命名
    save_path: 保存图片的路径
    """
    save_path = save_path + f'/{timestamp}'
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 遍历adaboost_result中的每个模型
    for model_name, model_data in adaboost_result.items():
        if 'performance_analysis' not in model_data:
            continue
            
        performance_analysis = model_data['performance_analysis']
        
        # 获取n_estimators和对应的性能指标
        n_values = list(performance_analysis.keys())
        accuracy_values = [perf['accuracy'] for n, perf in performance_analysis.items()]
        f1_values = [perf['f1_score'] for n, perf in performance_analysis.items()]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 绘制accuracy随n_estimators变化的曲线
        ax1.plot(n_values, accuracy_values, 'o-', linewidth=1, color='blue')
        ax1.set_xlabel('Number of Estimators')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{model_name} - Accuracy vs Number of Estimators')
        ax1.grid(True)
        
        # 绘制F1 Score随n_estimators变化的曲线
        ax2.plot(n_values, f1_values, 'o-', linewidth=1, color='orange')
        ax2.set_xlabel('Number of Estimators')
        ax2.set_ylabel('F1 Score')
        ax2.set_title(f'{model_name} - F1 Score vs Number of Estimators')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path + f'/{model_name}_performance_analysis_{timestamp}.png', dpi=300)
        plt.close()