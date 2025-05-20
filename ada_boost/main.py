import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # 修改导入

from data_loader import load_mnist_data
from svm_models import compare_svm_kernels
from adaboost import compare_adaboost_base_estimators, AdaBoost
from evaluation import evaluate_models, detailed_model_analysis
from visualization import plot_performance_comparison, plot_confusion_matrix, plot_learning_curves

def main():
    """
    主程序入口
    """
    # 设置随机种子
    np.random.seed(42)
    
    # 加载并预处理MNIST数据集
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # 为降低计算复杂度，可以减少样本数量
    # 如处理时间过长，可以取消下面的注释
    # sample_size = 10000  # 每个集合的样本数量
    # X_train = X_train[:sample_size]
    # y_train = y_train[:sample_size]
    # X_test = X_test[:2000]
    # y_test = y_test[:2000]
    
    # 多分类任务，直接使用原始数字标签（0-9）
    print("\n1. 比较不同核函数的SVM性能")
    svm_results = compare_svm_kernels(X_train, X_test, y_train, y_test)
    
    print("\n2. 比较不同基分类器的AdaBoost性能")
    n_classes = len(np.unique(y_train))  # 类别数量
    print(f"检测到的类别数量: {n_classes}")
    
    adaboost_results = compare_adaboost_base_estimators(
        X_train, X_test, y_train, y_test, n_estimators=30, n_classes=n_classes
    )
    
    # 评估所有模型性能
    performance_df = evaluate_models(svm_results, adaboost_results)
    print("\n模型性能比较:")
    print(performance_df)
    
    # 绘制性能比较图
    plot_performance_comparison(performance_df)
    
    # 针对每个模型进行详细分析
    for kernel, metrics in svm_results.items():
        report, conf_matrix = detailed_model_analysis(
            metrics['model'], X_test, y_test, f'SVM ({kernel})'
        )
        plot_confusion_matrix(
            conf_matrix, 
            f'SVM ({kernel})', 
            f'confusion_matrix_svm_{kernel}.png'
        )
    
    for estimator_type, metrics in adaboost_results.items():
        base_name = '决策树桩' if estimator_type == 'tree' else '线性SVM'
        report, conf_matrix = detailed_model_analysis(
            metrics['model'], X_test, y_test, f'AdaBoost with {base_name}'
        )
        plot_confusion_matrix(
            conf_matrix, 
            f'AdaBoost with {base_name}', 
            f'confusion_matrix_adaboost_{estimator_type}.png'
        )
        
        # 绘制AdaBoost学习曲线
        error_rates = metrics['model'].error_rates
        plot_learning_curves(
            error_rates, 
            f'AdaBoost with {base_name}', 
            f'learning_curve_adaboost_{estimator_type}.png'
        )
    
    print("分析完成。结果和可视化已保存。")

if __name__ == "__main__":
    main()
