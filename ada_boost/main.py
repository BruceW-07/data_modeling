import numpy as np
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # 修改导入

from data_loader import load_mnist_data, load_mnist_local
from svm_models import compare_svm_kernels
from adaboost import compare_adaboost_base_estimators, AdaBoost
from evaluation import get_performance_df, detailed_model_analysis
from visualization import plot_performance_comparison, plot_confusion_matrix, plot_learning_curves

def main():
    """
    主程序入口
    """
    # 设置随机种子
    np.random.seed(27)
    
    # 加载并预处理MNIST数据集
    # X_train, X_test, y_train, y_test = load_mnist_data()
    X_train, X_test, y_train, y_test = load_mnist_local()
    
    sample_size = 1000
    test_size = 200
    X_train = X_train[:sample_size]
    y_train = y_train[:sample_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]
    
    # SVM基础实现
    print("\n1. 比较不同核函数的SVM性能")
    svm_results = compare_svm_kernels(X_train, X_test, y_train, y_test)
    
    # AdaBoost提升SVM性能
    print("\n2. 比较不同基分类器的AdaBoost性能")
    adaboost_results = compare_adaboost_base_estimators(
        X_train, X_test, y_train, y_test, n_estimators=30, n_classes=10
    )
    
    # 评估所有模型性能
    performance_df = get_performance_df(svm_results, adaboost_results)
    print("\n模型性能比较:")
    print(performance_df)

    # 保存性能比较结果到CSV文件
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    csv_path = f'{result_dir}/model_performance_comparison.csv'
    performance_df.to_csv(csv_path, index=True)
    print(f"\n性能比较结果已保存至: {csv_path}")
    
    # 绘制性能比较图
    plot_performance_comparison(performance_df)
    
    print("分析完成。结果和可视化已保存。")

if __name__ == "__main__":
    main()
