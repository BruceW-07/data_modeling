import numpy as np
import time
import pandas as pd
import os

from data_loader import load_mnist_data, load_mnist_local
from svm_models import compare_svm_kernels
from adaboost import compare_adaboost_base_estimators
from visualization import plot_performance_comparison, plot_adaboost_performance

def get_performance_df(svm_results, adaboost_results):
    """
    将测试结果转为 DataFrame 格式
    
    参数:
    svm_results: SVM模型结果
    adaboost_results: AdaBoost模型结果
    
    返回:
    performance_df: 包含性能比较的DataFrame
    """
    performance_data = []
    
    # 提取SVM模型性能
    for kernel, metrics in svm_results.items():
        performance_data.append({
            'Model': f'SVM ({kernel})',
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1_score'],
            'Training Time (s)': metrics['training_time']
        })
    
    # 提取AdaBoost模型性能
    for estimator_type, metrics in adaboost_results.items():
        # if estimator_type == 'sklearn_tree':
        #     base_name = 'sklearn decision tree'
        # elif estimator_type == 'sklearn_svm':
        #     base_name = 'sklearn SVM'
        performance_data.append({
            'Model': f'AdaBoost with {estimator_type}',
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1_score'],
            'Training Time (s)': metrics['training_time'],
            'Number of Estimators': metrics['n_estimators']
        })
    
    # 创建性能比较DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    return performance_df

def save_results(performance_df, svm_results, adaboost_results, result_dir, timestamp):
    """
    保存性能比较结果和训练好的模型

    参数:
    performance_df: 包含性能比较的DataFrame
    svm_results: SVM模型结果
    adaboost_results: AdaBoost模型结果
    result_dir: 结果保存目录
    """
    result_dir = result_dir + f'/{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    csv_path = f'{result_dir}/model_performance_comparison_{timestamp}.csv'
    performance_df.to_csv(csv_path, index=True)
    print(f"\n性能比较结果已保存至: {csv_path}")

    # 保存训练好的模型
    model_dir = 'bin' + f'/{timestamp}/models'
    os.makedirs(model_dir, exist_ok=True)
    for model_name, metrics in {**svm_results, **adaboost_results}.items():
        model = metrics['model']
        model_path = f'{model_dir}/{model_name}_model_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            import pickle
            pickle.dump(model, f)
        print(f"模型 {model_name} 已保存至: {model_path}")

def main():
    """
    主程序入口
    """
    # 设置随机种子
    np.random.seed(27)
    
    # 加载并预处理MNIST数据集
    # X_train, X_test, y_train, y_test = load_mnist_data()
    X_train, X_test, y_train, y_test = load_mnist_local()
    
    # sample_size = 1000
    # test_size = 200
    # X_train = X_train[:sample_size]
    # y_train = y_train[:sample_size]
    # X_test = X_test[:test_size]
    # y_test = y_test[:test_size]
    
    # SVM基础实现
    print("\n1. 比较不同核函数的SVM性能")
    svm_results = compare_svm_kernels(X_train, X_test, y_train, y_test)
    
    # AdaBoost提升SVM性能
    print("\n2. 比较不同基分类器的AdaBoost性能")
    adaboost_results = compare_adaboost_base_estimators(
        X_train, X_test, y_train, y_test, n_estimators=30, n_classes=10
    )
    
    # 获取性能比较DataFrame
    performance_df = get_performance_df(svm_results, adaboost_results)
    print("\n模型性能比较:")
    print(performance_df)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 保存结果
    save_results(performance_df, svm_results, adaboost_results, 'results', timestamp)

    # 绘制性能比较图
    plot_performance_comparison(performance_df, timestamp)

    # 绘制AdaBoost性能分析图
    plot_adaboost_performance(adaboost_results, timestamp)
    
    print("分析完成。结果和可视化已保存。")

if __name__ == "__main__":
    main()
