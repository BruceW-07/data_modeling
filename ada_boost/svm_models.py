import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel='linear', C=1.0):
    """
    训练SVM模型并评估性能
    
    参数:
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    kernel (str): 核函数类型 ('linear' 或 'rbf')
    C (float): 正则化参数
    
    返回:
    model: 训练好的SVM模型
    accuracy: 分类准确率
    f1: F1 分数
    training_time: 训练时间
    """
    print(f"训练使用{kernel}核的SVM...")
    
    # 创建SVM模型 - SVC默认使用one-vs-one策略进行多分类
    model = SVC(kernel=kernel, C=C, decision_function_shape='ovr')
    
    # 测量训练时间
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 预测并评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"{kernel}核SVM - 准确率: {accuracy:.4f}, F1值: {f1:.4f}")
    print(f"训练时间: {training_time:.2f} 秒")
    
    return model, accuracy, f1, training_time

def compare_svm_kernels(X_train, X_test, y_train, y_test):
    """
    比较不同核函数的SVM性能
    
    参数:
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    
    返回:
    results: 包含模型性能对比的字典
    """
    results = {}
    
    # 线性核SVM
    linear_model, linear_acc, linear_f1, linear_time = train_and_evaluate_svm(
        X_train, X_test, y_train, y_test, kernel='linear'
    )
    results['linear'] = {
        'model': linear_model,
        'accuracy': linear_acc,
        'f1_score': linear_f1,
        'training_time': linear_time
    }
    
    # RBF核SVM
    rbf_model, rbf_acc, rbf_f1, rbf_time = train_and_evaluate_svm(
        X_train, X_test, y_train, y_test, kernel='rbf'
    )
    results['rbf'] = {
        'model': rbf_model,
        'accuracy': rbf_acc,
        'f1_score': rbf_f1,
        'training_time': rbf_time
    }
    
    return results
