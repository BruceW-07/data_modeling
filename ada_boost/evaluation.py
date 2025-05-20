import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_models(svm_results, adaboost_results):
    """
    综合评估所有模型性能
    
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
        base_name = '决策树桩' if estimator_type == 'tree' else '线性SVM'
        performance_data.append({
            'Model': f'AdaBoost with {base_name}',
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1_score'],
            'Training Time (s)': metrics['training_time']
        })
    
    # 创建性能比较DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    return performance_df

def detailed_model_analysis(model, X_test, y_test, model_name):
    """
    对模型进行详细分析
    
    参数:
    model: 训练好的模型
    X_test, y_test: 测试数据
    model_name (str): 模型名称
    
    返回:
    report: 分类报告
    conf_matrix: 混淆矩阵
    """
    y_pred = model.predict(X_test)
    
    # 计算分类报告
    report = classification_report(y_test, y_pred)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name}的详细分析:")
    print("分类报告:")
    print(report)
    print("混淆矩阵:")
    print(conf_matrix)
    
    return report, conf_matrix
