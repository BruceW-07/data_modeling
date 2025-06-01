# 基于SVM的AdaBoost增强手写数字分类

## 项目概述

本项目实现了基于SVM的AdaBoost算法用于MNIST手写数字分类任务。主要内容包括：

1. 比较线性核函数的SVM和RBF核函数的SVM的性能
2. 从零实现AdaBoost算法，对比决策树桩和线性SVM作为基分类器的性能

## 项目结构

- `main.py`: 主程序入口
- `data_loader.py`: 数据加载与预处理
- `svm_models.py`: SVM模型实现
- `adaboost.py`: AdaBoost算法实现
- `visualization.py`: 结果可视化
- `report.md`: 实验报告

## 使用方法

1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

2. 运行主程序：
   ```
   python main.py
   ```

## 实验结果

运行程序后，将生成以下结果文件：

- `bin/{timestamp}/svm_model_performance_comparison_{timestamp}.png`: SVM模型性能对比图
- `bin/{timestamp}/adaboost_model_performance_comparison_{timestamp}.png`: AdaBoost模型性能对比图
- `bin/{timestamp}/decision_stump_performance_analysis_{timestamp}.png`: 决策树桩AdaBoost学习曲线
- `bin/{timestamp}/linear_svm_performance_analysis_{timestamp}.png`: 线性SVM AdaBoost学习曲线
- `bin/{timestamp}/models/`：包含训练好的模型文件
- `results/{timestamp}/model_performance_comparison_{timestamp}.txt`: 模型性能对比结果文本文件

完整实验结果和分析请参考`report.md`文件。
