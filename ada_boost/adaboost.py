import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # 修改导入
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost:
    """
    AdaBoost算法实现 - 支持多分类（SAMME算法）
    """
    def __init__(self, base_estimator_type, n_estimators=50, n_classes=10):
        """
        初始化AdaBoost分类器
        
        参数:
        base_estimator: 基分类器
        n_estimators (int): 基分类器数量
        n_classes (int): 类别数量
        """
        self.base_estimator_type = base_estimator_type
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
        self.error_rates = []
        self.n_classes = n_classes
        
    def fit(self, X, y):
        """
        训练AdaBoost模型（使用SAMME算法解决多分类问题）
        
        参数:
        X: 特征数据
        y: 标签
        
        返回:
        self: 训练好的AdaBoost模型
        """
        n_samples = X.shape[0]
        # 初始化权重
        weights = np.ones(n_samples) / n_samples
        
        # 确保标签从0到n_classes-1
        y_encoded = y.copy()
        
        for i in range(self.n_estimators):
            print(f"训练第{i+1}个基分类器...", end=' ')
            t = time.time()

            # 获取基分类器
            estimator = self.new_estimator()
            
            # 对样本分配对应的权重后进行训练
            estimator.fit(X, y_encoded, sample_weight=weights)

            print(f"耗时: {time.time() - t:.2f}秒")
            
            y_pred = estimator.predict(X)
            
            # 计算错误率
            incorrect = (y_pred != y_encoded)
            error = np.sum(weights * incorrect)
            self.error_rates.append(error)
            
            # 如果错误率太高，停止训练
            if error >= 1.0 - 1e-10:
                if i == 0:  # 如果第一个分类器就出现极端情况
                    raise ValueError("分类器无法学习，错误率过高")
                else:
                    print(f"第{i+1}个基分类器错误率过高，停止训练")
                    break
            
            # 计算基分类器权重（根据 SAMME 算法）
            alpha = np.log((1 - error) / max(error, 1e-10)) + np.log(self.n_classes - 1)
            
            # 更新样本权重
            weights *= np.exp(alpha * incorrect)
            weights /= np.sum(weights)  # 归一化
            
            # 保存基分类器及其权重
            self.estimators.append(estimator)
            self.alphas.append(alpha)
            
        return self
    
    def predict(self, X, n_estimators=None):
        """
        使用AdaBoost模型进行预测 - 多分类版本
        
        参数:
        X: 特征数据
        
        返回:
        y_pred: 预测的标签
        """
        if len(self.estimators) == 0:
            raise ValueError("模型尚未训练")
        
        # 初始化投票矩阵
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes))

        # 如果未指定n_estimators，则使用所有训练的基分类器
        if n_estimators is None:
            n_estimators = len(self.estimators)
        else:
            n_estimators = min(n_estimators, len(self.estimators))
        
        # 汇总所有基分类器的投票
        for i in range(n_estimators):
            estimator = self.estimators[i]
            alpha = self.alphas[i]
            
            # 预测类别
            y_pred = estimator.predict(X)
            
            # 将预测转换为one-hot编码
            for j, pred in enumerate(y_pred):
                votes[j, int(pred)] += alpha
                
        # 返回得票最多的类别
        return np.argmax(votes, axis=1)
    
    def new_estimator(self):
        """
        新建基分类器
        
        返回:
        estimator: 新建的基分类器
        """
        if self.base_estimator_type == 'stump':
            return DecisionTreeClassifier(max_depth=1)  # 决策树桩
        elif self.base_estimator_type == 'svm':
            # return SVC(kernel='linear', probability=True, max_iter=1000, decision_function_shape='ovr') # 线性SVM
            return SVC(kernel='linear', C=1.0, decision_function_shape='ovr') # 线性SVM
        else:
            raise ValueError(f"不支持的基分类器类型：{self.base_estimator_type}.")


def train_and_evaluate_adaboost(X_train, X_test, y_train, y_test, base_estimator_type='stump', n_estimators=50, n_classes=10):
    """
    训练AdaBoost模型并评估性能
    
    参数:
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    base_estimator_type (str): 基分类器类型 ('stump' 或 'svm')
    n_estimators (int): 基分类器数量
    n_classes (int): 类别数量
    
    返回:
    model: 训练好的AdaBoost模型
    accuracy: 分类准确率
    f1: F1 分数
    training_time: 训练时间
    """
    # 选择基分类器
    if base_estimator_type == 'stump':
        estimator_name = "决策树桩"
    elif base_estimator_type == 'svm':
        estimator_name = "线性SVM"
    else:
        raise ValueError("不支持的基分类器类型")
    
    print(f"训练AdaBoost，基分类器为{estimator_name}...")
    
    # 创建AdaBoost模型
    model = AdaBoost(base_estimator_type=base_estimator_type, n_estimators=n_estimators, n_classes=n_classes)
    
    # 测量训练时间
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 预测并评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"AdaBoost with {estimator_name} - 准确率: {accuracy:.4f}, F1值: {f1:.4f}")
    print(f"训练时间: {training_time:.2f} 秒")
    
    return model, accuracy, f1, training_time

def performance_analysis_of_adaboost(model, X_test, y_test, n_estimators):
    """
    对使用 AdaBoost 模型进行性能分析
    
    参数:
    model: 训练好的 AdaBoost 模型
    X_test: 测试数据特征
    y_test: 测试数据标签
    n_estimators: 基分类器数量
    
    返回:
    result: 包含每个基分类器数量的准确率和 F1 分数的字典
    """
    result = {}
    
    step = max(1, n_estimators // 30)  # 每10个基分类器进行一次评估
    for n in range(1, n_estimators + 1, step):
        y_pred = model.predict(X_test, n_estimators=n)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        result[n] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    return result

def compare_adaboost_base_estimators(X_train, X_test, y_train, y_test, n_estimators=50, n_classes=10):
    """
    比较不同基分类器的AdaBoost性能
    
    参数:
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    n_estimators (int): 基分类器数量
    n_classes (int): 类别数量
    
    返回:
    results: 包含模型性能对比的字典
    """
    results = {}
    
    # 决策树桩作为基分类器
    n_estimators = 300
    tree_model, tree_acc, tree_f1, tree_time = train_and_evaluate_adaboost(
        X_train, X_test, y_train, y_test, base_estimator_type='stump', 
        n_estimators=n_estimators, n_classes=n_classes
    )
    print(f"对使用决策树桩的AdaBoost模型进行性能分析...")
    stump_performance_analysis = performance_analysis_of_adaboost(tree_model, X_test, y_test, n_estimators)
    print(f"性能分析完成")
    results['decision stump'] = {
        'model': tree_model,
        'accuracy': tree_acc,
        'f1_score': tree_f1,
        'training_time': tree_time,
        'n_estimators': n_estimators,
        'performance_analysis': stump_performance_analysis
    }
    
    # 线性SVM作为基分类器
    n_estimators = 9
    svm_model, svm_acc, svm_f1, svm_time = train_and_evaluate_adaboost(
        X_train, X_test, y_train, y_test, base_estimator_type='svm', 
        n_estimators=n_estimators, n_classes=n_classes
    )
    print(f"对使用线性SVM的AdaBoost模型进行性能分析...")
    svm_performance_analysis = performance_analysis_of_adaboost(svm_model, X_test, y_test, n_estimators)
    print(f"性能分析完成")
    results[f'linear svm'] = {
        'model': svm_model,
        'accuracy': svm_acc,
        'f1_score': svm_f1,
        'training_time': svm_time,
        'n_estimators': n_estimators,
        'performance_analysis': svm_performance_analysis
    }
    
    return results
