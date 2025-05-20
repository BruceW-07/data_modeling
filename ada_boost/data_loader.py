import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def load_mnist_data():
    """
    加载MNIST数据集并进行预处理，使用原始的训练集和测试集划分
    
    参数:
    test_size: 保留参数，不再使用（为保持接口一致）
    random_state: 保留参数，不再使用（为保持接口一致）
    
    返回:
    X_train, X_test, y_train, y_test: 处理后的训练和测试数据
    """
    # 加载MNIST数据集
    print("正在加载MNIST数据集...")
    mnist = fetch_openml('mnist_784', version=1)
        
    X, y = mnist.data, mnist.target
    
    # 将标签转换为整数
    y = y.astype(np.int64)
    
    # 使用MNIST原始的训练集和测试集划分
    # MNIST数据集的前60,000个样本是训练集，后10,000个样本是测试集
    X_train = X[:60000]
    y_train = y[:60000]
    X_test = X[60000:]
    y_test = y[60000:]
    
    # 标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"数据集加载完成: X_train形状: {X_train.shape}, y_train形状: {y_train.shape}")
    print(f"X_test形状: {X_test.shape}, y_test形状: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test
