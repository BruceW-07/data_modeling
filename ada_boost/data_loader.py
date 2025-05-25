import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def save_mnist_local():
    # 创建保存数据集的目录
    data_dir = "mnist_data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("正在下载MNIST数据集...")
    mnist = fetch_openml('mnist_784', version=1)
    
    X, y = mnist.data, mnist.target
    y = y.astype(np.int64)
    
    # 划分训练集和测试集
    X_train = X[:60000]
    y_train = y[:60000]
    X_test = X[60000:]
    y_test = y[60000:]
    
    # 保存为NumPy的.npy文件
    print("正在保存数据集到本地...")
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    print(f"数据集已保存到 {os.path.abspath(data_dir)} 目录")


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

def load_mnist_local():
    """
    从本地文件加载已保存的MNIST数据集
    
    返回:
    X_train, X_test, y_train, y_test: 处理后的训练和测试数据
    """
    data_dir = "data"
    
    print("正在从本地加载MNIST数据集...")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        # 标准化处理
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"数据集加载完成: X_train形状: {X_train.shape}, y_train形状: {y_train.shape}")
        print(f"X_test形状: {X_test.shape}, y_test形状: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("本地数据文件不存在，请先运行save_mnist_local.py下载并保存数据集")
        return None, None, None, None


if __name__ == "__main__":
    save_mnist_local()