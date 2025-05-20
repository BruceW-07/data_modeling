import logging
import os
import time
from datetime import datetime

def setup_logger(log_dir="logs"):
    """
    设置日志记录器，同时输出到控制台和文件
    
    参数:
    log_dir (str): 日志文件存储目录
    
    返回:
    logger: 配置好的日志记录器
    """
    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 设置日志文件名，包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"adaboost_{timestamp}.log")
    
    # 创建 logger 实例
    logger = logging.getLogger("adaboost")
    logger.setLevel(logging.INFO)
    
    # 如果 logger 已经有处理器，则先清除
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 将处理器添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"日志记录已开始，日志文件: {log_file}")
    
    return logger
