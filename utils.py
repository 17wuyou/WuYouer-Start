# utils.py

import os
import pickle
import logging
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossLogger:
    """
    一个用于记录和可视化训练/验证损失的类。
    为每次运行创建一个带时间戳的专属日志目录。
    """
    def __init__(self, log_dir_base):
        """
        初始化Logger。
        :param log_dir_base: 'logs/' 目录的路径。
        """
        # 1. 创建带时间戳的专属运行目录
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.run_log_dir = os.path.join(log_dir_base, f"loss_{timestamp}")
        os.makedirs(self.run_log_dir, exist_ok=True)
        
        # 2. 定义所有日志文件的完整路径
        self.train_loss_path = os.path.join(self.run_log_dir, "epoch_loss.txt")
        self.val_loss_path = os.path.join(self.run_log_dir, "epoch_val_loss.txt")
        self.plot_path = os.path.join(self.run_log_dir, "epoch_loss.png")
        
        # 3. 初始化用于存储损失值的列表
        self.train_losses = []
        self.val_losses = []

        # 4. (推荐) 初始化TensorBoard写入器，用于生成events文件
        self.tensorboard_writer = SummaryWriter(self.run_log_dir)

        logging.info(f"Logging for this run will be saved in: {self.run_log_dir}")

    def log_epoch(self, train_loss, val_loss, epoch):
        """在每个epoch结束后记录损失。"""
        # 记录到列表
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # 写入到 .txt 文件 (追加模式)
        with open(self.train_loss_path, 'a') as f:
            f.write(f"{train_loss}\n")
        with open(self.val_loss_path, 'a') as f:
            f.write(f"{val_loss}\n")
            
        # 写入到 TensorBoard
        self.tensorboard_writer.add_scalars('Loss', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)

    def plot_and_save(self):
        """在训练结束后，绘制并保存损失曲线图。"""
        if not self.train_losses or not self.val_losses:
            logging.warning("No losses recorded, skipping plot generation.")
            return

        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-o', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-o', label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.plot_path)
        plt.close() # 关闭图像，防止在Jupyter等环境中显示
        logging.info(f"Loss plot saved to {self.plot_path}")

    def close(self):
        """关闭所有写入器。"""
        self.tensorboard_writer.close()


# --- 以下是原有的工具函数，保持不变 ---
def load_adj_matrix(filepath):
    # ... (代码不变)
    if not os.path.exists(filepath):
        logging.error(f"Adjacency matrix file not found at {filepath}")
        raise FileNotFoundError(f"Adjacency matrix file not found at {filepath}")
    try:
        with open(filepath, 'rb') as f:
            sensor_info = pickle.load(f, encoding='latin1')
        adj_mx = sensor_info[2]
        logging.info(f"Adjacency matrix loaded successfully from {filepath}.")
        return adj_mx
    except Exception as e:
        logging.error(f"Could not load adjacency matrix: {e}")
        raise

def calculate_diffusion_matrix(adj_mx, num_diffusion_steps):
    # ... (代码不变)
    def calculate_random_walk_matrix(adj_mx):
        d_out = np.sum(adj_mx, axis=1)
        d_inv = np.power(d_out, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = np.diag(d_inv)
        return d_mat_inv @ adj_mx
    forward_transition = calculate_random_walk_matrix(adj_mx)
    backward_transition = calculate_random_walk_matrix(adj_mx.T)
    diffusion_matrices = []
    for k in range(num_diffusion_steps):
        diffusion_matrices.append(torch.from_numpy(np.linalg.matrix_power(forward_transition, k)).float())
        diffusion_matrices.append(torch.from_numpy(np.linalg.matrix_power(backward_transition, k)).float())
    return diffusion_matrices

class StandardScaler:
    # ... (代码不变)
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def generate_sliding_windows(data, seq_len, horizon):
    # ... (代码不变)
    num_samples, num_nodes, num_features = data.shape
    total_len = seq_len + horizon
    xs, ys = [], []
    for i in range(num_samples - total_len + 1):
        x = data[i : i + seq_len, ...]
        y = data[i + seq_len : i + total_len, ...]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)