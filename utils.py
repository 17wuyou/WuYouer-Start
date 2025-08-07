# utils.py

import os
import pickle
import logging
import numpy as np
import torch

def load_adj_matrix(filepath):
    """从 .pkl 文件加载邻接矩阵"""
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
    """计算扩散转移矩阵"""
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
    """标准差标准化器"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def generate_sliding_windows(data, seq_len, horizon):
    """
    为 Seq2Seq 模型生成滑动窗口数据对
    """
    num_samples, num_nodes, num_features = data.shape
    total_len = seq_len + horizon
    
    xs, ys = [], []
    for i in range(num_samples - total_len + 1):
        x = data[i : i + seq_len, ...]
        y = data[i + seq_len : i + total_len, ...]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)