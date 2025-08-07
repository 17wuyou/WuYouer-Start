# run.py

import os
import argparse
import sys
import yaml
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from pandas import read_hdf # 使用 pandas 加载数据

from models import DCRNN
# 注意：我们不再从utils导入load_sensor_data，因为逻辑已移入此文件
from utils import (load_adj_matrix, calculate_diffusion_matrix,  
                   StandardScaler, generate_sliding_windows)


def setup_logging(config):
    log_dir = config['log']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, "run.log")),
                            logging.StreamHandler()
                        ])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_dataloaders(config):
    # --- 核心修改区域开始 ---
    # 加载HDF5文件为Pandas DataFrame
    if not os.path.exists(config['data']['sensor_data_path']):
        logging.error(f"Sensor data file not found at {config['data']['sensor_data_path']}")
        raise FileNotFoundError
    df = read_hdf(config['data']['sensor_data_path'])
    
    # 从DataFrame中提取纯数值，作为我们的基础数据
    sensor_data = df.values

    # 特征工程: 使用DataFrame的索引来创建时间特征
    # 修正: 从 df.index 获取时间信息，而不是 data.index
    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_of_day = np.tile(time_ind, [df.shape[1], 1]).transpose()
    
    # 将基础数据和时间特征拼接起来
    # sensor_data shape: (num_samples, num_nodes) -> (num_samples, num_nodes, 1)
    # time_of_day shape: (num_samples, num_nodes) -> (num_samples, num_nodes, 1)
    # data shape: (num_samples, num_nodes, 2)
    data = np.concatenate(
        [sensor_data.reshape(-1, df.shape[1], 1), 
         time_of_day.reshape(-1, df.shape[1], 1)], 
        axis=-1
    )
    # --- 核心修改区域结束 ---

    # 划分数据集
    num_samples = data.shape[0]
    train_end = int(num_samples * config['data']['train_split'])
    val_end = train_end + int(num_samples * config['data']['val_split'])
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 数据标准化 (只对第一个特征，即速度进行标准化)
    scaler = StandardScaler(mean=train_data[..., 0].mean(), std=train_data[..., 0].std())
    for category in [train_data, val_data, test_data]:
        category[..., 0] = scaler.transform(category[..., 0])

    # 创建滑动窗口
    seq_len, horizon = config['train']['seq_len'], config['train']['horizon']
    x_train, y_train = generate_sliding_windows(train_data, seq_len, horizon)
    x_val, y_val = generate_sliding_windows(val_data, seq_len, horizon)
    x_test, y_test = generate_sliding_windows(test_data, seq_len, horizon)
    
    # 转换为 PyTorch Tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # 创建 DataLoader
    batch_size = config['train']['batch_size']
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

def train_epoch(model, dataloader, optimizer, loss_fn, device, config):
    model.train()
    total_loss = 0
    for x_batch, y_batch in tqdm(dataloader, desc="Training"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch, y_batch)
        
        loss = loss_fn(output, y_batch[..., :config['model']['out_features']])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_norm_clip'])
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, loss_fn, device, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Validating"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_fn(output, y_batch[..., :config['model']['out_features']])
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    setup_logging(config)
    set_seed(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        adj_matrix = load_adj_matrix(config['data']['adj_path'])
        train_loader, val_loader, _, scaler = prepare_dataloaders(config)
    except FileNotFoundError:
        logging.error("Failed to load data, please check file paths in config.yaml and data folder structure.")
        sys.exit(1)

    diffusion_matrices = calculate_diffusion_matrix(adj_matrix, config['model']['num_diffusion_steps'])
    
    model = DCRNN(config, diffusion_matrices).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    loss_fn = nn.L1Loss()

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['train']['epochs'] + 1):
        logging.info(f"--- Epoch {epoch}/{config['train']['epochs']} ---")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, config)
        logging.info(f"Epoch {epoch} Training MAE Loss: {train_loss:.4f}")
        val_loss = validate_epoch(model, val_loader, loss_fn, device, config)
        logging.info(f"Epoch {epoch} Validation MAE Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_save_path = os.path.join(config['log']['model_save_dir'], "best_dcrnn_model.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Validation loss improved. Saved model to {model_save_path}")
        else:
            patience_counter += 1
            logging.info(f"Validation loss did not improve. Patience: {patience_counter}/{config['train']['patience']}")

        if patience_counter >= config['train']['patience']:
            logging.info("Early stopping triggered.")
            break
            
    logging.info("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the DCRNN model.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config)