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
import h5py # 引入 h5py
import pandas as pd # 引入 pandas
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from models import DCRNN
from utils import (LossLogger, load_adj_matrix, calculate_diffusion_matrix,  
                   StandardScaler, generate_sliding_windows)


def setup_logging(config):
    # ... (代码不变)
    log_dir = config['log']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, "run.log")),
                            logging.StreamHandler()
                        ])

def set_seed(seed):
    # ... (代码不变)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_dataloaders(config):
    # --- 核心修复：绕过 pandas.read_hdf ---
    filepath = config['data']['sensor_data_path']
    if not os.path.exists(filepath):
        logging.error(f"Sensor data file not found at {filepath}")
        raise FileNotFoundError
        
    try:
        # 使用更底层的 h5py 库来读取文件，以绕过兼容性问题
        with h5py.File(filepath, 'r') as f:
            # 从HDF5文件中提取原始数据、索引和列名
            data_values = f['df']['block0_values'][:]
            # 时间戳索引是字节字符串，需要解码和转换
            time_index = pd.to_datetime([x.decode('utf-8') for x in f['df']['axis1'][:]])
            # 传感器ID也是字节字符串，需要解码
            sensor_ids = [x.decode('utf-8') for x in f['df']['axis0'][:]]
        
        # 手动将原始数据重新组装成我们需要的DataFrame
        df = pd.DataFrame(data_values, index=time_index, columns=sensor_ids)
        logging.info("Successfully loaded HDF5 data using h5py and reconstructed DataFrame.")
    except Exception as e:
        logging.error(f"Failed to read HDF5 file with h5py. Error: {e}")
        raise

    sensor_data = df.values
    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_of_day = np.tile(time_ind, [df.shape[1], 1]).transpose()
    data = np.concatenate(
        [sensor_data.reshape(-1, df.shape[1], 1), 
         time_of_day.reshape(-1, df.shape[1], 1)], 
        axis=-1
    )
    # --- 修复结束 ---

    num_samples = data.shape[0]
    train_end = int(num_samples * config['data']['train_split'])
    val_end = train_end + int(num_samples * config['data']['val_split'])
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    scaler = StandardScaler(mean=train_data[..., 0].mean(), std=train_data[..., 0].std())
    for category in [train_data, val_data, test_data]:
        category[..., 0] = scaler.transform(category[..., 0])

    seq_len, horizon = config['train']['seq_len'], config['train']['horizon']
    x_train, y_train = generate_sliding_windows(train_data, seq_len, horizon)
    x_val, y_val = generate_sliding_windows(val_data, seq_len, horizon)
    x_test, y_test = generate_sliding_windows(test_data, seq_len, horizon)
    
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    batch_size = config['train']['batch_size']
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler


def train_epoch(model, dataloader, optimizer, loss_fn, device, config):
    # ... (代码不变)
    model.train()
    total_loss = 0
    for x_batch, y_batch in tqdm(dataloader, desc="Training"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_batch_targets = y_batch[..., :config['model']['out_features']]
        output = model(x_batch, y_batch_targets)
        loss = loss_fn(output, y_batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_norm_clip'])
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device, config):
    # ... (代码不变)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Validating"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            y_batch_targets = y_batch[..., :config['model']['out_features']]
            loss = loss_fn(output, y_batch_targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main(config_path):
    # ... (代码不变, 除了在 try-except 块中打印更详细的错误) ...
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    setup_logging(config)
    loss_logger = LossLogger(config['log']['log_dir'])
    set_seed(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        adj_matrix = load_adj_matrix(config['data']['adj_path'])
        train_loader, val_loader, _, scaler = prepare_dataloaders(config)
    except Exception as e:
        logging.error(f"Failed during data preparation: {e}", exc_info=True)
        loss_logger.close() 
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
        loss_logger.log_epoch(train_loss, val_loss, epoch)
        
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
    loss_logger.plot_and_save()
    loss_logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the DCRNN model.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config)