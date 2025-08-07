# models.py

import torch
import torch.nn as nn
import random  # <--- 在这里添加缺失的导入

class DiffusionConv(nn.Module):
    """扩散卷积层"""
    def __init__(self, in_features, out_features, num_diffusion_steps, adj_matrices):
        super().__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.adj_matrices = adj_matrices
        
        # 这是一个常见的简化实现，将所有扩散结果拼接后通过一个线性层
        self.fc = nn.Linear(in_features * 2 * num_diffusion_steps, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        supports = []
        for P_k in self.adj_matrices:
            # 确保转移矩阵在正确的设备上
            P_k = P_k.to(x.device)
            support = torch.einsum('nn, bni -> bni', P_k, x)
            supports.append(support)
        
        x_g = torch.cat(supports, dim=-1) # (Batch, Num_nodes, In_features * 2 * K)
        return self.fc(x_g)


class DCGRUCell(nn.Module):
    """扩散卷积门控循环单元"""
    def __init__(self, in_features, hidden_features, num_diffusion_steps, adj_matrices):
        super().__init__()
        self.hidden_features = hidden_features
        self.gate_conv = DiffusionConv(in_features + hidden_features, hidden_features * 2, num_diffusion_steps, adj_matrices)
        self.candidate_conv = DiffusionConv(in_features + hidden_features, hidden_features, num_diffusion_steps, adj_matrices)

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=-1)
        gates = self.gate_conv(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_features, dim=-1)
        r = torch.sigmoid(reset_gate)
        u = torch.sigmoid(update_gate)
        combined_candidate = torch.cat([x_t, r * h_prev], dim=-1)
        C = torch.tanh(self.candidate_conv(combined_candidate))
        h_new = u * h_prev + (1 - u) * C
        return h_new


class Encoder(nn.Module):
    def __init__(self, config, adj_matrices):
        super().__init__()
        self.hidden_features = config['model']['hidden_features']
        self.num_layers = config['model']['num_layers']
        
        self.layers = nn.ModuleList([
            DCGRUCell(config['model']['in_features'] if i == 0 else self.hidden_features,
                      self.hidden_features,
                      config['model']['num_diffusion_steps'],
                      adj_matrices)
            for i in range(self.num_layers)
        ])

    def forward(self, x_seq):
        batch_size, _, num_nodes, _ = x_seq.shape
        # 初始化所有层的隐藏状态
        hidden_states = [torch.zeros(batch_size, num_nodes, self.hidden_features, device=x_seq.device) for _ in range(self.num_layers)]

        for t in range(x_seq.shape[1]): # 遍历时间序列
            x_input = x_seq[:, t, :, :]
            for i, layer in enumerate(self.layers):
                # 上一层的输出是下一层的输入
                h_prev = hidden_states[i]
                h_new = layer(x_input, h_prev)
                hidden_states[i] = h_new
                x_input = h_new # 为下一层准备输入
        
        # 编码器只返回最后一层的最后隐藏状态
        return hidden_states[-1]


class Decoder(nn.Module):
    def __init__(self, config, adj_matrices):
        super().__init__()
        self.out_features = config['model']['out_features']
        self.hidden_features = config['model']['hidden_features']
        self.num_layers = config['model']['num_layers']
        
        # 解码器通常也需要是多层的
        self.layers = nn.ModuleList([
             DCGRUCell(self.out_features if i == 0 else self.hidden_features,
                      self.hidden_features,
                      config['model']['num_diffusion_steps'],
                      adj_matrices)
            for i in range(self.num_layers)
        ])
        
        self.output_layer = nn.Linear(self.hidden_features, self.out_features)

    def forward(self, h_init, horizon, targets, teacher_forcing_ratio):
        # 假设解码器和编码器层数相同，h_init 是最后一层的状态
        # 实践中可能需要处理所有层的状态，这里简化
        h_t = h_init
        batch_size, num_nodes, _ = h_t.shape
        decoder_input = torch.zeros(batch_size, num_nodes, self.out_features, device=h_init.device)
        outputs = []

        for t in range(horizon):
            # 简化：只通过单层解码器进行迭代，更完整的实现会遍历所有层
            h_t = self.layers[0](decoder_input, h_t)
            output_t = self.output_layer(h_t)
            outputs.append(output_t)
            
            # 使用 random.random()
            use_teacher_forcing = (targets is not None) and (random.random() < teacher_forcing_ratio)
            decoder_input = targets[:, t, :, :] if use_teacher_forcing else output_t
            
        return torch.stack(outputs, dim=1)


class DCRNN(nn.Module):
    """完整的DCRNN模型"""
    def __init__(self, config, adj_matrices):
        super().__init__()
        self.encoder = Encoder(config, adj_matrices)
        self.decoder = Decoder(config, adj_matrices)
        self.horizon = config['train']['horizon']
        self.teacher_forcing_ratio = config['train']['teacher_forcing_ratio']

    def forward(self, x_seq, targets=None):
        context = self.encoder(x_seq)
        predictions = self.decoder(context, self.horizon, targets, self.teacher_forcing_ratio)
        return predictions