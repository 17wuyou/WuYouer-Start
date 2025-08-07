# models.py

import torch
import torch.nn as nn
import random

class DiffusionConv(nn.Module):
    def __init__(self, in_features, out_features, num_diffusion_steps, adj_matrices):
        super().__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.adj_matrices = adj_matrices
        self.fc = nn.Linear(in_features * 2 * num_diffusion_steps, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        supports = []
        for P_k in self.adj_matrices:
            P_k = P_k.to(x.device)
            support = torch.einsum('nn, bni -> bni', P_k, x)
            supports.append(support)
        x_g = torch.cat(supports, dim=-1)
        return self.fc(x_g)

class DCGRUCell(nn.Module):
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
        hidden_states = [torch.zeros(batch_size, num_nodes, self.hidden_features, device=x_seq.device) for _ in range(self.num_layers)]

        for t in range(x_seq.shape[1]):
            x_input = x_seq[:, t, :, :]
            for i, layer in enumerate(self.layers):
                h_prev = hidden_states[i]
                h_new = layer(x_input, h_prev)
                hidden_states[i] = h_new
                x_input = h_new
        
        # 返回最后一层的最终隐藏状态作为上下文向量
        return hidden_states[-1]

class Decoder(nn.Module):
    def __init__(self, config, adj_matrices):
        super().__init__()
        self.out_features = config['model']['out_features']
        self.hidden_features = config['model']['hidden_features']
        self.num_layers = config['model']['num_layers']
        
        self.layers = nn.ModuleList([
             DCGRUCell(self.out_features if i == 0 else self.hidden_features,
                       self.hidden_features,
                       config['model']['num_diffusion_steps'],
                       adj_matrices)
            for i in range(self.num_layers)
        ])
        
        self.output_layer = nn.Linear(self.hidden_features, self.out_features)

    def forward(self, h_init, horizon, targets, teacher_forcing_ratio):
        # 初始化所有解码器层的隐藏状态
        hidden_states = [h_init for _ in range(self.num_layers)]
        
        batch_size, num_nodes, _ = h_init.shape
        decoder_input = torch.zeros(batch_size, num_nodes, self.out_features, device=h_init.device)
        outputs = []

        for t in range(horizon):
            x_input = decoder_input
            for i, layer in enumerate(self.layers):
                h_prev = hidden_states[i]
                h_new = layer(x_input, h_prev)
                hidden_states[i] = h_new
                x_input = h_new

            # 最后一层的输出用于最终预测
            output_t = self.output_layer(hidden_states[-1])
            outputs.append(output_t)
            
            use_teacher_forcing = (targets is not None) and (random.random() < teacher_forcing_ratio)
            decoder_input = targets[:, t, :, :] if use_teacher_forcing else output_t
            
        return torch.stack(outputs, dim=1)

class DCRNN(nn.Module):
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