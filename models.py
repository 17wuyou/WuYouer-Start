# models.py (Part 1)

class DCGRUCell(nn.Module):
    """扩散卷积门控循环单元"""
    def __init__(self, in_features, hidden_features, num_diffusion_steps, adj_matrices):
        super().__init__()
        self.hidden_features = hidden_features

        # 定义计算重置门 r 和更新门 u 的扩散卷积
        self.gate_conv = DiffusionConv(in_features + hidden_features, hidden_features * 2, num_diffusion_steps, adj_matrices)

        # 定义计算候选状态 C 的扩散卷积
        self.candidate_conv = DiffusionConv(in_features + hidden_features, hidden_features, num_diffusion_steps, adj_matrices)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t (torch.Tensor): 当前时刻的输入 (Batch, Num_nodes, In_features)
            h_prev (torch.Tensor): 上一时刻的隐藏状态 (Batch, Num_nodes, Hidden_features)

        Returns:
            torch.Tensor: 当前时刻的隐藏状态 (Batch, Num_nodes, Hidden_features)
        """
        # 拼接输入和上一时刻的隐藏状态
        combined = torch.cat([x_t, h_prev], dim=-1)

        # 计算门控信号
        gates = self.gate_conv(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_features, dim=-1)

        r = torch.sigmoid(reset_gate)
        u = torch.sigmoid(update_gate)

        # 计算候选隐藏状态
        combined_candidate = torch.cat([x_t, r * h_prev], dim=-1)
        C = torch.tanh(self.candidate_conv(combined_candidate))

        # 计算并返回当前时刻的隐藏状态
        h_new = u * h_prev + (1 - u) * C
        return h_new

# models.py (Part 2)

class Encoder(nn.Module):
    def __init__(self, in_features, hidden_features, num_layers, num_diffusion_steps, adj_matrices):
        super().__init__()
        # 可以堆叠多层DCGRU
        self.layers = nn.ModuleList([
            DCGRUCell(in_features if i == 0 else hidden_features, hidden_features, num_diffusion_steps, adj_matrices)
            for i in range(num_layers)
        ])

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq (torch.Tensor): 输入序列 (Batch, Seq_len, Num_nodes, In_features)

        Returns:
            torch.Tensor: 编码器最后的隐藏状态 (Num_layers, Batch, Num_nodes, Hidden_features)
        """
        batch_size, _, num_nodes, _ = x_seq.shape
        hidden_states = [torch.zeros(batch_size, num_nodes, hidden_features) for _ in self.layers]

        for t in range(x_seq.shape[1]): # 遍历时间序列
            x_t = x_seq[:, t, :, :]
            for i, layer in enumerate(self.layers):
                x_t = hidden_states[i] = layer(x_t, hidden_states[i])
        
        return torch.stack(hidden_states, dim=0)


class Decoder(nn.Module):
    def __init__(self, out_features, hidden_features, num_layers, num_diffusion_steps, adj_matrices):
        super().__init__()
        self.out_features = out_features
        self.layers = nn.ModuleList([
            DCGRUCell(out_features if i == 0 else hidden_features, hidden_features, num_diffusion_steps, adj_matrices)
            for i in range(num_layers)
        ])
        # 将最后的隐藏状态映射回预测的特征维度
        self.output_layer = nn.Linear(hidden_features, out_features)

    def forward(self, h_init: torch.Tensor, horizon: int, targets: torch.Tensor = None, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Args:
            h_init (torch.Tensor): 编码器的初始隐藏状态 (Num_layers, Batch, Num_nodes, Hidden_features)
            horizon (int): 需要预测的时间步数 T
            targets (torch.Tensor, optional): 真实值，用于计划采样
            teacher_forcing_ratio (float): 计划采样中，使用真实值的概率

        Returns:
            torch.Tensor: 预测的序列 (Batch, Horizon, Num_nodes, Out_features)
        """
        hidden_states = [h_init[i] for i in range(h_init.shape[0])]
        
        batch_size, num_nodes, _ = hidden_states[0].shape
        # 解码器的第一个输入通常是0或者一个特殊标记
        decoder_input = torch.zeros(batch_size, num_nodes, self.out_features)
        
        outputs = []
        for t in range(horizon):
            # 更新所有层的隐藏状态
            x_t = decoder_input
            for i, layer in enumerate(self.layers):
                 x_t = hidden_states[i] = layer(x_t, hidden_states[i])
            
            # 计算输出
            output_t = self.output_layer(x_t)
            outputs.append(output_t)

            # --- 计划采样 ---
            use_teacher_forcing = (targets is not None) and (torch.rand(1) < teacher_forcing_ratio)
            if use_teacher_forcing:
                decoder_input = targets[:, t, :, :] # 使用真实值
            else:
                decoder_input = output_t # 使用模型自己的预测

        return torch.stack(outputs, dim=1)


class DCRNN(nn.Module):
    """完整的DCRNN模型"""
    def __init__(self, config):
        super().__init__()
        # 从utils中预计算
        self.adj_matrices = calculate_diffusion_matrix(...) 
        
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        self.teacher_forcing_ratio = config['teacher_forcing_ratio']

    def forward(self, x_seq, targets=None):
        context_vector = self.encoder(x_seq)
        predictions = self.decoder(context_vector, horizon=12, targets=targets, teacher_forcing_ratio=self.teacher_forcing_ratio)
        return predictions