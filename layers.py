# layers.py

import torch
import torch.nn as nn

class DiffusionConv(nn.Module):
    """扩散卷积层"""
    def __init__(self, in_features, out_features, num_diffusion_steps, adj_matrices):
        super().__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.in_features = in_features
        self.out_features = out_features
        self.adj_matrices = adj_matrices # 从utils预先计算好的扩散矩阵列表

        # 定义可学习的权重，每个扩散矩阵对应一组权重
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features, 2 * num_diffusion_steps))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入信号 (Batch, Num_nodes, In_features)

        Returns:
            torch.Tensor: 输出 (Batch, Num_nodes, Out_features)
        """
        batch_size, num_nodes, _ = x.shape
        outputs = []

        # 遍历每种扩散/转移矩阵
        for i, P_k in enumerate(self.adj_matrices):
            # P_k 是 (Num_nodes, Num_nodes)
            # x 是 (Batch, Num_nodes, In_features)
            # x_transformed 是 (Batch, Num_nodes, In_features)
            x_transformed = torch.einsum('nn, bni -> bni', P_k, x)

            # weights[:, :, i] 是 (In_features, Out_features)
            # out 是 (Batch, Num_nodes, Out_features)
            out = torch.einsum('bni, io -> bno', x_transformed, self.weights[:, :, i])
            outputs.append(out)

        # 聚合所有扩散的结果
        result = torch.sum(torch.stack(outputs, dim=0), dim=0) # (Batch, Num_nodes, Out_features)
        result = result + self.bias
        return result