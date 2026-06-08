import torch.nn as nn
import torch

class fuser(nn.Module):
    def __init__(self, moduledim, sequence_length=256):
        super().__init__()
        # Local Attention: (moduledim → 75 → moduledim)
        self.local_att = nn.Sequential(
            nn.Conv1d(moduledim, 75, 1),
            nn.BatchNorm1d(75),
            nn.ReLU(inplace=True),
            nn.Conv1d(75, moduledim, 1),
            nn.BatchNorm1d(moduledim),
        )
        # Global Attention（不再恢复序列长度）
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化（序列长度→1）
            nn.Conv1d(moduledim, 75, 1),
            nn.BatchNorm1d(75),
            nn.ReLU(inplace=True),
            nn.Conv1d(75, moduledim, 1),
            nn.BatchNorm1d(moduledim),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)  # (B, moduledim, L)
        xg = self.global_att(xa)  # (B, moduledim, 1)
        # 在 fuser_loop 中对齐 xg 的序列长度到 L
        xlg = xl + xg  # 现在xl和xg的序列长度一致（由 fuser_loop 对齐）
        wei = self.sigmoid(xlg)
        return 2 * x * wei + 2 * residual * (1 - wei)

class CrossModalAttention(nn.Module):
    def __init__(self, moduledim):
        super().__init__()
        self.att = nn.MultiheadAttention(moduledim, num_heads=5)

    def forward(self, video, obj):
        attn_output, _ = self.att(video, obj, obj)
        return attn_output

class fuser_loop(nn.Module):
    def __init__(self, input_channels, moduledim, sequence_length=256):
        super().__init__()
        self.channel_align = nn.Conv1d(input_channels, moduledim, 1)
        self.seq_align = nn.AdaptiveAvgPool1d(sequence_length)  # 强制对齐序列长度
        self.fuser = fuser(moduledim, sequence_length)
        self.cross_att = CrossModalAttention(moduledim)

    def forward(self, x, residual):
        # 对齐输入和残差的通道数及序列长度
        x_aligned = self.channel_align(x)  # (B, moduledim, L)
        x_aligned = self.seq_align(x_aligned)  # (B, moduledim, 256)
        
        residual_aligned = self.channel_align(residual)
        residual_aligned = self.seq_align(residual_aligned)  # (B, moduledim, 256)
        
        # 第一次融合
        xo1 = self.fuser(x_aligned, residual_aligned)  # (B, moduledim, 256)
        
        # 第二次融合
        xo2 = self.fuser(xo1, residual_aligned)  # (B, moduledim, 256)
        
        # 跨模态注意力
        xo2 = xo2.permute(0,2,1)
        residual_aligned = residual_aligned.permute(0,2,1)
        xo2 = self.cross_att(xo2, residual_aligned)  # (B, moduledim, 256)
        
        return xo2

# # 测试代码
# x = torch.randn(32, 75, 256)   # 输入特征 (B, 75, 256)
# y = torch.randn(32, 75, 256)   # 残差特征 (B, 75, 256)
#
# model = fuser_loop(input_channels=75, moduledim=75, sequence_length=256)
# output = model(x, y)
# print(output.shape)  # 输出: torch.Size([32, 75, 256])