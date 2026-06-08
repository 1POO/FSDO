import torch
from torch import nn
import torch.nn.functional as F


class LocalLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.temperature = temperature  # 可改为nn.Parameter实现可学习
        self.alpha = alpha  # 全局损失权重

    def forward(self, vid_feat, txt_feat, pos_mask, src_vid_mask=None, src_txt_mask=None):
        # ------ 局部（视频帧和整段文本）对齐损失计算（同CTC_Loss） ------
        # ------ 细粒度 (视频和句子中的每个token) 对齐损失
        # ------ 局部对齐损失计算（无参数版） ------
        # 归一化
        vid_feat_norm = F.normalize(vid_feat, p=2, dim=-1)
        txt_feat_norm = F.normalize(txt_feat, p=2, dim=-1)

        # 局部相似度矩阵 (bs, t, n)
        local_sim_matrix = torch.einsum('btd,bnd->btn', vid_feat_norm, txt_feat_norm)

        # 处理文本mask
        if src_txt_mask is not None:
            local_sim_matrix = local_sim_matrix.masked_fill(~src_txt_mask.bool().unsqueeze(1), -1e4)

        # 取每个帧的最大相似度 (bs, t)
        max_sim, _ = local_sim_matrix.max(dim=2)

        # 应用视频mask
        if src_vid_mask is not None:
            max_sim = max_sim * src_vid_mask

        # 局部损失
        local_logits = max_sim / self.temperature
        loss_local = F.binary_cross_entropy_with_logits(local_logits, pos_mask.float())


        return loss_local

