#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cross_attention_demo.py
=======================
一个最小但完整的 Cross-Attention 示例：
- 文本 token 向量  →  作为 Query
- 图像 patch 向量 →  作为 Key / Value
依赖：
    pip install torch==2.2.2
"""

import torch
import torch.nn as nn
from torch import Tensor

class CrossAttention(nn.Module):
    """
    通用 Cross-Attention 块：
    Q 来自源序列  (src_len,  B, d_model)
    K/V 来自目标序列 (tgt_len,  B, d_model)
    返回：
        attended   (src_len,  B, d_model)
        attn_probs (B,          src_len, tgt_len)
    """
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, q: Tensor, kv: Tensor, attn_mask=None) -> tuple[Tensor, Tensor]:
        # q: (S_q, B, D)   kv: (S_kv, B, D)
        # MultiheadAttention 要求 (seq_len, batch, embed_dim)
        attn_out, attn_probs = self.mha(query=q, key=kv, value=kv, attn_mask=attn_mask)
        x = self.norm(q + attn_out)      # 残差 + LN
        x = self.norm(x + self.ffn(x))   # Feed-Forward + 残差 + LN
        return x, attn_probs             # x shape = (S_q, B, D)

# ============ Demo ============ #
if __name__ == "__main__":
    torch.manual_seed(0)

    B        = 2      # batch size
    T_txt    = 16     # 文本 token 数
    P_img    = 49     # 图像 patch 数 (e.g., 7×7)
    D_model  = 512

    # 随机生成“文本”与“图像”嵌入，实际应用时换成 BERT/Vision Transformer 等模型输出
    txt_feats = torch.randn(T_txt, B, D_model)   # (T_txt, B, D_model)
    img_feats = torch.randn(P_img, B, D_model)   # (P_img, B, D_model)

    # 1) 文本→图像：让文本 token 聚焦对应图像 patch
    cross_attn_txt2img = CrossAttention(D_model)
    txt_ctx, prob_txt2img = cross_attn_txt2img(q=txt_feats, kv=img_feats)
    print(f"[Text→Image] 输出形状: {txt_ctx.shape}, 注意力形状: {prob_txt2img.shape}")

    # 2) 图像→文本：让图像 patch 聚焦描述它们的文本 token
    cross_attn_img2txt = CrossAttention(D_model)
    img_ctx, prob_img2txt = cross_attn_img2txt(q=img_feats, kv=txt_feats)
    print(f"[Image→Text] 输出形状: {img_ctx.shape}, 注意力形状: {prob_img2txt.shape}")

    # 你可以把 txt_ctx / img_ctx 进一步喂入后续 Transformer 层或多模态融合模块
