#!/usr/bin/env python3
"""
融合 SDP/DP 的简短实践示例。
实际链路：tn → 编码器(transformers/zhmodel) → audio_features, text_features → 本模块 → sdp_embedding, dp_durations
本脚本用随机张量模拟「编码器输出」，演示如何调用融合模块。
"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "core"))

import torch
from bertvits2_fused_module import FusedSdpDpModule

# 1) 加载融合模块（只需一次）
module = FusedSdpDpModule(device="cuda")

# 2) 准备输入：实际项目中来自「tn → 编码器」；这里用随机张量模拟
#    形状必须为 [batch, seq_len, feat_dim]，例如 [1, 128, 256]
batch, seq_len, feat_dim = 1, 128, 256
audio_features = torch.randn(batch, seq_len, feat_dim, device="cuda")
text_features = torch.randn(batch, seq_len, feat_dim, device="cuda")

# 3) 一次调用得到说话人嵌入和时长
sdp_embedding, dp_durations = module.forward(audio_features, text_features)

print("SDP shape:", sdp_embedding.shape)   # [1, 256]  说话人嵌入
print("DP shape:", dp_durations.shape)   # [1, 128]  每帧时长
# 后续：sdp_embedding、dp_durations 交给解码器/声学模型使用

# ---------- 在主项目里替换原 SDP/DP 时，通常只需改这几行 ----------
# 原：sdp = sdp_model.run(audio_features); dp = dp_model.run(text_features, x_mask)
# 现：
#   sdp_embedding, dp_durations = fused_module.forward(audio_features, text_features, x_mask)
