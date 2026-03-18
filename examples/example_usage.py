#!/usr/bin/env python3
"""融合模块示例：需在包根目录执行，并将 core 加入 PYTHONPATH。"""

import sys
import os
# 包根目录的上一级或当前目录
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "core"))

import numpy as np
from fused_sdp_dp_model import FusedSdpDpModel
from bertvits2_fused_module import FusedSdpDpModule

def main():
    print("Fused SDP/DP Example")
    print("=" * 50)
    model = FusedSdpDpModel()
    batch_size, seq_len, feat_dim = 4, 128, 256
    audio = np.random.randn(batch_size, seq_len, feat_dim).astype(np.float32)
    text = np.random.randn(batch_size, seq_len, feat_dim).astype(np.float32)
    sdp, dp = model.fused_forward(audio, text, return_torch=False)
    print(f"SDP shape: {sdp.shape}, DP shape: {dp.shape}")
    print("Sample SDP[0,:3]:", sdp[0, :3])
    print("Sample DP[0,:5]:", dp[0, :5])
    print()
    print("FusedSdpDpModule (BertVITS2 兼容):")
    module = FusedSdpDpModule(device="cuda")
    import torch
    a_t = torch.from_numpy(audio)
    t_t = torch.from_numpy(text)
    sdp_t, dp_t = module.forward(a_t, t_t)
    # 预期: SDP [batch, feat_dim]=[4,256], DP [batch, seq_len]=[4,128]
    print(f"SDP shape: {sdp_t.shape} (expected [batch, {feat_dim}]), DP shape: {dp_t.shape} (expected [batch, {seq_len}])")
    print("OK")

if __name__ == "__main__":
    main()
