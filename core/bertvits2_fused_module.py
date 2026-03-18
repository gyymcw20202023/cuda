#!/usr/bin/env python3
"""BertVITS2 融合模块：与 vortex.models.bertvits2 兼容的 DpModel / SdpModel / FusedSdpDpModule。"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
import os

try:
    from fused_sdp_dp_model import FusedSdpDpModel
    FUSED_AVAILABLE = True
except ImportError:
    FUSED_AVAILABLE = False


class FusedDpModel:
    """融合 DP 模型，替换 DpModel。run(x, x_mask) 输入 x: [batch, channels, seq_len]。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        speaker_path: Optional[str] = None,
        device: str = "cuda",
        use_cuda_graph: bool = False,
        lib_path: Optional[str] = None,
        **kwargs
    ):
        if not FUSED_AVAILABLE:
            raise ImportError("FusedSdpDpModel not available. Add core/ to PYTHONPATH and build kernels.")
        self.device = device
        self.fused_model = FusedSdpDpModel(lib_path=lib_path, device=device)
        self.speaker_embedding = np.load(speaker_path) if speaker_path and os.path.exists(speaker_path) else None
        print("✓ FusedDpModel initialized")

    def run(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        g: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        batch, channels, seq_len = x.shape
        x_t = x.permute(0, 2, 1).contiguous()
        if x_mask is not None:
            mask = x_mask.squeeze(1) if x_mask.dim() == 3 else x_mask
            x_t = x_t * mask.unsqueeze(-1)
        audio_np = torch.zeros_like(x_t).cpu().numpy().astype(np.float32)
        text_np = x_t.cpu().numpy().astype(np.float32)
        _, dp_out = self.fused_model.fused_forward(audio_np, text_np, return_torch=False)
        logw = torch.from_numpy(dp_out).to(self.device)
        if x_mask is not None:
            mask = x_mask.squeeze(1) if x_mask.dim() == 3 else x_mask
            if isinstance(mask, torch.Tensor):
                mask = mask.to(self.device)
            logw = logw * mask
        return logw


class FusedSdpModel:
    """融合 SDP 模型，替换 SdpModel。run(audio_features) 输入 [batch, seq_len, feat_dim] 或 [batch, C, H, W]。"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", lib_path: Optional[str] = None, **kwargs):
        if not FUSED_AVAILABLE:
            raise ImportError("FusedSdpDpModel not available. Add core/ to PYTHONPATH and build kernels.")
        self.device = device
        self.fused_model = FusedSdpDpModel(lib_path=lib_path, device=device)
        print("✓ FusedSdpModel initialized")

    def run(self, audio_features: torch.Tensor, **kwargs) -> torch.Tensor:
        orig = audio_features.shape
        if len(orig) == 3:
            audio_reshaped = audio_features
        elif len(orig) == 4:
            b, c, h, w = orig
            audio_reshaped = audio_features.view(b, h * w, c)
        else:
            b = orig[0]
            audio_reshaped = audio_features.view(b, -1, orig[-1])
        batch, seq_len, feat_dim = audio_reshaped.shape
        text_fake = torch.zeros(batch, seq_len, feat_dim, device=audio_reshaped.device)
        audio_np = audio_reshaped.cpu().numpy().astype(np.float32)
        text_np = text_fake.cpu().numpy().astype(np.float32)
        sdp_out, _ = self.fused_model.fused_forward(audio_np, text_np, return_torch=False)
        return torch.from_numpy(sdp_out).to(self.device)


class FusedSdpDpModule:
    """融合 SDP+DP 模块。forward(audio_features, text_features, x_mask=None) 一次得到 sdp_embedding 与 dp_durations。"""

    def __init__(
        self,
        sdp_model_path: Optional[str] = None,
        dp_model_path: Optional[str] = None,
        speaker_path: Optional[str] = None,
        device: str = "cuda",
        lib_path: Optional[str] = None,
        **kwargs
    ):
        if not FUSED_AVAILABLE:
            raise ImportError("FusedSdpDpModel not available. Add core/ to PYTHONPATH and build kernels.")
        self.device = device
        self.fused_model = FusedSdpDpModel(lib_path=lib_path, device=device)
        self.speaker_embedding = np.load(speaker_path) if speaker_path and os.path.exists(speaker_path) else None
        print("✓ FusedSdpDpModule initialized")

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 若为 (batch, feat_dim, seq_len) 则转为 (batch, seq_len, feat_dim)；已是 (batch, seq_len, feat_dim) 则不 permute
        if len(text_features.shape) == 3 and text_features.shape[1] > text_features.shape[2]:
            text_features = text_features.permute(0, 2, 1).contiguous()
        if len(audio_features.shape) != 3:
            batch = audio_features.shape[0]
            audio_features = audio_features.view(batch, -1, audio_features.shape[-1])
        batch_a, seq_a, feat_a = audio_features.shape
        batch_t, seq_t, feat_t = text_features.shape
        min_dim = min(feat_a, feat_t)
        audio_features = audio_features[:, :, :min_dim]
        text_features = text_features[:, :, :min_dim]
        feat_dim = audio_features.shape[-1]
        seq_len = audio_features.shape[1]
        audio_np = audio_features.cpu().numpy().astype(np.float32)
        text_np = text_features.cpu().numpy().astype(np.float32)
        sdp_out, dp_out = self.fused_model.fused_forward(audio_np, text_np, return_torch=False)
        # SDP: [batch, embed_dim], DP: [batch, seq_len]; embed_dim=feat_dim
        assert sdp_out.shape == (audio_features.shape[0], feat_dim), f"SDP shape {sdp_out.shape} vs expected (batch, {feat_dim})"
        assert dp_out.shape == (audio_features.shape[0], seq_len), f"DP shape {dp_out.shape} vs expected (batch, {seq_len})"
        sdp_embedding = torch.from_numpy(sdp_out).to(self.device)
        dp_durations = torch.from_numpy(dp_out).to(self.device)
        if x_mask is not None:
            mask = x_mask.squeeze(1) if x_mask.dim() == 3 else x_mask
            if isinstance(mask, torch.Tensor):
                mask = mask.to(self.device)
            dp_durations = dp_durations * mask
        return sdp_embedding, dp_durations


DpModel = FusedDpModel
SdpModel = FusedSdpModel
