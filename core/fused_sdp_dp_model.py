#!/usr/bin/env python3
"""BertVITS2 融合 SDP/DP 模型，兼容原有 ONNX 接口。"""

import numpy as np
import torch
from typing import Union, Tuple
from fused_kernel_wrapper import FusedKernelWrapper


class FusedSdpDpModel:
    """融合 SDP/DP 模型。fused_forward() 同时计算 SDP+DP；sdp_forward/dp_forward 可单独调用。"""

    def __init__(self, lib_path: str = None, device: str = "cuda"):
        self.device = device
        self.wrapper = FusedKernelWrapper(lib_path=lib_path)
        if self.wrapper.lib is None:
            raise RuntimeError("Failed to load CUDA library. Build kernels: cd kernels && bash build.sh")
        print(f"✓ FusedSdpDpModel initialized: {self.wrapper.lib_path}")

    def _to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    def _to_torch(self, x: np.ndarray, device: str = None) -> torch.Tensor:
        return torch.from_numpy(x).to(device or self.device)

    def fused_forward(
        self,
        audio_features: Union[np.ndarray, torch.Tensor],
        text_features: Union[np.ndarray, torch.Tensor],
        return_torch: bool = True,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        audio_np = self._to_numpy(audio_features)
        text_np = self._to_numpy(text_features)
        sdp_out, dp_out = self.wrapper.fused_forward(audio_np, text_np)
        if return_torch:
            sdp_out, dp_out = self._to_torch(sdp_out), self._to_torch(dp_out)
        return sdp_out, dp_out

    def sdp_forward(
        self, audio_features: Union[np.ndarray, torch.Tensor], return_torch: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        audio_np = self._to_numpy(audio_features)
        bs, seq_len, feat_dim = audio_np.shape
        text_np = np.zeros((bs, seq_len, feat_dim), dtype=np.float32)
        sdp_out, _ = self.wrapper.fused_forward(audio_np, text_np)
        return self._to_torch(sdp_out) if return_torch else sdp_out

    def dp_forward(
        self, text_features: Union[np.ndarray, torch.Tensor], return_torch: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        text_np = self._to_numpy(text_features)
        bs, seq_len, feat_dim = text_np.shape
        audio_np = np.zeros((bs, seq_len, feat_dim), dtype=np.float32)
        _, dp_out = self.wrapper.fused_forward(audio_np, text_np)
        return self._to_torch(dp_out) if return_torch else dp_out

    def run(
        self,
        audio_features: Union[np.ndarray, torch.Tensor] = None,
        text_features: Union[np.ndarray, torch.Tensor] = None,
        return_torch: bool = True,
    ):
        if audio_features is not None and text_features is not None:
            return self.fused_forward(audio_features, text_features, return_torch)
        if audio_features is not None:
            return self.sdp_forward(audio_features, return_torch)
        if text_features is not None:
            return self.dp_forward(text_features, return_torch)
        raise ValueError("Provide at least audio_features or text_features")

    def benchmark(self, batch_size=4, seq_len=128, feat_dim=256, num_iterations=100, warmup=10):
        import time
        audio = np.random.randn(batch_size, seq_len, feat_dim).astype(np.float32)
        text = np.random.randn(batch_size, seq_len, feat_dim).astype(np.float32)
        for _ in range(warmup):
            _ = self.wrapper.fused_forward(audio, text)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = self.wrapper.fused_forward(audio, text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "throughput_samples_per_sec": batch_size * 1000 / np.mean(times),
        }


if __name__ == "__main__":
    model = FusedSdpDpModel()
    a = np.random.randn(2, 128, 256).astype(np.float32)
    b = np.random.randn(2, 128, 256).astype(np.float32)
    sdp, dp = model.fused_forward(a, b, return_torch=False)
    print("SDP:", sdp.shape, "DP:", dp.shape)
    print("benchmark:", model.benchmark(batch_size=4))
