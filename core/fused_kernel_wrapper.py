#!/usr/bin/env python3
"""CUDA fused kernel Python wrapper for SDP/DP fusion."""

import numpy as np
import ctypes
import subprocess
import os

class FusedKernelWrapper:
    """Python interface for fused CUDA kernel."""

    def __init__(self, lib_path=None):
        self.lib_path = lib_path
        self.lib = None
        self._compile_and_load()

    def _compile_and_load(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        current_dir = os.getcwd()

        possible_lib_paths = [
            self.lib_path,
            os.path.join(parent_dir, "kernels", "fused_sdp_dp_optimized.so"),
            os.path.join(base_dir, "fused_sdp_dp_optimized.so"),
            os.path.join(base_dir, "fused_kernel.so"),
            os.path.join(current_dir, "fused_sdp_dp_optimized.so"),
            os.path.join(current_dir, "fused_kernel.so"),
        ]

        for p in possible_lib_paths:
            if p and os.path.exists(p):
                try:
                    self.lib = ctypes.CDLL(p)
                    self.lib_path = p
                    print(f"✓ CUDA library loaded: {p}")
                    return
                except Exception:
                    continue

        cu_candidates = [
            os.path.join(parent_dir, "kernels", "fused_sdp_dp_optimized.cu"),
            os.path.join(base_dir, "fused_sdp_dp_optimized.cu"),
            os.path.join(current_dir, "fused_sdp_dp_optimized.cu"),
            os.path.join(parent_dir, "kernels", "fused_sdp_dp.cu"),
            os.path.join(base_dir, "fused_sdp_dp.cu"),
        ]
        cu_file = None
        for path in cu_candidates:
            if os.path.exists(path):
                cu_file = path
                break

        if not cu_file:
            print("ERROR: No CUDA source (.cu) found. Put fused_sdp_dp_optimized.cu in kernels/ and run kernels/build.sh")
            self.lib = None
            return

        cu_dir = os.path.dirname(cu_file)
        base_cu = os.path.basename(cu_file)
        so_name = "fused_sdp_dp_optimized.so" if "optimized" in base_cu else "fused_kernel.so"
        so_file = os.path.join(cu_dir, so_name)

        try:
            print(f"Compiling: {cu_file}")
            cmd = f"cd {cu_dir} && nvcc -shared -Xcompiler -fPIC -O3 --use_fast_math -arch=sm_75 -o {so_name} {base_cu}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            if os.path.exists(so_file):
                self.lib = ctypes.CDLL(so_file)
                self.lib_path = so_file
                print(f"✓ Compiled and loaded: {so_file}")
            else:
                self.lib = None
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e.stderr or e}")
            self.lib = None

    def fused_forward(self, audio_features, text_features):
        if self.lib is None:
            raise RuntimeError("CUDA library not loaded")
        batch_size, seq_len, feat_dim = audio_features.shape
        embed_dim = feat_dim
        audio_features = audio_features.astype(np.float32)
        text_features = text_features.astype(np.float32)
        sdp_embedding = np.zeros((batch_size, embed_dim), dtype=np.float32)
        dp_durations = np.zeros((batch_size, seq_len), dtype=np.float32)
        f = self.lib.fused_forward
        f.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        f.restype = None
        f(
            audio_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            text_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sdp_embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dp_durations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size, seq_len, feat_dim, embed_dim,
        )
        return sdp_embedding, dp_durations
