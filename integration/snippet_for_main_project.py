# -*- coding: utf-8 -*-
"""
复制到主项目（如保存为 sdp_dp_switch.py），用于无缝切换“原版 SDP/DP”与“融合 SDP/DP”。
使用前请将 DELIVERY_ROOT 改为本交付包的实际路径。
"""
import os
import sys

# 改为本交付包实际路径，例如：/data3/test-bert-modifiled/bert-vits2-fused-module-delivery
DELIVERY_ROOT = os.environ.get(
    "BERT_VITS2_FUSED_DELIVERY",
    "/path/to/bert-vits2-fused-module-delivery"
)
DELIVERY_CORE = os.path.join(DELIVERY_ROOT, "core")

_fused_module = None

def use_fused():
    return os.environ.get("USE_FUSED", "1") == "1"

def get_sdp_dp_module():
    """返回 FusedSdpDpModule 单例；仅在 use_fused() 为 True 时调用。"""
    global _fused_module
    if _fused_module is None:
        if DELIVERY_CORE not in sys.path:
            sys.path.insert(0, DELIVERY_CORE)
        from bertvits2_fused_module import FusedSdpDpModule
        _fused_module = FusedSdpDpModule(device="cuda")
    return _fused_module
