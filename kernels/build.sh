#!/bin/bash
# 编译融合 CUDA 库，生成 fused_sdp_dp_optimized.so
set -e
cd "$(dirname "$0")"
ARCH="${1:-sm_75}"
echo "Building with -arch=$ARCH"
nvcc -shared -Xcompiler -fPIC -O3 --use_fast_math -arch="$ARCH" \
  -o fused_sdp_dp_optimized.so fused_sdp_dp_optimized.cu
echo "Done: $(pwd)/fused_sdp_dp_optimized.so"
