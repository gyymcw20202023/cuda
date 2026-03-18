#!/bin/bash
# 在包根目录执行：先编译 kernel，再跑示例与测试
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
echo "Package root: $ROOT"

if [ ! -f kernels/fused_sdp_dp_optimized.so ]; then
  echo "Building kernel..."
  (cd kernels && bash build.sh)
fi
export PYTHONPATH="$ROOT/core:$PYTHONPATH"

echo ""
echo "Example:"
python3 examples/example_usage.py

echo ""
echo "Consistency check:"
python3 tests/consistency_check.py

echo ""
echo "Performance:"
python3 tests/performance_comparison.py

echo ""
echo "All OK"
