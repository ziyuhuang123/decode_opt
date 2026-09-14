#!/usr/bin/env bash
# build_peak.sh — 编译 tools/peak_bw_probe.cu
# nvcc 参数严格按 CONTRACT.md §0 的编译命令模板。
set -euo pipefail

W=/home/admin/workspace/hengfeng_data/weave_new
P=$W/DSA-learn/mk/h20/cutedsl/decode_gemm
CUTLASS=$W/weave_v2/deepgemm/third-party/cutlass/include
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}

SRC=$P/tools/peak_bw_probe.cu
OUT=${1:-$P/tools/peak_bw_probe}

echo "[build] nvcc $("$NVCC" --version | grep release | sed 's/^ *//')"
echo "[build] src = $SRC"
echo "[build] out = $OUT"
echo "[build] cutlass = $CUTLASS"

# CONTRACT §0 模板：
#   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -DNDEBUG \
#     -I$P/kernels -I$CUTLASS <src> -lcuda -lcublasLt -o <out>
"$NVCC" -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -DNDEBUG \
  -I"$P/kernels" -I"$CUTLASS" \
  "$SRC" -lcuda -lcublasLt -o "$OUT"

echo "[build] OK -> $OUT"
ls -la "$OUT"
