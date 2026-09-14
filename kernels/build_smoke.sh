#!/bin/bash
# Build kernels/smoke_test.cu with the exact command line mandated by CONTRACT §0.
# Big template instantiation list -> compile in the background and poll, as the
# project convention requires.
set -u
P=/home/admin/workspace/hengfeng_data/weave_new/DSA-learn/mk/h20/cutedsl/decode_gemm
CUTLASS=/home/admin/workspace/hengfeng_data/weave_new/weave_v2/deepgemm/third-party/cutlass/include
OUT=${1:-$P/kernels/smoke_test}
LOG=$P/kernels/smoke_build.log
cd "$P" || exit 1
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -DNDEBUG \
  -I"$P/kernels" -I"$CUTLASS" \
  "$P/kernels/smoke_test.cu" -lcuda -lcublasLt -o "$OUT" > "$LOG" 2>&1
rc=$?
echo "build rc=$rc log=$LOG"
if [ $rc -ne 0 ]; then tail -60 "$LOG"; fi
exit $rc
