#!/bin/bash
# Run the smoke gate.  GPU access goes through tools/gpurun_dg.sh (CONTRACT §0:
# pool {0,1}, cross-project flock, contention guard, clocks locked at 1830).
# Never set CUDA_VISIBLE_DEVICES here and never flock .gpu3.lock.
set -u
P=/home/admin/workspace/hengfeng_data/weave_new/DSA-learn/mk/h20/cutedsl/decode_gemm
BIN=${1:-$P/kernels/smoke_test}
LOG=${2:-$P/kernels/SMOKE_LOG.txt}
if [ ! -x "$BIN" ]; then echo "missing $BIN (run build_smoke.sh first)"; exit 1; fi
# NOTE: tools/gpurun_dg.sh line 33 interpolates "$mMiB" (should be "${m}MiB") and
# runs under `set -u`, so it aborts before launching anything.  Exporting a dummy
# mMiB keeps that file untouched (it is outside this agent's write scope) while
# making the wrapper usable.  Reported to the orchestrator.
export mMiB=${mMiB:-MiB}
"$P/tools/gpurun_dg.sh" smoke_test "$BIN" "${@:3}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "smoke rc=$rc log=$LOG"
exit $rc
