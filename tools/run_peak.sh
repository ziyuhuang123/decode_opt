#!/usr/bin/env bash
# =============================================================================
# run_peak.sh — 跑 peak_bw_probe，产出 results/peak_bw.json + results/peak_bw_raw.log
#
# 【GPU 选择：本脚本一概不管，全部交给 tools/gpurun_dg.sh】
#   gpurun_dg.sh peak_probe <cmd...> 负责：
#     1) 在 pool {0,1} 里挑一张通过争用守卫的卡（util<=5% 且 mem<=2000MiB 且无 compute-app pid）
#     2) 抢 _lockbench_20260913/locks/gpu<N>.lock 的 flock（跨项目串行）
#     3) 跑前重申 nvidia-smi -i <N> -lgc 1830,1830
#     4) 设 CUDA_VISIBLE_DEVICES=<N> 后 exec 我们的命令，退出码透传
#   所以这里：不设 CUDA_VISIBLE_DEVICES、不 flock .gpu3.lock、不轮询 GPU2/3。
#   GPU2/3 被 weave_v1 v19_swapab 的 keeper 预留（RELEASE_KEEPER 机制），不许碰/kill/等。
#
# 实际用到的物理卡号由 probe 自己按 PCI bus id 反查，写进 peak_bw.json 的 gpu_id。
# clocks_sm_mhz 记 probe 在 sweep 期间用 nvidia-smi 实测采样到的 clocks.sm（取最小值，保守）。
# =============================================================================
set -uo pipefail

W=/home/admin/workspace/hengfeng_data/weave_new
P=$W/DSA-learn/mk/h20/cutedsl/decode_gemm
GPURUN=$P/tools/gpurun_dg.sh
BIN=$P/tools/peak_bw_probe
OUT=$P/results/peak_bw.json
RAW=$P/results/peak_bw_raw.log
TARGET_ID=${TARGET_ID:-peak_probe}

ROUNDS=${ROUNDS:-7}
WARMUPS=${WARMUPS:-2}
BATCH=${BATCH:-8}
WAIT_QUIET=${WAIT_QUIET:-30}          # probe 上卡后的二次确认：需连续安静多少秒
WAIT_TIMEOUT=${WAIT_TIMEOUT:-1800}
QUIET_MEM_MAX=${QUIET_MEM_MAX:-2000}
QUIET_UTIL_MAX=${QUIET_UTIL_MAX:-5}
QUIET_POLL=${QUIET_POLL:-3}
LOCKCLK_UNUSED=${LOCKCLK:-}           # 保留兼容：锁频由 gpurun_dg.sh 负责，这里不动
PROBE_TIMEOUT=${PROBE_TIMEOUT:-3600}  # 硬超时：万一 kernel 死锁也不能把卡吊死

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds)         ROUNDS=$2; shift 2 ;;
    --warmups)        WARMUPS=$2; shift 2 ;;
    --batch)          BATCH=$2; shift 2 ;;
    --wait-quiet)     WAIT_QUIET=$2; shift 2 ;;
    --wait-timeout)   WAIT_TIMEOUT=$2; shift 2 ;;
    --quiet-mem-max)  QUIET_MEM_MAX=$2; shift 2 ;;
    --quiet-util-max) QUIET_UTIL_MAX=$2; shift 2 ;;
    --quiet-poll)     QUIET_POLL=$2; shift 2 ;;
    --target-id)      TARGET_ID=$2; shift 2 ;;
    --out)            OUT=$2; shift 2 ;;
    --raw)            RAW=$2; shift 2 ;;
    --bin)            BIN=$2; shift 2 ;;
    --)               shift; EXTRA+=("$@"); break ;;
    *)                EXTRA+=("$1"); shift ;;
  esac
done
: "$LOCKCLK_UNUSED"

[[ -x "$GPURUN" ]] || { echo "[run_peak] FATAL: $GPURUN 不存在或不可执行" >&2; exit 3; }
[[ -x "$BIN"    ]] || { echo "[run_peak] FATAL: $BIN 不存在，先跑 tools/build_peak.sh" >&2; exit 3; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$RAW")"

# --clock-locked 1：锁频由 gpurun_dg.sh 负责（它每次跑前重申 nvidia-smi -i <N> -lgc 1830,1830），
# 本脚本不自己 -lgc/-rgc。probe 仍会用 nvidia-smi 实测 clocks.sm 与 throttle reasons 做交叉校验，
# 并以实测值写 clocks_sm_mhz。
PROBE_ARGS=(
  --clock-locked 1
  --rounds "$ROUNDS" --warmups "$WARMUPS" --batch "$BATCH"
  --wait-quiet "$WAIT_QUIET" --wait-timeout "$WAIT_TIMEOUT"
  --quiet-mem-max "$QUIET_MEM_MAX" --quiet-util-max "$QUIET_UTIL_MAX" --quiet-poll "$QUIET_POLL"
  --out "$OUT"
  "${EXTRA[@]+"${EXTRA[@]}"}"
)

{
  echo "#############################################################"
  echo "# peak_bw_probe raw log"
  echo "# utc        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# host       : $(hostname)"
  echo "# launcher   : tools/gpurun_dg.sh $TARGET_ID  (pool={0,1}, 跨项目 flock + 争用守卫 + -lgc 1830)"
  echo "# probe args : ${PROBE_ARGS[*]}"
  echo "# 说明       : 物理卡号(gpu_id)与 clocks.sm 由 probe 自己实测并写进 peak_bw.json"
  echo "#              GPU2/3 未被本脚本触碰（weave_v1 keeper 预留，不许碰/kill/等）"
  echo "#############################################################"
  echo
} | tee -a "$RAW"

# gpurun_dg.sh 选卡 + flock + 锁频 + 设 CUDA_VISIBLE_DEVICES，然后 exec probe。
# 硬超时兜底，退出码透传。
timeout --kill-after=30 "$PROBE_TIMEOUT" \
  "$GPURUN" "$TARGET_ID" "$BIN" "${PROBE_ARGS[@]}" > >(tee -a "$RAW") 2>&1
RC=$?

{
  echo
  echo "# launcher exit code = $RC   (0=ok, 4=probe 拒绝测量/卡被占, 124=gpurun_dg 等卡超时, 其它=异常)"
  echo "# utc end            = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# json               = $OUT"
} | tee -a "$RAW"

echo "[run_peak] done rc=$RC  json=$OUT  raw=$RAW"
exit "$RC"
