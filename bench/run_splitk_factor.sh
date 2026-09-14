#!/bin/bash
# SPDX-License-Identifier: MIT
# bench/run_splitk_factor.sh —— S 路 split-K 的甜区实测（splitk-S agent 2026-09-13）
#
#   bash bench/run_splitk_factor.sh                  # qwen36 全 M 阶梯 + kimi_k3 M=1 对照
#   ONLY=qwen|kimi bash bench/run_splitk_factor.sh   # 只跑一半
#   FACTORS=1,2,4 MS=1,8 PROTOCOLS=PDL,ISO REPS=3 bash bench/run_splitk_factor.sh
#
# 产物（主 CSV 的 16 列 + splitk_factor）：
#   results/splitk_factor_qwen.csv      qwen36  M=1,2,4,8,16,32  x S∈{1,2,4,8}
#   results/splitk_factor_kimi_k3.csv   kimi_k3 M=1              x S∈{1,4}（负对照）
# 每个 M 扫两条基准几何（bench_decode --splitk-factor-sweep）：
#   base0 = 该形状该 M 的现役 canonical（qwen 是 tune override BM56/WG4/REG96/STG3）
#   base1 = 形状无关 canonical（qwen BM32/WG2/REG80/STG3，README §6.2 的 S6 几何）
# 缺档（S=8 在 WGsPerTile=4 上过不了整除性）会记进 skipped，不算失败。
#
# GPU 一律经 tools/gpurun_dg.sh（run_sweep.sh 已封装 flock/守卫/2s 看门狗/锁频 1830/
# rc=75 自动重排 + --resume 补缺口）。COLLECT 指到 /tmp：本实验的 canonical 主 CSV
# 是空的，不要混进 results/ 顶层给 plot agent 吃。
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=$(cd "$HERE/.." && pwd)
BIN=${BIN:-$HERE/bench_decode_sk}
FACTORS=${FACTORS:-1,2,4,8}
KIMI_FACTORS=${KIMI_FACTORS:-1,4}
MS=${MS:-1,2,4,8,16,32}
KIMI_MS=${KIMI_MS:-1}
PROTOCOLS=${PROTOCOLS:-PDL}
REPS=${REPS:-3}
WCG=${WCG:-6}          # 冷权重缓存 GiB：2 条几何 x 3 个 M_TILE，每组 >=512MiB
SEMR=${SEMR:-16}       # split-K semaphore 轮转槽数（1 = 关，会重现跨 launch 污染）
ONLY=${ONLY:-both}
MAX_RETRY=${MAX_RETRY:-20}
LOG=${LOG:-/tmp/splitk_factor.log}
exec > >(tee -a "$LOG") 2>&1
step() { echo "[skf] $(date -u +%FT%TZ) $*"; }

[ -x "$BIN" ] || { step "BIN=$BIN 不存在 -> make BIN=bench_decode_sk OBJDIR=build_sk TUNE=0 -j12"; exit 1; }
step "bin=$BIN ($(stat -c %y "$BIN" | cut -d. -f1)) factors=$FACTORS ms=$MS proto=$PROTOCOLS reps=$REPS sem_rotate=$SEMR"

run_one() {   # <model> <ms> <factors> <out csv> <batch>
  local model=$1 ms=$2 fac=$3 out=$4 batch=$5
  step "$model M=$ms S=$fac -> $out"
  BATCH="$batch" RESULTS="$P/results/$batch" COLLECT=/tmp/skf_collect \
    RESUME=1 MAX_RETRY="$MAX_RETRY" BIN="$BIN" \
    bash "$HERE/run_sweep.sh" --models "$model" --steps 6 --ms "$ms" \
      --protocols "$PROTOCOLS" --reps "$REPS" --weight-cache-gb "$WCG" \
      --splitk-factor-sweep --splitk-factors "$fac" --factor-out "$out" \
      --sem-rotate "$SEMR"
  local rc=$?
  local n=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))
  step "$model rc=$rc rows=$n"
  return $rc
}

rc_all=0
if [ "$ONLY" = both ] || [ "$ONLY" = qwen ]; then
  run_one qwen36 "$MS" "$FACTORS" "$P/results/splitk_factor_qwen.csv" skf_qwen || rc_all=1
fi
if [ "$ONLY" = both ] || [ "$ONLY" = kimi ]; then
  run_one kimi_k3 "$KIMI_MS" "$KIMI_FACTORS" "$P/results/splitk_factor_kimi_k3.csv" skf_kimi || rc_all=1
fi
step "ALL DONE rc=$rc_all  qwen=$(( $(wc -l < "$P/results/splitk_factor_qwen.csv" 2>/dev/null || echo 1) - 1 )) rows  kimi=$(( $(wc -l < "$P/results/splitk_factor_kimi_k3.csv" 2>/dev/null || echo 1) - 1 )) rows"
exit $rc_all
