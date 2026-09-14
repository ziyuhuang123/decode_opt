#!/bin/bash
# SPDX-License-Identifier: MIT
#
# bench/run_sweep.sh —— decode_gemm 全流程 sweep（GPU 一律经 tools/gpurun_dg.sh v2）。
#
#   ./run_sweep.sh                                       # 5 模型 x 7 step x M 阶梯
#   ./run_sweep.sh --demo                                # 小 sweep 自测
#   ./run_sweep.sh --models kimi_k3 --steps all --ms 1,8,64
#   BATCH=A ./run_sweep.sh --models kimi_k3,glm52 ...    # 并行跑多批（各自独立 CSV）
#   BIN=./bench_decode_tune ./run_sweep.sh --tune ...
#   RESUME=0 ./run_sweep.sh ...                          # 不续跑，新开一份 CSV
#   PLOT=1 ./run_sweep.sh ...                            # 跑完顺带出正式图（fig_bw_vs_M_*）
#
# GPU 纪律（CONTRACT §0 v2，2026-09-13 用户裁决「卡用了才用，不许长期占」）：
#   * 一律经 tools/gpurun_dg.sh v2：BORROW 池 {2,3} 优先（v19_swapab keeper 预留但
#     物理空闲，授权借用；util<=5 且除 keeper 白名单外无 compute pid；运行中 2s
#     看门狗，外来 compute 落卡 -> abort 让路 exit 75），CLEAN 池 {0,1} 兜底
#     （util<=5 / mem<=2000MiB / 完全无 compute pid）。两池都抢
#     _lockbench_20260913/locks/gpu<N>.lock 跨项目 flock，跑前锁频 1830。
#   * keeper 进程不 kill、不碰。
#   * 看门狗 abort(exit 75) / timeout(124) / 被 kill(143) -> 本脚本自动重排，
#     靠 bench_decode 的增量落盘 + --resume 只补缺口（最多 MAX_RETRY 次）。
#
# 心跳：等卡超过 WAIT_HEARTBEAT(600s) 打 WAITING；拿到卡后每 HEARTBEAT(300s) 打一行
#       HEARTBEAT，带 4 张卡的 util/mem/foreign pid + 已完成行数/应有行数。
#
# 产物：$P/results/$BATCH/ 下 bench_<UTC>.csv(16 列) / _detail.csv / .json /
#       sweep_<UTC>.log；跑完自动复制到 $P/results/（COLLECT）供 plot agent 吃。
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=$(cd "$HERE/.." && pwd)
BIN=${BIN:-$HERE/bench_decode}
BATCH=${BATCH:-main}
TARGET=${DG_TARGET:-bench_$BATCH}
RESULTS=${RESULTS:-$P/results/$BATCH}
COLLECT=${COLLECT:-$P/results}
RESUME=${RESUME:-1}
MAX_RETRY=${MAX_RETRY:-30}
HEARTBEAT=${HEARTBEAT:-300}
WAIT_HEARTBEAT=${WAIT_HEARTBEAT:-600}
RETRY_SLEEP=${RETRY_SLEEP:-5}
export DG_POLL=${DG_POLL:-3}
mkdir -p "$RESULTS" "$COLLECT"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$RESULTS/sweep_$TS.log"

log() { echo "$@" | tee -a "$LOG"; }

# ---- 应有行数（心跳里的 done/total）----
expected_rows() {
  python3 - "$@" <<'PY'
import sys
a = sys.argv[1:]
def val(flag, dflt):
    if flag in a:
        i = a.index(flag)
        if i + 1 < len(a):
            return a[i + 1]
    return dflt
def expand(txt, lo, hi, dflt_n):
    if txt in (None, "all", ""):
        return dflt_n
    n = set()
    for it in txt.split(","):
        if it == "all":
            n.update(range(lo, hi + 1)); continue
        if "-" in it:
            x, y = it.split("-", 1); n.update(range(int(x), int(y) + 1))
        elif it.strip():
            n.add(int(it))
    return len(n)
models = val("--models", "all")
nm = 5 if models == "all" else len([x for x in models.split(",") if x])
ns = expand(val("--steps", "all"), 0, 6, 7)
nms = expand(val("--ms", None), 1, 128, 8)
print(nm * ns * nms * 3)
PY
}
TOTAL_ROWS=$(expected_rows "$@" 2>/dev/null || echo "?")

log "[run_sweep] $(date -u +%FT%TZ) batch=$BATCH project=$P"
log "[run_sweep] bin=$BIN target=$TARGET results=$RESULTS collect=$COLLECT"
log "[run_sweep] resume=$RESUME max_retry=$MAX_RETRY heartbeat=${HEARTBEAT}s DG_POLL=$DG_POLL"
log "[run_sweep] args: $*   (应有 CSV 数据行 ~ $TOTAL_ROWS)"

if [ ! -x "$BIN" ]; then
  log "[run_sweep] $BIN 不存在 -> make TUNE=0 -j8"
  make -C "$HERE" TUNE=0 -j8 >>"$LOG" 2>&1 || { echo "[run_sweep] BUILD FAIL，见 $LOG" >&2; exit 1; }
fi

gpu_snapshot() {   # 4 张卡的 util/mem/foreign pid（keeper 不算 foreign）
  for i in 0 1 2 3; do
    u=$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
    m=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
    fp=""
    for p in $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \r'); do
      c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
      case "$c" in *keeper.py*) fp="$fp keeper($p)";; *) fp="$fp FOREIGN($p)";; esac
    done
    printf "GPU%s util=%s%% mem=%sMiB%s | " "$i" "${u:-?}" "${m:-?}" "${fp:- noproc}"
  done
}

done_rows() {
  f=$(ls -1t "$RESULTS"/bench_2*.csv 2>/dev/null | grep -v _detail | head -1)
  [ -z "$f" ] && { echo 0; return; }
  echo $(( $(wc -l < "$f") - 1 ))
}

run_once() {   # 一次尝试；返回 gpurun_dg 的退出码
  local args=(--out "$RESULTS")
  [ "$RESUME" = "1" ] && args+=(--resume)
  args+=("$@")
  local note="gpurun_dg.sh v2: BORROW{2,3}+CLEAN{0,1} flock+guards+watchdog2s+lgc1830"
  "$P/tools/gpurun_dg.sh" "$TARGET" \
    env BENCH_GPURUN_TARGET="$TARGET" BENCH_EXCLUSIVE_CHECK="$note" \
        BENCH_RESULTS_DIR="$RESULTS" BENCH_BATCH="$BATCH" \
        bash -c 'export BENCH_GPURUN_GPU="$CUDA_VISIBLE_DEVICES"; exec "$@"' \
        bench "$BIN" "${args[@]}" > >(tee -a "$LOG") 2>&1 &
  local gpid=$!
  # 注意：不能写在一行 local 里 —— bash 会先展开所有词，$start 那时还没赋值，
  # set -u 下会直接 unbound variable 把脚本打死（踩过一次）
  local start next_beat got now logmark
  # 只看本次 attempt 新增的日志行（log 是跨 attempt 追加的，否则会误判「已拿到卡」）
  logmark=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  start=$(date +%s)
  next_beat=$((start + WAIT_HEARTBEAT))
  got=0
  while kill -0 "$gpid" 2>/dev/null; do
    sleep 5
    if [ "$got" = "0" ] && tail -n "+$((logmark + 1))" "$LOG" 2>/dev/null | grep -q -- "\-> \(BORROW\|CLEAN\) GPU"; then
      got=1
      next_beat=$(( $(date +%s) + HEARTBEAT ))
      log "[run_sweep] $(date -u +%FT%TZ) 拿到卡，排队 $(( $(date +%s) - start ))s"
    fi
    now=$(date +%s)
    if [ "$now" -ge "$next_beat" ]; then
      if [ "$got" = "1" ]; then
        log "[run_sweep] HEARTBEAT +$((now - start))s rows=$(done_rows)/$TOTAL_ROWS | $(gpu_snapshot)"
      else
        log "[run_sweep] WAITING $((now - start))s 还没有干净卡 | $(gpu_snapshot)"
      fi
      next_beat=$((now + HEARTBEAT))
    fi
  done
  wait "$gpid"; local rc=$?
  sleep 1   # 让 process substitution 的 tee 落盘
  return $rc
}

log "[run_sweep] 起跑前 GPU 状态: $(gpu_snapshot)"

attempt=0; rc=1
while :; do
  attempt=$((attempt + 1))
  log "[run_sweep] === attempt $attempt/$((MAX_RETRY + 1)) $(date -u +%FT%TZ) ==="
  run_once "$@"; rc=$?
  log "[run_sweep] attempt $attempt rc=$rc rows=$(done_rows)/$TOTAL_ROWS"
  if [ $rc -eq 0 ]; then break; fi
  case $rc in
    75)  reason="看门狗让路（外来 compute 落到借用卡）";;
    124) reason="gpurun_dg 等卡 TIMEOUT";;
    143) reason="被 SIGTERM（可能是别的调度）";;
    *)   reason="非争用错误，不自动重排"; log "[run_sweep] rc=$rc: $reason"; break;;
  esac
  if [ $attempt -gt $MAX_RETRY ]; then
    log "[run_sweep] 重排次数用尽（$MAX_RETRY），停止；已落盘的数据可用 --resume 继续"
    break
  fi
  # 指数退避（上限 60s）：借用卡被外来 compute 抢走时，别 5s 就扑上去反复作废
  backoff=$((RETRY_SLEEP * attempt)); [ $backoff -gt 60 ] && backoff=60
  log "[run_sweep] rc=$rc: $reason -> ${backoff}s 后自动重排（--resume 补缺口，已落盘 $(done_rows)/$TOTAL_ROWS 行）"
  RESUME=1
  sleep "$backoff"
done

# ---- 收结果：复制到 COLLECT，方便 plot agent 一处吃 ----
log "[run_sweep] final rc=$rc"
if [ "$RESULTS" != "$COLLECT" ]; then
  for f in "$RESULTS"/bench_2*.csv "$RESULTS"/bench_2*.json "$RESULTS"/tune_2*.csv; do
    [ -e "$f" ] || continue
    cp -f "$f" "$COLLECT"/ && log "[run_sweep] collected -> $COLLECT/$(basename "$f")"
  done
fi
latest_csv=$(ls -1t "$RESULTS"/bench_2*.csv 2>/dev/null | grep -v _detail | head -1)
latest_json=$(ls -1t "$RESULTS"/bench_2*.json 2>/dev/null | head -1)
if [ -n "$latest_csv" ]; then
  log "[run_sweep] csv  = $latest_csv ($(done_rows)/$TOTAL_ROWS data rows)"
  log "[run_sweep] header: $(head -1 "$latest_csv")"
  log "[run_sweep] cols  : $(head -1 "$latest_csv" | awk -F, '{print NF}')"
fi
[ -n "$latest_json" ] && log "[run_sweep] json = $latest_json"
log "[run_sweep] log  = $LOG"

# ---- 出图（orchestrator 2026-09-13 要求把 plot 命令挂在 run_sweep 尾部）----
# 默认不自动出图：单个 batch 跑完就出图会把正式图覆盖成半成品。要出图：
#   PLOT=1 ./run_sweep.sh --models ... --steps ...        # 本批跑完顺带出正式图
#   bash bench/make_plots.sh                              # 只出图（不跑 GPU）
# 正式图口径：results/bench_2*.csv 按 mtime 时间序全部喂给 analysis/plot.py
# （后者覆盖前者同 key 行），--fig-prefix fig => results/fig_bw_vs_M_*.png；
# 显式不碰 orchestrator 的 fig_v1_*（那是 v1 数据的存档图）。
if [ "${PLOT:-0}" = "1" ]; then
  log "[run_sweep] PLOT=1 -> bench/make_plots.sh (prefix=${PREFIX:-fig})"
  if PREFIX="${PREFIX:-fig}" bash "$HERE/make_plots.sh" >> "$LOG" 2>&1; then
    log "[run_sweep] 图已出: $COLLECT/${PREFIX:-fig}_bw_vs_M_*.png"
  else
    log "[run_sweep] 出图失败（看 $LOG 尾部）；手工: bash bench/make_plots.sh"
  fi
fi
exit $rc
