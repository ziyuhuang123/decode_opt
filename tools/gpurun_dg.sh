#!/bin/bash
# gpurun_dg.sh <target_id> <cmd...>
# decode_gemm 的 GPU 借用调度器 v2（2026-09-13 用户裁决：卡用了才用，不许长期占）。
#
# 卡池语义：
#   BORROW 池 {2,3}：被 v19_swapab 的 keeper 进程「预留但物理空闲」。用户授权借用：
#       只在 util<=5% 且除 keeper 白名单外无 compute pid 时启动；
#       运行中每 2s 看门狗采样，一旦外来 compute 进程落卡 => 30s 内 abort 本作废，让路。
#   CLEAN 池 {0,1}：严格守卫（util<=5 且 mem<=2000 且完全无 compute pid），无 keeper。
# 两池都先抢 _lockbench_20260913/locks/gpu<N>.lock 的 flock（跨项目串行）。
# 跑前重申 -lgc 1830,1830。CUDA_VISIBLE_DEVICES 由本脚本设置。
set -u
LB=/home/admin/workspace/hengfeng_data/weave_new/DSA-learn/mk/h20/cutedsl/_lockbench_20260913
# ${VAR-default} (no colon) on purpose: an EMPTY DG_BORROW/DG_CLEAN is a real request to skip
# that pool, not an unset variable. The frozen contract's execution.allowed_gpu_ids may be
# narrower than the borrow pool (this run: [0, 1]), and hw_capture refuses to sample PM
# counters on a card outside it, so `DG_BORROW= gpurun_dg.sh ...` must mean "clean pool only".
BORROW="${DG_BORROW-2 3}"
CLEAN="${DG_CLEAN-0 1}"
LOCK_MHZ="${DG_LOCK_MHZ:-1830}"
UTIL_MAX="${DG_UTIL_MAX:-5}"
MEM_MAX="${DG_MEM_MAX:-2000}"
TIMEOUT="${DG_TIMEOUT:-21600}"
POLL="${DG_POLL:-3}"
WATCH="${DG_WATCH:-2}"
TARGET="${1:?usage: gpurun_dg.sh <target_id> <cmd...>}"; shift

q() { nvidia-smi -i "$1" --query-gpu="$2" --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r'; }
pids() { nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \r' | tr '\n' ' '; }
is_keeper() {  # pid -> 0 if keeper.py reservation holder
  local c; c=$(tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null)
  case "$c" in *keeper.py*) return 0;; *) return 1;; esac
}
foreign_pids() {  # card -> pids that are neither keeper nor our descendants
  local card="$1" self="$2" p out=""
  for p in $(pids "$card"); do
    [ "$p" = "$self" ] && continue
    if is_keeper "$p"; then continue; fi
    if ps -o pid= -o ppid= -p "$p" 2>/dev/null | awk -v s="$self" '$2==s{found=1} END{exit !found}'; then continue; fi
    # descendant check (our cmd may spawn children)
    local anc=$p hit=0
    while [ "$anc" != "1" ] && [ "$anc" != "0" ] && [ -n "$anc" ]; do
      [ "$anc" = "$self" ] && { hit=1; break; }
      anc=$(ps -o ppid= -p "$anc" 2>/dev/null | tr -d ' ')
    done
    [ "$hit" = 1 ] && continue
    out="$out $p"
  done
  echo "$out"
}

start=$(date +%s)
while :; do
  # ---- borrow pool first (idle-reserved cards, user-authorized) ----
  for GPU in $BORROW; do
    exec 9>"$LB/locks/gpu$GPU.lock"
    if flock -n 9; then
      u=$(q "$GPU" utilization.gpu); f=$(foreign_pids "$GPU" $$)
      if [ -n "$u" ] && [ "$u" -le "$UTIL_MAX" ] && [ -z "$f" ]; then
        nvidia-smi -i "$GPU" -lgc "$LOCK_MHZ,$LOCK_MHZ" >/dev/null 2>&1
        echo "[gpurun_dg] $TARGET -> BORROW GPU$GPU (util=$u keepers-only)" >&2
        CUDA_VISIBLE_DEVICES="$GPU" "$@" &
        cmdpid=$!
        # watchdog: yield the card the moment foreign compute lands
        while kill -0 "$cmdpid" 2>/dev/null; do
          sleep "$WATCH"
          f=$(foreign_pids "$GPU" "$cmdpid")
          if [ -n "$f" ]; then
            echo "[gpurun_dg] CONTENTION on GPU$GPU foreign=$f -> aborting $TARGET (data discarded)" >&2
            kill -TERM -"$cmdpid" 2>/dev/null || kill -TERM "$cmdpid" 2>/dev/null
            sleep 3; kill -KILL "$cmdpid" 2>/dev/null
            wait "$cmdpid" 2>/dev/null
            echo "[gpurun_dg] $TARGET ABORTED contended=true" >&2
            exit 75
          fi
        done
        wait "$cmdpid"; rc=$?
        echo "[gpurun_dg] $TARGET done rc=$rc (borrowed GPU$GPU)" >&2
        exit $rc
      fi
      flock -u 9
    fi
  done
  # ---- clean pool (strict) ----
  for GPU in $CLEAN; do
    exec 8>"$LB/locks/gpu$GPU.lock"
    if flock -n 8; then
      u=$(q "$GPU" utilization.gpu); m=$(q "$GPU" memory.used); p=$(pids "$GPU")
      if [ -n "$u" ] && [ "$u" -le "$UTIL_MAX" ] && [ -n "$m" ] && [ "$m" -le "$MEM_MAX" ] && [ -z "$p" ]; then
        nvidia-smi -i "$GPU" -lgc "$LOCK_MHZ,$LOCK_MHZ" >/dev/null 2>&1
        echo "[gpurun_dg] $TARGET -> CLEAN GPU$GPU (util=$u mem=$m procs=none)" >&2
        CUDA_VISIBLE_DEVICES="$GPU" "$@"
        rc=$?
        echo "[gpurun_dg] $TARGET done rc=$rc (clean GPU$GPU)" >&2
        exit $rc
      fi
      flock -u 8
    fi
  done
  now=$(date +%s)
  if [ $((now - start)) -ge "$TIMEOUT" ]; then
    echo "[gpurun_dg] $TARGET TIMEOUT no card available" >&2; exit 124
  fi
  sleep "$POLL"
done
