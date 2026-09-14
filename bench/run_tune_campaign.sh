#!/bin/bash
# bench/run_tune_campaign.sh <A|B>  — 分模型串行的 --tune 驱动（step4 网格，PDL only）
#   A = kimi_k3 glm52 deepseek_v4_pro      B = qwen36 minimax_m3
# 每 (model,M) 一个 chunk；--tune-out 指向 results/tune_<model>.csv，二进制内部
# purge+append + 每 config 一行 flush => 看门狗 abort(exit 75) 最多丢当前 config，
# 重跑自动跳过已完成 config。主 CSV 落到 /tmp（tune-only 不产生 canonical 行）。
set -u
P=/home/admin/workspace/hengfeng_data/weave_new/DSA-learn/mk/h20/cutedsl/decode_gemm
cd "$P" || exit 1
BIN="${BIN:-$P/bench/bench_decode_tune}"
WHICH="${1:?usage: run_tune_campaign.sh A|B}"
REPS="${REPS:-2}"
mkdir -p /tmp/tune_collect results
case "$WHICH" in
  A) MODELS="kimi_k3 glm52 deepseek_v4_pro" ;;
  B) MODELS="qwen36 minimax_m3" ;;
  *) MODELS="$WHICH" ;;
esac
echo "[tune$WHICH] start $(date -u +%FT%TZ) bin=$(ls -la --time-style=+%H:%M:%S "$BIN" | awk '{print $6}') models=$MODELS"
for MODEL in $MODELS; do
  for M in 1 16 64; do
    OUT="$P/results/tune_${MODEL}.csv"
    for try in 1 2 3 4 5 6; do
      echo "[tune$WHICH] chunk $MODEL M=$M try=$try $(date -u +%H:%M:%S)"
      tools/gpurun_dg.sh "tune${WHICH}_${MODEL}_M${M}" "$BIN" \
        --models "$MODEL" --steps 4 --ms "$M" --tune-only \
        --protocols PDL --reps "$REPS" \
        --out /tmp/tune_collect --tune-out "$OUT"
      rc=$?
      echo "[tune$WHICH] chunk $MODEL M=$M rc=$rc try=$try $(date -u +%H:%M:%S)"
      [ "$rc" = 75 ] && { echo "[tune$WHICH] watchdog abort -> requeue after 20s"; sleep 20; continue; }
      [ "$rc" = 0 ] || [ "$rc" = 1 ] && break
      [ "$rc" = 124 ] && { echo "[tune$WHICH] no-card timeout -> requeue after 60s"; sleep 60; continue; }
      echo "[tune$WHICH] chunk rc=$rc unexpected -> requeue after 15s"; sleep 15
    done
  done
  n=$(wc -l < "$P/results/tune_${MODEL}.csv" 2>/dev/null || echo 0)
  echo "[tune$WHICH] MODEL $MODEL done rows=$n $(date -u +%H:%M:%S)"
done
echo "[tune$WHICH] ALL DONE $(date -u +%H:%M:%S)"
