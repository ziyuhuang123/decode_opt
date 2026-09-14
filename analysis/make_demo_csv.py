#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成符合 CONTRACT §6 schema 的**合成**（假）bench CSV，只用于 plot.py 渲染自测。

⚠ 这里所有数值都是按物理模型编出来的，不是实测结果，不能进文章。
  文件名强制以 `demo_` 开头，plot.py 也会在图上打红字 DEMO 横幅。

物理模型（参考 $REF/RESULTS_h20_fp8_decode.md 的实测量级，让假数据看起来像真的）：
  bytes(M)   = N*K + M*K + 2*M*N                      # CONTRACT §6 带宽分子
  t_ideal    = bytes / (ceiling(step) * 4000 GB/s)     # 访存模式天花板（单 op 字节数决定）
  t          = t_ideal + overhead(protocol, step)      # 每次 launch 的固定开销（PDL 主要削这个）
  bw_mem     = bytes / t
  bw         = min(bw_mem, roofline_cap(M) * compute_eff(step))   # 大 M 撞计算 roof
  ceiling/op 字节数/overhead 的量级取自 REF 的诚实测量表（5KB→38%、10KB→61%、20KB→81%、40KB→92%；
  固定开销 4-6µs；PDL 后 M=1 N=6144 K=7168 = 12.9µs / 3412 GB/s / 85.3%）。

三种协议全部写出（ISO / B2B / PDL），因为 CONTRACT §6 要求 bench 全记录；
plot.py 画图时按口径自己挑：step0=B2B，step>=1=PDL。

用法：
  python3 analysis/make_demo_csv.py --out results/demo_bench.csv
  python3 analysis/plot.py --demo          # 一步：生成 + 画图
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_JSON = PROJECT_ROOT / "models" / "models.json"

# CONTRACT §6 列顺序（不许改）
HEADER = ["model", "N", "K", "M", "step", "step_name", "protocol",
          "latency_us_p50", "bandwidth_gbps", "pct_of_spec_peak", "rel_l2", "pass", "config"]

SPEC_PEAK_GBPS = 4000.0        # HBM3 理论峰值（CONTRACT §7）
COMPUTE_PEAK_TFLOPS = 282.7    # H20 FP8 实测峰值（CONTRACT §8）
M_LADDER = [1, 2, 4, 8, 16, 32, 64, 128]
M_TILE_LADDER = [8, 16, 32, 64, 128]
PROTOCOLS = ["ISO", "B2B", "PDL"]

STEP_NAMES = {0: "baseline", 1: "pdl", 2: "blockk", 3: "prepack",
              4: "tile_stage", 5: "epi_overlap", 6: "splitk"}
N_STEPS = len(STEP_NAMES)

# CONTRACT §3 canonical config（M_TILE=8 那一行是原文；M_TILE>=16 按 §3 规则推）
# step -> (BM, WG, REG, STG, TILES, SUBS, PREPACK, PDL, EPI, SPLITK)
STEP_CFG = {
    0: (64, 4, 112, 2, 4, 1, 0, 0, 0, 0),
    1: (64, 4, 112, 2, 4, 1, 0, 1, 0, 0),
    2: (64, 4, 112, 2, 4, 2, 0, 1, 0, 0),
    3: (64, 4, 112, 2, 4, 2, 1, 1, 0, 0),
    4: (48, 2, 80, 3, 2, 2, 1, 1, 0, 0),
    5: (48, 2, 80, 3, 2, 2, 1, 1, 1, 0),
    6: (48, 2, 80, 3, 2, 2, 1, 1, 1, 1),
}

# 访存模式天花板：随 step 变大（单 op 字节数 5KB→40KB、prepack 全连续、更深流水）
CEILING = {0: 0.620, 1: 0.620, 2: 0.720, 3: 0.800, 4: 0.880, 5: 0.905, 6: 0.915}
# 每次 launch 的固定开销 µs（REF 诊断：4-6µs；PDL 把它和下一次 launch 的 prologue 重叠掉）
OVERHEAD = {
    0: {"ISO": 6.5, "B2B": 4.5, "PDL": 4.5},   # step0 无 in-kernel trigger，PDL 属性基本空转
    1: {"ISO": 6.0, "B2B": 4.0, "PDL": 1.9},   # REF: PDL 把 4-6µs 固定开销压到 ~1.9µs
    2: {"ISO": 5.8, "B2B": 3.8, "PDL": 1.7},
    3: {"ISO": 5.5, "B2B": 3.5, "PDL": 1.5},
    4: {"ISO": 5.2, "B2B": 3.2, "PDL": 1.1},
    5: {"ISO": 5.0, "B2B": 3.0, "PDL": 0.8},   # epilogue store 与下个 kernel prologue 重叠
    6: {"ISO": 5.4, "B2B": 3.3, "PDL": 0.9},   # splitk 多一趟 gmem reduce
}
# 计算瓶颈区的效率（大 M 时 wgmma 发射率 / 寄存器压力决定，也是逐步变好）
COMPUTE_EFF = {0: 0.800, 1: 0.830, 2: 0.860, 3: 0.885, 4: 0.910, 5: 0.925, 6: 0.935}

# 兜底形状 = models/models.json 的定稿真形状（N,K 是真的，带宽数值仍是编的）。
# 形状差异必须看得出来：qwen36 权重只有 8.4MB（M=1 纯访存 ~2.3µs，launch 固定开销占比大 → 到不了 85%），
# deepseek_v4_pro 权重 117MB（长 launch 摊薄开销 → 逼近 85%）。
FALLBACK_MODELS = [
    ("kimi_k3", "Kimi-K3", 7168, 12288, "o_proj"),
    ("qwen36", "Qwen3.6-35B-A3B", 2048, 4096, "o_weight"),
    ("glm52", "GLM-5.2", 6144, 16384, "wo"),
    ("deepseek_v4_pro", "DeepSeek-V4-Pro", 7168, 16384, "wo_b"),
    ("minimax_m3", "MiniMax-M3", 6144, 8192, "w_o"),
]


@dataclass
class DemoModel:
    id: str
    display: str
    N: int
    K: int
    layer: str = ""


def moved_bytes(N: int, K: int, M: int) -> float:
    return float(N) * K + float(M) * K + 2.0 * M * N


def roofline_cap_gbps(N: int, K: int, M: int) -> float:
    """BW_cap(M) = min(spec, bytes / t_floor)，t_floor = 2*M*N*K / 282.7 TFLOPS。"""
    t_floor = 2.0 * M * N * K / (COMPUTE_PEAK_TFLOPS * 1e12)
    return min(SPEC_PEAK_GBPS, moved_bytes(N, K, M) / t_floor / 1e9)


def m_tile_of(M: int) -> int:
    return next((t for t in M_TILE_LADDER if M <= t), M_TILE_LADDER[-1])


def config_str(step: int, M: int) -> str:
    """CONTRACT §3 canonical config；M_TILE>=16 按 §3 规则调（TILES=WG，128 用 REG=168/STG=2）。"""
    bm, wg, reg, stg, tiles, subs, pre, pdl, epi, sk = STEP_CFG[step]
    mt = m_tile_of(M)
    if mt >= 16:
        wg = 2
        tiles = wg                 # TILES_PER_CTA = NUM_MATH_WG（§3）
        reg = 168 if mt >= 128 else 112
        stg = 2 if mt >= 64 else 3
        if mt >= 128:
            subs = 1               # smem/reg 双约束（§3）
        bm = max(32, min(bm, 48))
    return (f"M_TILE={mt},BM={bm},WG={wg},REG={reg},STG={stg},TILES={tiles},SUBS={subs},"
            f"PREPACK={pre},PDL={pdl},EPI={epi},SPLITK={sk}")


def splitk_boost(step: int, N: int, M: int) -> float:
    """S6 splitk 的收益：CTA 数不足（小 N）或 M 大时明显，否则几乎为 0 甚至略亏。"""
    if step < 6:
        return 1.0
    bm = STEP_CFG[step][0]
    ctas = max(1, math.ceil(N / bm))
    util = min(1.0, ctas / 132.0)             # H20 132 SM
    gain = 0.10 * (1.0 - util)                # 并行度不足 → K-split 有用
    if M >= 32:
        gain += 0.02
    if util >= 0.95 and M < 32:
        gain = -0.008                          # CTA 已经铺满，splitk 只多一趟 reduce
    return 1.0 + gain


def bw_gbps(model: DemoModel, step: int, M: int, protocol: str, jitter: float = 0.0) -> float:
    N, K = model.N, model.K
    bts = moved_bytes(N, K, M)
    ceil_ = min(0.955, CEILING[step] * splitk_boost(step, N, M))
    t_ideal = bts / (ceil_ * SPEC_PEAK_GBPS * 1e9) * 1e6        # µs
    t = t_ideal + OVERHEAD[step][protocol]
    bw_mem = bts / (t * 1e-6) / 1e9
    cap = roofline_cap_gbps(N, K, M) * COMPUTE_EFF[step]
    return max(1.0, min(bw_mem, cap) * (1.0 + jitter))


def _crc(*keys) -> int:
    """确定性 32bit 哈希（不用内置 hash()：它被 PYTHONHASHSEED 随机化，会破坏可复现）。"""
    import zlib
    h = 0
    for k in keys:
        h = zlib.crc32(str(k).encode("utf-8"), h)
    return h


def det_jitter(seed: int, *keys) -> float:
    """确定性 ±1.2% 抖动（同 seed 同结果，避免看起来太完美又不影响可复现）。"""
    return ((_crc(seed, *keys) % 2401) / 2400.0 - 0.5) * 0.024


def demo_models(models_json: Path | None = None) -> list[DemoModel]:
    """优先用 models/models.json 的真形状（数值仍是假的），没有就用兜底形状。"""
    if models_json and models_json.exists():
        try:
            obj = json.loads(models_json.read_text(encoding="utf-8"))
            out = []
            for m in obj.get("models", []) or []:
                g = m.get("gemm") or {}
                N, K = int(g.get("N", 0)), int(g.get("K", 0))
                if N > 0 and K > 0:
                    out.append(DemoModel(str(m.get("id")), str(m.get("display") or m.get("id")),
                                         N, K, str(g.get("layer") or "")))
            if out:
                print(f"[make_demo_csv] 用 {models_json.name} 的真形状（{len(out)} 个模型），数值仍是合成的",
                      flush=True)
                return out
        except Exception as exc:  # noqa: BLE001
            print(f"[make_demo_csv][WARN] {models_json} 解析失败（{exc}），用兜底形状", flush=True)
    else:
        print(f"[make_demo_csv][WARN] {models_json} 不存在 → 用兜底 demo 形状（等 shapes agent 产出后自动跟随）",
              flush=True)
    return [DemoModel(*t) for t in FALLBACK_MODELS]


def step_monotonic_bw(md: "DemoModel", M: int, protocol: str, seed: int = 7) -> dict[int, float]:
    """某个 (model, M, protocol) 上 7 个 step 的带宽，强制 step 递增（cumulative 技术栈）。

    write_demo_csv 和 --summary 都走这里，保证打印出来的数字 == CSV 里的数字。
    """
    out: dict[int, float] = {}
    prev = 0.0
    for st in range(N_STEPS):
        bw = bw_gbps(md, st, M, protocol, jitter=det_jitter(seed, md.id, st, M))
        bw = max(bw, prev * 1.002)
        prev = bw
        out[st] = bw
    return out


def write_demo_csv(path: Path, models_json: Path | None = None,
                   ms: Sequence[int] = M_LADDER, seed: int = 7,
                   models: Sequence["DemoModel"] | None = None) -> int:
    """写 CSV，返回数据行数（不含表头）。"""
    if models is None:
        models = demo_models(models_json if models_json is not None else MODELS_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for md in models:
            # 先把每个 (M, protocol) 的 7 个 step 算出来（含 step 单调约束），再按 step 外层写行
            series = {(M, proto): step_monotonic_bw(md, M, proto, seed)
                      for M in ms for proto in PROTOCOLS}
            for step in range(N_STEPS):
                for M in ms:
                    for proto in PROTOCOLS:
                        bw = series[(M, proto)][step]
                        bts = moved_bytes(md.N, md.K, M)
                        lat = bts / (bw * 1e9) * 1e6
                        rel_l2 = 3.0e-4 + 1.2e-3 * ((_crc("l2", md.id, step, M) % 997) / 997.0)
                        w.writerow([
                            md.id, md.N, md.K, M, step, STEP_NAMES[step], proto,
                            f"{lat:.4f}", f"{bw:.3f}", f"{bw / SPEC_PEAK_GBPS:.6f}",
                            f"{rel_l2:.6e}", 1, config_str(step, M),
                        ])
                        n += 1
    return n


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="输出 CSV 路径（默认 results/demo_bench_<UTC>.csv；必须以 demo_ 开头）")
    ap.add_argument("--models-json", default=str(MODELS_JSON), help="models/models.json（可选，取真形状）")
    ap.add_argument("--ms", default=",".join(str(m) for m in M_LADDER), help="M 阶梯，逗号分隔")
    ap.add_argument("--seed", type=int, default=7, help="抖动种子（同 seed 结果可复现）")
    ap.add_argument("--summary", action="store_true", help="打印每模型 S0/S6 的带宽摘要")
    args = ap.parse_args(argv)

    ms = sorted({int(x) for x in args.ms.split(",") if x.strip()})
    if args.out:
        out = Path(args.out)
        if not out.name.startswith("demo_"):
            print(f"[make_demo_csv][WARN] {out.name} 不以 demo_ 开头：合成数据必须和真结果区分开，已改名",
                  flush=True)
            out = out.with_name("demo_" + out.name)
    else:
        out = RESULTS_DIR / f"demo_bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"

    models = demo_models(Path(args.models_json))
    n = write_demo_csv(out, models_json=Path(args.models_json), ms=ms, seed=args.seed,
                       models=models)
    print(f"[make_demo_csv] 写出 {out}（{n} 行 = {len(models)} 模型 × {N_STEPS} step × "
          f"{len(ms)} 个 M × {len(PROTOCOLS)} 协议）", flush=True)
    print("[make_demo_csv] ⚠ 合成数据，仅用于渲染自测，不是实测结果，不能进文章。", flush=True)
    print(f"[make_demo_csv] 画图：python3 analysis/plot.py --csv {out} --fig-prefix fig_demo", flush=True)

    if args.summary:
        def proto_of(step: int) -> str:
            return "B2B" if step == 0 else "PDL"      # 画图口径（CONTRACT §6）

        print(f"\n{'model':<18}{'N':>7}{'K':>7}{'wgt_MB':>8}  " +
              "".join(f"{('S%d' % st):>7}" for st in range(N_STEPS)))
        for M in (min(ms), max(ms)):
            print(f"--- 画图口径（S0=B2B, S1-S6=PDL）M={M} 的合成等效带宽 GB/s "
                  f"（= CSV 里的 bandwidth_gbps）---")
            for md in models:
                bw = step_monotonic_bw(md, M, "PDL" if M else "PDL", args.seed)
                bwb = step_monotonic_bw(md, M, "B2B", args.seed)
                vals = [(bwb[0] if st == 0 else bw[st]) for st in range(N_STEPS)]
                print(f"{md.id:<18}{md.N:>7}{md.K:>7}{md.N * md.K / 1e6:>8.1f}  " +
                      "".join(f"{v:>7.0f}" for v in vals) +
                      f"   | roof={roofline_cap_gbps(md.N, md.K, M):>6.0f}  "
                      f"lat(S5)={moved_bytes(md.N, md.K, M) / (vals[5] * 1e3):>7.2f}us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
