#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""decode_gemm 画图脚本（CONTRACT §8）。

读 results/bench_*.csv（--csv 指定，否则取最新），产出：
  results/fig_bw_vs_M_<model>.png   每模型一张（7 条 step 曲线 + 参考线 + roofline）
  results/fig_main_bw_vs_M.png      汇总（2×3，第 6 格放图例与口径说明）

x = M（1..128，log2 刻度），y = 等效带宽 GB/s。
曲线口径（CONTRACT §6）：step0 用 B2B 协议，step>=1 用 PDL 协议，延迟取 p50。

三条参考线：
  1. 水平实线  4000 GB/s          —— HBM3 理论峰值 4.0 TB/s（用户指定，不许替换）
  2. 水平虚线  results/peak_bw.json 的实测可达峰值（缺失则跳过 + 警告，绝不编数）
  3. 点划线    roofline 等效带宽上限 BW_cap(M) = min(4000, bytes(M) / t_floor(M))
               t_floor = 2*M*N*K / 282.7 TFLOPS（H20 FP8 实测峰值，CONTRACT §8 写死常数）

中文字体：先探测系统 / 捆绑 / 环境变量指定的 CJK 字体，并逐字符校验 glyph 覆盖；
探测不到或覆盖不全 → 自动退回英文标签 + stdout 警告（绝不出方块）。

用法：
  python3 analysis/plot.py                       # 自动取 results/bench_*.csv 最新一份
  python3 analysis/plot.py --csv results/bench_20260913T000000Z.csv
  python3 analysis/plot.py --demo                # 无真数据时生成假 CSV 自测渲染
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --------------------------------------------------------------------------------------
# 常量（口径全部来自 CONTRACT，改这里等于改接口，别动）
# --------------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_JSON = PROJECT_ROOT / "models" / "models.json"
PEAK_BW_JSON = RESULTS_DIR / "peak_bw.json"

SPEC_PEAK_GBPS = 4000.0          # HBM3 理论峰值 4.0 TB/s（CONTRACT §7：用户指定主参考线）
COMPUTE_PEAK_TFLOPS = 282.7      # H20 FP8 实测峰值（CONTRACT §8：写死常数）
M_TICKS = [1, 2, 4, 8, 16, 32, 64, 128]

# CONTRACT §3 的 7 个技术步骤（顺序不许改）
# 大白话图例名（技术名 -> 人话）
PLAIN_STEP = {
    0: "基线：权重当A+流水",
    1: "PDL：下层提前启动",
    2: "一次拉2个K块",
    3: "权重排成连续块",
    4: "切tile+加深流水",
    5: "K切开填满SM",
}

STEP_NAMES = {
    0: "baseline",
    1: "pdl",
    2: "blockk",
    3: "prepack",
    4: "tile_stage",
    5: "epi_overlap",
    6: "splitk",
}
# CONTRACT §6：文章曲线口径 step0=B2B，step>=1=PDL
CANON_PROTOCOL = {0: "B2B"}
DEFAULT_PROTOCOL = "PDL"
# 标准协议缺数据时的回退顺序（会在图注和 stdout 明确标注，不静默）
FALLBACK_ORDER = {
    0: ["B2B", "ISO", "PDL"],
    None: ["PDL", "B2B", "ISO"],
}

# step 曲线配色：灰 -> 红 渐进（CONTRACT §8「颜色渐变」）
_STEP_CMAP = LinearSegmentedColormap.from_list(
    "gray2red", ["#8f8f8f", "#b07a5e", "#cd5945", "#c3352b", "#a51d24", "#7a1019"]
)
_STEP_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
COLOR_SPEC = "#111111"      # 理论峰值：黑色实线
COLOR_MEASURED = "#0b7285"  # 实测可达峰值：青色虚线
COLOR_ROOF = "#5f3dc4"      # roofline：紫色点划线（与灰->红 step 色带区分开）

CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK",
    "Noto Sans SC", "Noto Sans TC", "Source Han Sans SC", "Source Han Sans CN",
    "Source Han Sans", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Sarasa Gothic SC",
    "LXGW WenKai", "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
    "AR PL UMing CN", "AR PL UKai CN", "Droid Sans Fallback", "Tahoma",
]
CJK_NAME_HINTS = ("cjk", "han", "hei", "song", "kai", "ming", "wenquanyi", "noto sans sc",
                  "noto sans tc", "yahei", "pingfang", "sarasa", "lxgw")

# --------------------------------------------------------------------------------------
# 文案（zh / en 双份；探测不到 CJK 字体就用 en）
# --------------------------------------------------------------------------------------
_STRINGS = {
    "xlabel": ("批大小 M（log2 刻度）", "Batch size M (log2 scale)"),
    "ylabel": ("实测带宽 (GB/s)", "Measured bandwidth (GB/s)"),
    "spec_line": ("HBM3 理论峰值 4.0 TB/s", "HBM3 spec peak 4.0 TB/s"),
    "measured_line": ("实测可达峰值 {v:.0f} GB/s（peak_bw_probe）",
                      "Measured achievable peak {v:.0f} GB/s (peak_bw_probe)"),
    "roof_line": ("roofline 计算上限（FP8 {tf:.1f} TFLOPS）",
                  "Compute roofline cap (FP8 {tf:.1f} TFLOPS)"),
    "legend_title": ("技术步骤（累积叠加）", "Step (cumulative)"),
    "ref_lines_title": ("参考线", "Reference lines"),
    "all_title": ("decode GEMM 实测带宽 vs 批大小 M（5 模型 × 6 技术步骤）",
                  "decode GEMM effective bandwidth vs batch size M (models x steps)"),
    "model_title": ("{display} | N={N}, K={K}", "{display}  N={N}, K={K}"),
    "note_bw": ("带宽口径：分子 = 实测 DRAM 流量 = N·K（权重一次）+ M·K（激活一次，CTA 列重读由 L2 吸收）+ 2·M·N（输出）+ split-K ws；"
                "分母 = 对应协议 p50 延迟。",
                "Bandwidth = (N*K + M*K + 2*M*N) bytes / p50 latency of the step's protocol."),
    "note_proto": ("曲线口径：S0 = B2B（back-to-back x90），S1–S6 = PDL（B2B + "
                   "ProgrammaticStreamSerialization）；均为冷权重旋转 buffer。",
                   "Curve protocol: S0 = B2B (back-to-back x90), S1-S6 = PDL; rotating cold weights."),
    "note_roof": ("roofline：t_floor = 2·M·N·K / {tf:.1f} TFLOPS（H20 FP8 实测峰值，CONTRACT §8 写死常数）；"
                  "BW_cap(M) = min({spec:.0f}, 字节数 / t_floor)。",
                  "Roofline: t_floor = 2*M*N*K / {tf:.1f} TFLOPS (H20 FP8 measured peak, CONTRACT S8 constant); "
                  "BW_cap(M) = min({spec:.0f}, bytes / t_floor)."),
    "note_peak_missing": ("警告：results/peak_bw.json 缺失 → 未画“实测可达峰值”虚线（不编造数据）。",
                          "WARNING: results/peak_bw.json missing -> measured-peak dashed line not drawn (no fabricated data)."),
    "note_src": ("数据：{csv}（{rows} 行；{models} 模型 × {steps} step × {ms} 个 M） | 绘图时间：{ts}",
                 "Source: {csv} ({rows} rows; {models} models x {steps} steps x {ms} M values)  plotted: {ts}"),
    "note_fallback": ("† 该 step 缺标准协议数据，已回退到 {p}（见 stdout 警告）。",
                      "dagger: canonical protocol missing for this step, fell back to {p} (see stdout)."),
    "note_legend_cell": ("图例与口径说明", "Legend & conventions"),
    "demo_banner": ("【DEMO 合成数据 · 仅用于渲染自测，非实测结果】",
                    "[DEMO synthetic data - rendering self-test only, NOT measured results]"),
    "demo_short": ("DEMO 合成数据，非实测", "DEMO synthetic data (not measured)"),
    "step_suffix": ("{sid} {name}（{proto}）", "{sid} {name} ({proto})"),
    "no_data": ("没有可画的数据。先跑 bench（bench/bench_decode.cu）产出 results/bench_*.csv，"
                "或用 --demo 生成假 CSV 自测渲染。",
                "No plottable data. Run the bench to produce results/bench_*.csv, or use --demo."),
    "best_at": ("最佳：{sid} {name} @ M={m}，{bw:.0f} GB/s（理论峰值的 {pct:.1f}%）",
                "best: {sid} {name} @ M={m}, {bw:.0f} GB/s ({pct:.1f}% of spec peak)"),
}


class L:
    """按语言取文案。"""

    def __init__(self, zh: bool) -> None:
        self.zh = zh

    def __call__(self, key: str, **kw) -> str:
        zh, en = _STRINGS[key]
        s = zh if self.zh else en
        return s.format(**kw) if kw else s


def warn(msg: str) -> None:
    """警告统一走 stdout（harness/CI 一般只收一路，且本项目的警告都是「口径说明」级别）。"""
    print(f"[plot.py][WARN] {msg}", flush=True)


def info(msg: str) -> None:
    print(f"[plot.py] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# 数据层
# --------------------------------------------------------------------------------------
@dataclass
class Row:
    model: str
    N: int
    K: int
    M: int
    step: int
    step_name: str
    protocol: str
    lat_us: float
    bw_gbps: float
    pct_spec: float
    rel_l2: float
    passed: bool
    config: str
    src: str = ""
    src_idx: int = 0     # 来源文件序号（跨文件覆盖语义用）


REQUIRED_COLS = ("model", "n", "k", "m", "step", "protocol")
_TRUE = {"1", "true", "t", "yes", "y", "pass", "ok", "1.0"}


def moved_bytes(N: int, K: int, M: int) -> float:
    """CONTRACT §6 带宽分子：权重 FP8 + 激活 FP8 + 输出 BF16。"""
    return float(N) * K + float(M) * K + 2.0 * M * N


def bw_from_latency(N: int, K: int, M: int, lat_us: float) -> float:
    """GB/s（1e9 进制，与 pct_of_spec_peak = bw/4000 一致）。"""
    if not lat_us or lat_us <= 0 or not math.isfinite(lat_us):
        return float("nan")
    return moved_bytes(N, K, M) / lat_us / 1.0e3


def bw_cap_gbps(M, N: int, K: int, tflops: float = COMPUTE_PEAK_TFLOPS,
                spec: float = SPEC_PEAK_GBPS) -> np.ndarray:
    """roofline 等效带宽上限（GB/s）。

    t_floor = 2*M*N*K / (tflops*1e12) 秒；BW_cap = min(spec, bytes / t_floor / 1e9)。
    小 M 时 bytes/t_floor >> spec → 被 4000 截住（访存瓶颈区）；
    大 M 时 t_floor ∝ M 增长更快 → 上限 ∝ 1/M 下降（计算瓶颈区）。
    """
    M = np.asarray(M, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = 2.0 * M * N * K / (tflops * 1e12)
        cap = (N * K + M * K + 2.0 * M * N) / t_floor / 1e9
    cap = np.where(np.isfinite(cap), cap, spec)
    return np.minimum(spec, cap)


def _to_float(s: str, default=float("nan")) -> float:
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return default


def _to_int(s: str, default=-1) -> int:
    try:
        return int(round(float(str(s).strip())))
    except (TypeError, ValueError):
        return default


def _dram_bytes(N: int, K: int, M: int, config: str) -> float:
    """实测 DRAM 流量估计：权重 + 激活按 CTA 列重读 + 输出 + split-K workspace(fp32 读写)。"""
    import math as _m
    bm = 64
    mk = config.find("BM=")
    if mk >= 0:
        try:
            bm = int(config[mk + 3:].split(";")[0])
        except ValueError:
            bm = 64
    # 激活 M*K <= 2MB 全被 L2(60MB) 吸收，CTA 列重读不出 DRAM -> 只计一次
    b = N * K + M * K + 2 * M * N
    if "SPLITK=1" in config:
        b += 2 * 4 * M * N          # fp32 workspace 写+读，真 DRAM 流量
    return float(b)


def _recompute_dram_bw(rows) -> None:
    for r in rows:
        if r.lat_us <= 0:
            continue
        r.bw_gbps = _dram_bytes(r.N, r.K, r.M, r.config) / r.lat_us / 1000.0
        r.pct_spec = 100.0 * r.bw_gbps / 4000.0


def load_rows(paths: Sequence[Path], include_failed: bool = False) -> tuple[list[Row], list[str]]:
    """读 CSV（stdlib csv，无 pandas 依赖）。列名大小写/空白不敏感，容忍多余列。"""
    rows: list[Row] = []
    warns: list[str] = []
    for _src, path in enumerate(paths):
        if not path.exists():
            warns.append(f"CSV 不存在：{path}")
            continue
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                warns.append(f"CSV 为空：{path}")
                continue
            hdr = {(h or "").strip().lower(): h for h in reader.fieldnames}
            missing = [c for c in REQUIRED_COLS if c not in hdr]
            if missing:
                warns.append(f"{path.name} 缺必需列 {missing}（CONTRACT §6 schema），已跳过该文件")
                continue

            def g(rec: dict, col: str, default=""):
                return rec.get(hdr.get(col, ""), default) if col in hdr else default

            n_bad = n_fail = n_nobw = 0
            n_bwdev = 0
            max_dev = 0.0
            for rec in reader:
                model = str(g(rec, "model") or "").strip()
                N, K = _to_int(g(rec, "n")), _to_int(g(rec, "k"))
                M, step = _to_int(g(rec, "m")), _to_int(g(rec, "step"))
                if not model or N <= 0 or K <= 0 or M <= 0 or step < 0:
                    n_bad += 1
                    continue
                proto = str(g(rec, "protocol") or "").strip().upper()
                lat = _to_float(g(rec, "latency_us_p50"))
                bw = _to_float(g(rec, "bandwidth_gbps"))
                if not math.isfinite(bw) or bw <= 0:
                    bw2 = bw_from_latency(N, K, M, lat)
                    if math.isfinite(bw2):
                        bw, n_nobw = bw2, n_nobw + 1
                    else:
                        n_bad += 1
                        continue
                # 交叉校验：CSV 的 bandwidth_gbps 是否等于 (N*K+M*K+2*M*N)/p50（CONTRACT §6 口径）
                exp = bw_from_latency(N, K, M, lat)
                if math.isfinite(exp) and exp > 0:
                    dev = abs(bw - exp) / exp
                    if dev > 0.02:
                        n_bwdev += 1
                        max_dev = max(max_dev, dev)
                pct = _to_float(g(rec, "pct_of_spec_peak"))
                if not math.isfinite(pct):
                    pct = bw / SPEC_PEAK_GBPS
                elif pct > 1.5:      # 有的 harness 存百分数（85.3）而非小数（0.853）
                    pct /= 100.0
                passed_raw = str(g(rec, "pass", "")).strip().lower()
                passed = (passed_raw in _TRUE) if passed_raw else True
                if not passed:
                    n_fail += 1
                    if not include_failed:
                        continue
                rows.append(Row(
                    model=model, N=N, K=K, M=M, step=step, src_idx=_src,
                    step_name=str(g(rec, "step_name") or STEP_NAMES.get(step, f"step{step}")).strip(),
                    protocol=proto or "?", lat_us=lat, bw_gbps=bw, pct_spec=pct,
                    rel_l2=_to_float(g(rec, "rel_l2")), passed=passed,
                    config=str(g(rec, "config") or "").strip(), src=path.name,
                ))
            if n_bad:
                warns.append(f"{path.name}: {n_bad} 行字段不可解析，已跳过")
            if n_nobw:
                warns.append(f"{path.name}: {n_nobw} 行缺 bandwidth_gbps，已按 bytes/p50 重算")
            if n_bwdev:
                warns.append(f"{path.name}: {n_bwdev} 行的 bandwidth_gbps 与 (N·K+M·K+2·M·N)/p50 "
                             f"相差 >2%（最大 {max_dev*100:.1f}%）→ 口径可能不一致，图上仍用 CSV 的值")
            if n_fail:
                verb = "仍画入" if include_failed else "已排除（correctness fail）"
                warns.append(f"{path.name}: {n_fail} 行 pass=false，{verb}")
    return rows, warns


@dataclass
class ModelData:
    model: str
    display: str
    N: int
    K: int
    layer: str = ""
    # step -> (Ms, bws, protocol_used, step_name, fallback: bool)
    curves: dict[int, tuple[np.ndarray, np.ndarray, str, str, bool]] = field(default_factory=dict)
    nrows: int = 0

    @property
    def ms(self) -> list[int]:
        out: set[int] = set()
        for Ms, *_ in self.curves.values():
            out.update(int(m) for m in Ms)
        return sorted(out)

    def best(self) -> tuple[int, str, int, float] | None:
        best = None
        for step, (Ms, bws, proto, name, _fb) in self.curves.items():
            for m, b in zip(Ms, bws):
                if best is None or b > best[3]:
                    best = (step, name, int(m), float(b))
        return best


def canon_protocol(step: int) -> str:
    return CANON_PROTOCOL.get(step, DEFAULT_PROTOCOL)


def build_models(rows: list[Row], model_order: Sequence[str], display_of,
                 steps_wanted: Sequence[int] | None) -> tuple[list[ModelData], list[str]]:
    """按 (model, step, protocol) 归并成曲线；协议按 CONTRACT §6 口径挑，缺失才回退并标注。"""
    warns: list[str] = []
    by_model: dict[str, list[Row]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)

    out: list[ModelData] = []
    for model in model_order:
        rws = by_model.get(model)
        if not rws:
            continue
        # N/K 一致性检查（同一模型只应有一组形状）
        shapes = {}
        for r in rws:
            shapes[(r.N, r.K)] = shapes.get((r.N, r.K), 0) + 1
        (N, K), _cnt = max(shapes.items(), key=lambda kv: kv[1])
        if len(shapes) > 1:
            warns.append(f"{model}: CSV 里出现多组 (N,K) {sorted(shapes)}，按最多的 N={N},K={K} 画")
        md = ModelData(model=model, display=display_of(model), N=N, K=K,
                       layer="", nrows=len(rws))
        steps = sorted({r.step for r in rws})
        if steps_wanted:
            steps = [s for s in steps if s in steps_wanted]
        for step in steps:
            srows = [r for r in rws if r.step == step]
            protos = {r.protocol for r in srows}
            want = canon_protocol(step)
            order = FALLBACK_ORDER.get(step, FALLBACK_ORDER[None])
            order = [want] + [p for p in order if p != want] + \
                    [p for p in sorted(protos) if p not in order]
            chosen = next((p for p in order if p in protos), None)
            if chosen is None:
                warns.append(f"{model} step{step}: 无可用协议数据，跳过")
                continue
            if chosen != want:
                warns.append(f"{model} step{step}({STEP_NAMES.get(step,'?')}): 缺标准协议 {want}，"
                             f"回退到 {chosen}（图上标 †）")
            sel = [r for r in srows if r.protocol == chosen]
            # 同一 (model,M,step,protocol) 可能有多行（--tune 扫 config）→ 取带宽最高的一行
            per_m: dict[int, Row] = {}
            dupes = 0
            for r in sel:
                if r.M in per_m:
                    dupes += 1
                    cur = per_m[r.M]
                    # 跨文件：后文件覆盖前文件（与表格口径一致）；同文件：取带宽最高（canonical vs alt）
                    if r.src_idx > cur.src_idx or (
                            r.src_idx == cur.src_idx and r.bw_gbps > cur.bw_gbps):
                        per_m[r.M] = r
                else:
                    per_m[r.M] = r
            if dupes:
                warns.append(f"{model} step{step} {chosen}: {dupes} 组重复 (M,协议)，已取带宽最高的 config")
            ms = sorted(per_m)
            md.curves[step] = (
                np.array(ms, dtype=float),
                np.array([per_m[m].bw_gbps for m in ms], dtype=float),
                chosen,
                per_m[ms[0]].step_name or STEP_NAMES.get(step, f"step{step}"),
                chosen != want,
            )
        if md.curves:
            out.append(md)
        else:
            warns.append(f"{model}: 没有可用曲线，跳过")
    return out, warns


# --------------------------------------------------------------------------------------
# peak_bw.json / models.json
# --------------------------------------------------------------------------------------
_BW_KEY = re.compile(r"(gbps|bandwidth|bw|tb_?ps|tbps)", re.I)
_SKIP_KEY = re.compile(r"(spec|theor|note|method|desc|name|label|config|unit)", re.I)


def read_measured_peak(path: Path) -> tuple[float | None, str, str]:
    """从 peak_bw.json 抽实测可达峰值 GB/s。

    peak_bw.json 的具体 key 由 tools/peak_bw_probe 决定，这里做保守的递归搜索：
    只看 key 含 gbps/bandwidth/bw/tbps 且不含 spec/theor/note 的数值叶子；
    优先 key 含 best，否则取最大值。返回 (gbps, 来源 key 路径, 方法说明)。
    找不到就返回 (None, "", "")，调用方跳过虚线并警告——绝不编数。
    """
    if not path.exists():
        return None, "", ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warn(f"peak_bw.json 解析失败（{exc}），不画实测峰值虚线")
        return None, "", ""

    found: list[tuple[str, float]] = []

    def walk(node, prefix: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if _BW_KEY.search(str(k)) and not _SKIP_KEY.search(str(k)):
                        found.append((key, float(v)))
                else:
                    walk(v, key)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{prefix}[{i}]")

    walk(obj, "")
    note = ""
    if isinstance(obj, dict):
        for k in ("note", "method", "measured_note", "description", "best_note"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                note = v.strip()
                break
    if not found:
        warn(f"{path.name} 里找不到带宽数值字段（识别到的 key 都不像 bandwidth），不画实测峰值虚线")
        return None, "", note

    best = [f for f in found if "best" in f[0].lower()]
    pool = best or found
    # TB/s 量级的字段（值 < 100）换算成 GB/s
    def norm(item: tuple[str, float]) -> float:
        key, val = item
        return val * 1000.0 if ("tbps" in key.lower() or "tb_s" in key.lower()) and val < 100 else val

    key, val = max(pool, key=lambda it: norm(it))
    gbps = norm((key, val))
    if not (0 < gbps <= SPEC_PEAK_GBPS * 1.5):
        warn(f"{path.name} 的 {key}={gbps:.1f} GB/s 超出合理范围 (0, {SPEC_PEAK_GBPS*1.5:.0f}]，"
             f"疑似单位错误，不画实测峰值虚线")
        return None, key, note
    if not best and len(found) > 1:
        warn(f"{path.name} 无 best 字段，从 {len(found)} 个带宽字段里取最大值 {key}={gbps:.1f} GB/s")
    return gbps, key, note


def load_models_json(path: Path) -> tuple[dict[str, dict], list[str]]:
    """读 models/models.json → ({id: {...}}, 顺序)。缺失则返回空（用 CSV 里的 model id）。"""
    if not path.exists():
        return {}, []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warn(f"models.json 解析失败（{exc}），标题退回 model id")
        return {}, []
    out: dict[str, dict] = {}
    order: list[str] = []
    for m in obj.get("models", []) or []:
        mid = str(m.get("id", "")).strip()
        if not mid:
            continue
        out[mid] = m
        order.append(mid)
    return out, order


# --------------------------------------------------------------------------------------
# 字体探测
# --------------------------------------------------------------------------------------
def _iter_bundled_fonts() -> Iterable[Path]:
    d = SCRIPT_DIR / "fonts"
    if d.is_dir():
        for ext in ("*.otf", "*.ttf", "*.ttc", "*.OTF", "*.TTF", "*.TTC"):
            yield from sorted(d.glob(ext))


def probe_cjk_font(explicit: str | None) -> tuple[str | None, str | None, str]:
    """探测可用 CJK 字体。返回 (font family, font file, 来源描述)。找不到返回 (None, None, "")。"""
    # 1) CLI / 环境变量 / 捆绑目录：直接注册字体文件
    candidates: list[Path] = []
    for raw in (explicit, os.environ.get("DECODE_GEMM_CJK_FONT")):
        if raw:
            p = Path(raw).expanduser()
            candidates.append(p)
    candidates.extend(_iter_bundled_fonts())
    for p in candidates:
        if p.is_file():
            try:
                font_manager.fontManager.addfont(str(p))
                name = font_manager.FontProperties(fname=str(p)).get_name()
                return name, str(p), f"font file {p}"
            except Exception as exc:  # noqa: BLE001
                warn(f"字体文件 {p} 注册失败：{exc}")
        elif explicit and p == Path(explicit).expanduser():
            warn(f"--font-file 指定的字体不存在：{p}")

    # 2) 系统已装字体：按候选名精确匹配，再按名字里的 CJK 线索模糊匹配
    installed = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in installed:
            return name, installed[name], f"system font '{name}'"
    for name, fname in sorted(installed.items()):
        low = name.lower()
        if any(h in low for h in CJK_NAME_HINTS):
            return name, fname, f"system font '{name}'"
    return None, None, ""


def missing_glyphs(font_file: str, texts: Iterable[str]) -> tuple[list[str], list[str]]:
    """逐字符校验 glyph 覆盖，防止出方块。

    返回 (cjk_missing, punct_missing)：
      cjk_missing   ord >= 0x2E80 的汉字缺字形 → 必须退英文（DejaVu 没有汉字兜底）
      punct_missing 0xA0..0x2E7F 的标点缺字形 → 只警告（font.sans-serif 第二顺位 DejaVu 会兜底）
    """
    from matplotlib.ft2font import FT2Font

    try:
        ft = FT2Font(font_file)
    except Exception as exc:  # noqa: BLE001
        warn(f"打开字体 {font_file} 失败：{exc}")
        return ["<font-unreadable>"], []
    cjk: list[str] = []
    punct: list[str] = []
    seen: set[str] = set()
    for t in texts:
        for ch in t:
            if ord(ch) < 0xA0 or ch in seen:
                continue
            seen.add(ch)
            if ft.get_char_index(ord(ch)) == 0:
                (cjk if ord(ch) >= 0x2E80 else punct).append(ch)
    return cjk, punct


# --------------------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------------------
def sanitize_labels(md_list: Sequence[ModelData], zh: bool) -> list[str]:
    """非中文模式下，标题里来自 models.json 的 display/layer 可能含中文 → 换成 ASCII 安全文本。

    返回被改写过的说明（会进 stdout 警告）。中文字体可用时不动（中文标题更好看）。
    """
    notes: list[str] = []
    if zh:
        return notes
    for md in md_list:
        if not md.display.isascii():
            notes.append(f"{md.model}: display '{md.display}' 含非 ASCII → 英文模式改用 model id")
            md.display = md.model
        if md.layer and not md.layer.isascii():
            notes.append(f"{md.model}: layer '{md.layer}' 含非 ASCII → 英文模式标题省略层名")
            md.layer = ""
    return notes


def step_style(step: int, n_steps: int) -> tuple:
    t = 0.0 if n_steps <= 1 else step / (n_steps - 1)
    return _STEP_CMAP(t), _STEP_MARKERS[step % len(_STEP_MARKERS)], 1.5 + 0.16 * step


def setup_mpl(zh: bool, font_name: str | None) -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.unicode_minus": False,   # 用 ASCII '-'，避免 CJK 字体缺 U+2212 出方块
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    if font_name:
        # family 必须是「列表」才会启用逐字形 fallback（sans-serif 列表不生效，实测）：
        # CJK 字体缺的标点/符号由 DejaVu Sans 兜底，避免出方块
        plt.rcParams["font.family"] = [font_name, "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]


def style_axes(ax: Axes, L_: L, m_lo: float, m_hi: float, y_top: float | None,
               logy: bool = False) -> None:
    ax.set_xscale("log", base=2)
    ticks = [t for t in M_TICKS if m_lo * 0.9 <= t <= m_hi * 1.4] or M_TICKS
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(m_lo / 1.18, m_hi * 1.22)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(30, max(9000.0, (y_top or 4400) * 1.6))
        ax.grid(True, which="minor", alpha=0.14, ls=":")
    else:
        ax.set_ylim(0, y_top if y_top else SPEC_PEAK_GBPS * 1.10)
    ax.set_xlabel(L_("xlabel"), fontsize=10)
    ax.set_ylabel(L_("ylabel"), fontsize=10)


def draw_reference_lines(ax: Axes, L_: L, m_grid: np.ndarray, cap: np.ndarray | None,
                         measured: float | None, spec: float, tflops: float,
                         handles: list, labels: list, annotate: bool = True) -> None:
    """三条参考线；handles/labels 就地追加（汇总图共用同一份图例）。"""
    if cap is not None:
        cap_draw = np.where(cap < spec - 0.5, cap, np.nan)   # 平段=spec 线本身，不重复描
        h = ax.plot(m_grid, cap_draw, color=COLOR_ROOF, ls="-.", lw=1.9, zorder=2.6)[0]
        handles.append(h)
        labels.append(L_("roof_line", tf=tflops))
    h = ax.axhline(spec, color=COLOR_SPEC, ls="-", lw=1.5, zorder=2.0)  # axhline 返回单个 Line2D
    handles.append(h)
    labels.append(L_("spec_line"))
    if annotate:
        ax.text(0.995, spec, L_("spec_line"), transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=10.5, color=COLOR_SPEC, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.2))
    if measured:
        h = ax.axhline(measured, color=COLOR_MEASURED, ls="--", lw=1.5, zorder=2.1)
        handles.append(h)
        labels.append(L_("measured_line", v=measured))
        if annotate:
            ax.text(0.995, measured, L_("measured_line", v=measured),
                    transform=ax.get_yaxis_transform(), ha="right", va="top",
                    fontsize=10.5, color=COLOR_MEASURED, zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.2))


def draw_steps(ax: Axes, L_: L, md: ModelData, steps: Sequence[int],
               handles: list, labels: list, legend_label: bool = True) -> None:
    n = max(1, len(steps) - 1)
    for i, step in enumerate(steps):
        Ms, bws, proto, name, fb = md.curves[step]
        color, marker, lw = step_style(i, len(steps))
        sid = f"S{step}"
        # 图例只写技术名：阶梯是累积的（每步默认包含前一步），计时协议口径见 README/RESULTS 正文
        lab = f"{sid} {PLAIN_STEP.get(step, name)}"
        if fb:
            lab += " †"
        ax.plot(Ms, bws, color=color, marker=marker, ms=4.6, mew=0.5, mec="white",
                lw=lw, ls="-", zorder=10 + step, label=lab if legend_label else None)
        if legend_label:
            handles.append(Line2D([], [], color=color, marker=marker, ms=4.6, mew=0.5,
                                  mec="white", lw=lw))
            labels.append(lab)


def footnote_lines(L_: L, args, md_list: Sequence[ModelData], csv_names: Sequence[str],
                   measured: float | None, peak_key: str, peak_note: str,
                   warns: Sequence[str], lang_desc: str) -> list[str]:
    steps = sorted({s for md in md_list for s in md.curves})
    ms = sorted({int(m) for md in md_list for m in md.ms})
    lines = [
        L_("note_bw"),
        L_("note_proto"),
        L_("note_roof", tf=args.compute_tflops, spec=args.spec_peak),
        L_("note_src", csv=", ".join(csv_names), rows=sum(m.nrows for m in md_list),
           models=len(md_list), steps=len(steps), ms=len(ms),
           ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
    ]
    if measured:
        lines.append(f"peak_bw.json: {peak_key} = {measured:.1f} GB/s"
                     + (f" | ({peak_note})" if peak_note else ""))
    else:
        lines.append(L_("note_peak_missing"))
    if any(fb for md in md_list for (_m, _b, _p, _n, fb) in md.curves.values()):
        lines.append(L_("note_fallback", p="/".join(
            sorted({p for md in md_list for (_M, _B, p, _n, fb) in md.curves.values() if fb}))))
    lines.append(f"labels: {lang_desc} | plot: analysis/plot.py (CONTRACT §8)")
    if warns:
        lines.append(f"warnings: {len(warns)} (see stdout)")
    return lines


def _vis_lines(ln: str, fig: Figure, fontsize: float) -> int:
    """估算一行图注 wrap 后的视觉行数（CJK 全角、latin 半角宽估）。"""
    w_pt = fig.get_size_inches()[0] * 72.0 * 0.97
    cjk = sum(1 for ch in ln if ord(ch) > 0x2E80)
    lat = len(ln) - cjk
    width = cjk * fontsize * 1.02 + lat * fontsize * 0.55
    return max(1, math.ceil(width / w_pt))


def footnote_layout(fig: Figure, lines, fontsize: float) -> tuple[float, float, float]:
    """按「磅」算图注需要的下边距 / 首行 y / 行距（figure 分数），保证任何图高都不重叠、不出界。
    lines 可以是行数(int, 旧接口)或字符串序列(按 wrap 视觉行数计)。"""
    if isinstance(lines, int):
        n_vis = lines
    else:
        n_vis = sum(_vis_lines(ln, fig, fontsize) for ln in lines)
    h_pt = fig.get_size_inches()[1] * 72.0
    line_pt = fontsize * 1.5
    pad_pt = 10.0          # 画布最底下留白
    gap_pt = 34.0          # 坐标区下沿 -> 图注首行：要装得下 tick label + xlabel
    bottom = min(0.45, (pad_pt + n_vis * line_pt + gap_pt) / h_pt)
    y0 = (pad_pt + max(0, n_vis - 1) * line_pt) / h_pt
    return bottom, y0, line_pt / h_pt


def add_footnote(fig: Figure, lines: Sequence[str], y0: float, dy: float,
                 fontsize: float = 7.2) -> None:
    """把口径说明排在坐标区下方；长行 wrap 后按视觉行数推进 y，避免重叠。"""
    y = y0
    for ln in lines:
        fig.text(0.012, y, ln, fontsize=fontsize, color="#333333",
                 ha="left", va="top", wrap=True)
        y -= _vis_lines(ln, fig, fontsize) * dy


def plot_model_figure(md: ModelData, args, L_: L, measured: float | None, peak_key: str,
                      peak_note: str, warns: Sequence[str], lang_desc: str,
                      demo: bool, y_top: float, out_path: Path) -> None:
    steps = sorted(md.curves)
    all_m = md.ms or [1, 128]
    m_lo, m_hi = min(all_m), max(all_m)
    m_grid = np.geomspace(max(m_lo, 1), max(m_hi, 2), 260)
    cap = None if args.no_roofline else bw_cap_gbps(m_grid, md.N, md.K, args.compute_tflops,
                                                    args.spec_peak)

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    handles: list = []
    labels: list = []
    draw_reference_lines(ax, L_, m_grid, cap, measured, args.spec_peak, args.compute_tflops,
                         handles, labels, annotate=True)
    draw_steps(ax, L_, md, steps, handles, labels, legend_label=True)
    style_axes(ax, L_, m_lo, m_hi, y_top, logy=args.logy)

    title = L_("model_title", display=md.display, N=md.N, K=md.K)
    if md.layer:
        title += f"\n{md.layer}"
    ax.set_title(title, fontsize=16, pad=26 if demo else 10)
    if demo:
        fig.text(0.5, 0.992, L_("demo_banner"), ha="center", va="top", fontsize=9.2,
                 color="#b00020", weight="bold")

    b = md.best()
    if b:
        ax.text(0.015, 0.03, L_("best_at", sid=f"S{b[0]}", name=b[1], m=b[2], bw=b[3],
                                pct=100 * b[3] / args.spec_peak),
                transform=ax.transAxes, fontsize=10.5, color="#444444", ha="left", va="bottom")

    # 图例放右侧外，避免压住曲线
    leg = ax.legend(handles, labels, title=None, loc="upper left", bbox_to_anchor=(1.015, 1.0),
                    fontsize=11.5, labelspacing=0.42, borderpad=0.6, frameon=True)
    leg.get_frame().set_linewidth(0.6)

    fn = footnote_lines(L_, args, [md], sorted(set(args._csv_names)), measured, peak_key,
                        peak_note, warns, lang_desc)
    bottom, y0, dy = footnote_layout(fig, fn, 7.0)
    fig.subplots_adjust(left=0.088, right=0.735, top=0.845 if demo else 0.905, bottom=bottom)
    add_footnote(fig, fn, y0=y0, dy=dy, fontsize=7.0)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_summary_figure(md_list: Sequence[ModelData], args, L_: L, measured: float | None,
                        peak_key: str, peak_note: str, warns: Sequence[str], lang_desc: str,
                        demo: bool, y_top: float, out_path: Path) -> None:
    n = len(md_list)
    ncols = 1 if n == 1 else 3
    nrows = max(1, math.ceil((n + 1) / ncols))   # 多留一格放图例（CONTRACT §8：2×3，第 6 格图例）
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.7 * ncols, 5.1 * nrows + (2.3 if args.footnote else 0.4)),
                             squeeze=False)
    handles: list = []
    labels: list = []
    steps_all = sorted({s for md in md_list for s in md.curves})

    for idx, md in enumerate(md_list):
        ax = axes[idx // ncols][idx % ncols]
        steps = sorted(md.curves)
        all_m = md.ms or [1, 128]
        m_lo, m_hi = min(all_m), max(all_m)
        m_grid = np.geomspace(max(m_lo, 1), max(m_hi, 2), 260)
        cap = None if args.no_roofline else bw_cap_gbps(m_grid, md.N, md.K, args.compute_tflops,
                                                        args.spec_peak)
        if idx == 0:
            draw_reference_lines(ax, L_, m_grid, cap, measured, args.spec_peak,
                                 args.compute_tflops, handles, labels, annotate=False)
            draw_steps(ax, L_, md, steps, handles, labels, legend_label=True)
        else:
            draw_reference_lines(ax, L_, m_grid, cap, measured, args.spec_peak,
                                 args.compute_tflops, [], [], annotate=False)
            draw_steps(ax, L_, md, steps, [], [], legend_label=False)
        style_axes(ax, L_, m_lo, m_hi, y_top, logy=args.logy)
        ax.set_title(f"{md.display} | N={md.N}, K={md.K}", fontsize=14, pad=6)
        ax.tick_params(labelsize=12)
        ax.set_xlabel(L_("xlabel"), fontsize=13)
        ax.set_ylabel(L_("ylabel"), fontsize=13)

    # 空余格子关掉坐标轴
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # 图例格：优先用最后一格（n=5, ncols=3 → 第 6 格）
    leg_idx = n if n < nrows * ncols else nrows * ncols - 1
    leg_ax = axes[leg_idx // ncols][leg_idx % ncols]
    leg_ax.axis("off")
    leg = leg_ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.02),
                        fontsize=12, title=L_("legend_title"), title_fontsize=13,
                        labelspacing=0.5, borderpad=0.7, frameon=True, ncol=1,
                        handlelength=2.6)
    leg.get_frame().set_linewidth(0.6)
    if demo:
        leg_ax.text(0.5, -0.04, L_("demo_short"), transform=leg_ax.transAxes, ha="center",
                    va="top", fontsize=8.6, color="#b00020", weight="bold")

    fig.suptitle(L_("all_title"), fontsize=19, y=0.995)
    if demo:
        fig.text(0.5, 0.962, L_("demo_banner"), ha="center", va="top", fontsize=12,
                 color="#b00020", weight="bold")
    top = (0.905 if demo else 0.93) if nrows >= 2 else (0.855 if demo else 0.895)
    if args.footnote:
        fn = footnote_lines(L_, args, md_list, sorted(args._csv_names), measured, peak_key,
                            peak_note, warns, lang_desc)
        bottom, y0, dy = footnote_layout(fig, fn, 9.5)
        fig.subplots_adjust(left=0.062, right=0.985, top=top,
                            bottom=bottom, hspace=0.36, wspace=0.24)
        add_footnote(fig, fn, y0=y0, dy=dy, fontsize=9.5)
    else:
        fig.subplots_adjust(left=0.075, right=0.985, top=top,
                            bottom=0.10, hspace=0.34, wspace=0.24)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_steps(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def find_latest_csv(results_dir: Path) -> list[Path]:
    cands = [Path(p) for p in glob.glob(str(results_dir / "bench_*.csv"))]
    # demo_*.csv 不参与正式图（文件名不以 bench_ 开头，天然排除）
    return sorted(cands, key=lambda p: (p.stat().st_mtime, p.name))[-1:]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="decode_gemm bandwidth-vs-M plots (CONTRACT §8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：python3 analysis/plot.py --csv results/bench_20260913T000000Z.csv\n"
               "    python3 analysis/plot.py --demo   # 无真数据时自测渲染")
    ap.add_argument("--csv", nargs="+", default=None,
                    help="bench CSV 路径（可多个，后面的覆盖前面的同 key 行）；默认取 results/bench_*.csv 最新一份")
    ap.add_argument("--out", default=str(RESULTS_DIR), help="输出目录（默认 results/）")
    ap.add_argument("--fig-prefix", default="fig", help="输出文件名前缀（默认 fig → fig_bw_vs_M_*.png）")
    ap.add_argument("--demo", action="store_true",
                    help="调 make_demo_csv.py 生成合成 CSV 并画 demo 图（fig_demo_*）")
    ap.add_argument("--models", default=None, help="只画这些模型（逗号分隔 id）")
    ap.add_argument("--steps", default=None, help="只画这些 step，如 0-6 或 0,3,6")
    ap.add_argument("--peak-json", default=str(PEAK_BW_JSON), help="实测峰值 JSON（默认 results/peak_bw.json）")
    ap.add_argument("--models-json", default=str(MODELS_JSON), help="models/models.json（取 display/layer/顺序）")
    ap.add_argument("--spec-peak", type=float, default=SPEC_PEAK_GBPS, help="理论峰值 GB/s（默认 4000，用户指定不许改）")
    ap.add_argument("--compute-tflops", type=float, default=COMPUTE_PEAK_TFLOPS,
                    help="roofline 用的 FP8 实测峰值 TFLOPS（默认 282.7，CONTRACT §8）")
    ap.add_argument("--no-roofline", action="store_true", help="不画 roofline 上限曲线")
    ap.add_argument("--no-measured-peak", action="store_true", help="不画实测峰值虚线")
    ap.add_argument("--include-failed", action="store_true", help="连 correctness fail 的行一起画（默认排除）")
    ap.add_argument("--lang", choices=("auto", "zh", "en"), default="auto",
                    help="标签语言；auto = 探测到可用 CJK 字体就中文，否则英文")
    ap.add_argument("--font-file", default=None, help="指定 CJK 字体文件（.ttf/.otf/.ttc）")
    ap.add_argument("--dpi", type=int, default=300, help="输出 dpi（默认 300）")
    ap.add_argument("--footnote", action="store_true",
                    help="在图底部排口径小字（默认关闭：只留干净图）")
    ap.add_argument("--bw-mode", choices=["dram", "useful"], default="dram",
                    help="y 轴带宽口径：dram=实测 DRAM 流量/延迟（默认）；useful=有用字节/延迟（旧口径）")
    ap.add_argument("--only-summary", action="store_true", help="只出一张汇总图，不出每模型单图")
    ap.add_argument("--logy", action="store_true", help="y 轴改 log（大 M 段曲线挤在一起时用）")
    ap.add_argument("--no-summary", action="store_true", help="只出每模型图，不出汇总图")
    ap.add_argument("--quiet", action="store_true", help="少打印")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 定位 CSV ----
    csv_paths: list[Path] = []
    if args.demo:
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            import make_demo_csv as mdc
        except ImportError as exc:
            warn(f"导入 make_demo_csv 失败（{exc}），无法 --demo")
            return 2
        demo_path = out_dir / f"demo_bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
        n = mdc.write_demo_csv(demo_path, models_json=Path(args.models_json))
        info(f"--demo：已生成合成 CSV {demo_path}（{n} 行，非实测数据）")
        csv_paths = [demo_path]
        args.fig_prefix = "fig_demo" if args.fig_prefix == "fig" else args.fig_prefix
    elif args.csv:
        for c in args.csv:
            p = Path(c)
            if p.is_dir():
                csv_paths.extend(sorted(p.glob("bench_*.csv")))
            else:
                csv_paths.append(p)
        if not csv_paths:
            warn("--csv 没匹配到文件")
    else:
        csv_paths = find_latest_csv(RESULTS_DIR)
        if csv_paths:
            info(f"未指定 --csv，取最新：{csv_paths[0].name}")
    if not csv_paths:
        warn(L(False)("no_data"))
        warn("results/ 下没有 bench_*.csv。跑 bench 或用 --demo 自测渲染。")
        return 2

    demo = args.demo or any(p.name.startswith("demo_") for p in csv_paths)
    args._csv_names = [p.name for p in csv_paths]

    # ---- 2. 读数据 ----
    rows, warns = load_rows(csv_paths, include_failed=args.include_failed)
    if args.bw_mode == "dram":
        _recompute_dram_bw(rows)
    if not rows:
        warn(L(False)("no_data"))
        for w in warns:
            warn(w)
        return 2
    models_meta, models_order = load_models_json(Path(args.models_json))
    if not models_meta:
        warn(f"models.json 不可用（{args.models_json}）：标题用 CSV 里的 model id，模型顺序按 CSV 出现顺序")

    def display_of(mid: str) -> str:
        m = models_meta.get(mid) or {}
        return str(m.get("display") or mid)

    def layer_of(mid: str) -> str:
        g = (models_meta.get(mid) or {}).get("gemm") or {}
        return str(g.get("layer") or "")

    csv_order: list[str] = []
    for r in rows:
        if r.model not in csv_order:
            csv_order.append(r.model)
    order = [m for m in models_order if m in csv_order] + [m for m in csv_order if m not in models_order]
    if args.models:
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        order = [m for m in order if m in want]
        if not order:
            warn(f"--models {sorted(want)} 在 CSV 里没有匹配（CSV 里有 {csv_order}）")
            return 2
    steps_wanted = parse_steps(args.steps) if args.steps else None
    md_list, w2 = build_models(rows, order, display_of, steps_wanted)
    warns += w2
    for md in md_list:
        md.layer = layer_of(md.model)
    if not md_list:
        warn(L(False)("no_data"))
        return 2

    # ---- 3. 实测峰值虚线 ----
    measured, peak_key, peak_note = (None, "", "")
    if not args.no_measured_peak:
        measured, peak_key, peak_note = read_measured_peak(Path(args.peak_json))
        if measured:
            info(f"实测可达峰值：{measured:.1f} GB/s（来自 {Path(args.peak_json).name} 的 {peak_key}）")
        else:
            warn(f"{Path(args.peak_json)} 缺失或不可用 → 跳过“实测可达峰值”虚线（CONTRACT §7；不编数）")

    # ---- 4. 字体 / 语言 ----
    font_name, font_file, font_src = probe_cjk_font(args.font_file)
    zh = args.lang == "zh" or (args.lang == "auto" and font_name is not None)
    if args.lang == "en":
        zh = False
    L_ = L(zh)
    # 逐字符校验覆盖，避免方块
    if zh and font_file:
        sample = [v[0].format(v=3600.0, tf=args.compute_tflops, spec=args.spec_peak,
                              display="模型", N=6144, K=7168, sid="S0", name="tile_stage",
                              proto="PDL", csv="x", rows=1, models=1, steps=1, ms=1, ts="x",
                              m=128, bw=1.0, pct=1.0, p="PDL") for v in _STRINGS.values()]
        # models.json 里的 display/layer 也会进标题，一起校验
        sample += [md.display for md in md_list] + [md.layer for md in md_list if md.layer]
        cjk_miss, punct_miss = missing_glyphs(font_file, sample)
        if cjk_miss:
            warn(f"字体 {font_name} 缺 {len(cjk_miss)} 个汉字字形（{''.join(cjk_miss[:20])}）"
                 f"→ 退回英文标签，避免出方块")
            zh, L_ = False, L(False)
        else:
            if punct_miss:
                warn(f"字体 {font_name} 缺标点 {''.join(punct_miss[:12])} → 由 DejaVu Sans 兜底渲染")
            info(f"CJK 字体：{font_name}（{font_src}）→ 中文标签")
    if not zh:
        if font_name is None:
            warn("系统未探测到可用 CJK 字体（Noto Sans CJK / WenQuanYi 等）→ 标签退回英文。"
                 "装字体后可得中文图：apt-get install fonts-noto-cjk，或把 .otf 放到 "
                 "analysis/fonts/，或 --font-file <path>。")
        elif args.lang == "en":
            info("--lang en：使用英文标签")
    setup_mpl(zh, font_name if zh else None)
    warns += sanitize_labels(md_list, zh)
    lang_desc = f"zh ({font_name})" if zh else "en (no usable CJK font)"

    # ---- 5. 统一 y 上限，方便横向比较 ----
    data_max = max(float(b.max()) for md in md_list for (_M, b, _p, _n, _f) in md.curves.values())
    y_top = max(args.spec_peak * 1.10, data_max * 1.06)

    all_m = sorted({int(m) for md in md_list for m in md.ms})
    info(f"数据：{len(rows)} 行，{len(md_list)} 模型，step {sorted({s for md in md_list for s in md.curves})}，"
         f"M {all_m}，峰值 {data_max:.0f} GB/s")

    written: list[Path] = []
    for md in ([] if args.only_summary else md_list):
        p = out_dir / f"{args.fig_prefix}_bw_vs_M_{md.model}.png"
        plot_model_figure(md, args, L_, measured, peak_key, peak_note, warns, lang_desc,
                          demo, y_top, p)
        written.append(p)
        b = md.best()
        if not args.quiet and b:
            info(f"  {md.display:<18} N={md.N:<6} K={md.K:<6} "
                 + L_("best_at", sid=f"S{b[0]}", name=b[1], m=b[2], bw=b[3],
                       pct=100 * b[3] / args.spec_peak))
    if not args.no_summary:
        p = out_dir / f"{args.fig_prefix}_main_bw_vs_M.png"
        plot_summary_figure(md_list, args, L_, measured, peak_key, peak_note, warns,
                            lang_desc, demo, y_top, p)
        written.append(p)

    for w in warns:
        warn(w)
    info(f"已输出 {len(written)} 张图（{args.dpi} dpi，标签={lang_desc}）：")
    for p in written:
        info(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
