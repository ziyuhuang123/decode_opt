# analysis/ — decode_gemm 画图（CONTRACT §8）与渲染自测

三个交付文件：

| 文件 | 作用 |
|---|---|
| `plot.py` | 读 `results/bench_*.csv` → 每模型一张图 + 一张汇总图（300 dpi） |
| `make_demo_csv.py` | 生成符合 CONTRACT §6 schema 的**合成** CSV（`demo_` 前缀），仅供渲染自测 |
| `README.md` | 本文件：图怎么读、每条线含义、数据口径、自检结论 |

可选：`analysis/fonts/` 下放一个 CJK 字体（`.otf/.ttf/.ttc`），`plot.py` 会自动拾取并出中文图。

---

## 1. 快速开始

```bash
PY=/opt/conda/bin/python3

# 真数据（自动取 results/bench_*.csv 里最新一份；demo_*.csv 永远不会被当成真数据）
$PY analysis/plot.py
$PY analysis/plot.py --csv results/bench_20260913T000000Z.csv --models kimi_k3,qwen36

# 没有真数据 / 想自测渲染：一步生成合成 CSV + 画 demo 图（fig_demo_* 前缀）
$PY analysis/plot.py --demo

# 只生成合成 CSV
$PY analysis/make_demo_csv.py --out results/demo_bench.csv --summary

# 常用开关
--steps 0,4-6        只画部分 step          --models kimi_k3     只画部分模型
--logy               y 轴改 log（大 M 段 7 条曲线挤在一起时用）
--no-roofline        不画 roofline          --no-measured-peak   不画实测峰值虚线
--peak-json PATH     实测峰值 JSON（默认 results/peak_bw.json）
--lang auto|zh|en    auto = 探测到可用 CJK 字体就中文，否则英文
--font-file PATH     指定 CJK 字体文件      --dpi 300
```

输出：`results/fig_bw_vs_M_<model>.png`（每模型）+ `results/fig_bw_vs_M_all.png`（汇总）。
`--demo` 或输入 CSV 以 `demo_` 开头时，前缀自动变 `fig_demo_`，且图上打红色 DEMO 横幅。

---

## 2. 图怎么读

**坐标**：x = 批大小 M（1..128，log2 刻度，刻度只标 1/2/4/8/16/32/64/128）；
y = 等效带宽 GB/s（线性；`--logy` 可换 log）。y 上限统一为 `max(4000×1.1, 数据最大×1.06)`，
所有子图同尺度，方便横向比模型。

**7 条 step 曲线**（CONTRACT §3 的 cumulative 技术栈，顺序不许改）：颜色从**灰 → 红**渐进
（S0 最灰、S6 最深红），marker 逐 step 不同（○ □ △ ◇ ▽ + ✕），线宽随 step 递增。
大 M 段曲线会向 roofline 收敛、彼此靠近——这是物理（计算瓶颈下访存优化不再有用），
靠 marker + 颜色深浅区分；嫌挤用 `--logy`。

**三条参考线**：

| 线 | 样式 | 含义 |
|---|---|---|
| HBM3 理论峰值 4.0 TB/s | 黑色**实线** @4000 | 用户指定的主参考线（CONTRACT §7），单模型图右端有文字标注 |
| 实测可达峰值 | 青色**虚线** | 读 `results/peak_bw.json`（`tools/peak_bw_probe` 产出）。**文件缺失/解析不出就不画**，并在 stdout + 图注里警告，绝不编数 |
| roofline 等效带宽上限 | 紫色**点划线** | `BW_cap(M) = min(4000, bytes(M)/t_floor(M))`，见 §4。只画低于 4000 的那一段（平段与黑线重合，重复描会糊成一条），拐点 = 访存瓶颈 → 计算瓶颈的切换点 M* |

**汇总图**：2×3 网格（模型数变化时自动 `ncols=3, nrows=ceil((n+1)/3)`），**第 6 格放图例 + DEMO 提示**
（CONTRACT §8 要求）；图注（figure 底部小字）= 数据口径 + 数据来源 + 警告计数。

**图注永远包含**：带宽分子/分母口径、曲线协议口径、roofline 常数来源、CSV 文件名与行数、
绘图时间、peak_bw.json 的取值 key（或缺失警告）、语言/字体说明。

---

## 3. 每条曲线 = 哪个协议（数据口径，CONTRACT §6）

| step | 名字 | 相对上一步新增 | 画图用的协议 |
|---|---|---|---|
| S0 | baseline | swap-A/B 流水线，无 PDL | **B2B**（back-to-back ×90） |
| S1 | pdl | launch_dependents 早触发 + PSS 属性 | **PDL** |
| S2 | blockk | SUBS=2（一次 TMA 拉 2 个 k_block） | PDL |
| S3 | prepack | tile-major packed 权重 | PDL |
| S4 | tile_stage | BM=48, WG=2, STG=3 | PDL |
| S5 | epi_overlap | epilogue store 前二次 trigger | PDL |
| S6 | splitk | CTA 级 K-split | PDL |

即「step0 = B2B、step≥1 = PDL」：PDL 是 S1 才引入的技术，S0 画 PDL 会不公平地抬高基线。
CSV 里三种协议（ISO/B2B/PDL）都记录，`plot.py` 按上表挑；**标准协议缺数据时回退**
（S0: B2B→ISO→PDL；S≥1: PDL→B2B→ISO），图例标 `†`、图注和 stdout 说明回退到哪个协议。

其余口径规则（都在 stdout 有警告，不静默）：

- **带宽分子** = `N*K + M*K + 2*M*N` 字节（FP8 权重 + FP8 激活 + BF16 输出，不含 scale）；
  **分母** = 对应协议的 `latency_us_p50`。GB/s 用 1e9 进制（与 `pct_of_spec_peak = bw/4000` 一致）。
- `pass=false` 的行默认**排除**（correctness 不过的数据不进文章曲线）；`--include-failed` 可强行画入。
- 同一 `(model, M, step, protocol)` 多行（`--tune` 扫 config）→ 取带宽最高的一行（best config）。
- 交叉校验：CSV 的 `bandwidth_gbps` 与 `(N*K+M*K+2*M*N)/p50` 相差 >2% → 警告「口径可能不一致」，
  图上仍用 CSV 的值（诚实优先，异常值会以离群点形式暴露）。
- `pct_of_spec_peak` 若存的是百分数（85.3）会自动 /100；`bandwidth_gbps` 缺失但有延迟 → 按口径重算并警告。
- 模型顺序/显示名/层名取 `models/models.json`（缺失则用 CSV 出现顺序 + model id）。

---

## 4. roofline 等效带宽上限（公式与量纲说明）

实现（`plot.py: bw_cap_gbps`，每模型用自己的 N,K）：

```
bytes(M)  = N*K + M*K + 2*M*N                    # §6 带宽分子
t_floor   = 2*M*N*K / 282.7e12        [s]        # 282.7 TFLOPS = H20 FP8 实测峰值（CONTRACT §8 写死常数）
BW_cap(M) = min(4000, bytes(M) / t_floor / 1e9)  [GB/s]
```

物理读法：小 M 时 `bytes/t_floor >> 4000` → 被理论峰值截住（访存瓶颈区，曲线应贴着 4000 以下跑）；
M 增大后 `t_floor ∝ M` 而 bytes 几乎不涨 → 上限 ∝ 1/M 下降（计算瓶颈区），拐点约在
`M* ≈ 282.7e3/2/4000 ≈ 35`（大 N,K 时；小 N,K 模型因 `+M/N+2M/K` 修正略偏大）。
**任何实测点都不应超过这条线**；demo 数据在 M=64/128 故意压到它下面 3%~20%。

> 量纲备注（给 orchestrator / 后续维护者）：CONTRACT §8 原文写
> `BW_cap(M) = min(4000, 2*M*N*K*1e-9 / t_compute_floor_us)`。按字面代入
> `t_compute_floor_us = 2*M*N*K/282.7e12*1e6` 会得到与 M 无关的常数 0.2827，量纲不闭合
> （分子是 FLOP 数不是字节数，且 GB/µs 与 GB/s 差 1e6）。本实现取「等效带宽 = 搬运字节数 / 计算下限时间」
> 这一 roofline 标准定义，分子用 §6 的 bytes 口径，与曲线 y 轴同口径、可直接比较。
> 若 CONTRACT 后续改公式，只需改 `bw_cap_gbps()` 一处。

图注里 roofline 的来源文案固定为：
`Roofline: t_floor = 2*M*N*K / 282.7 TFLOPS (H20 FP8 measured peak, CONTRACT §8 constant)`。

---

## 5. 中文字体策略（不许出方块）

探测顺序：`--font-file` → 环境变量 `DECODE_GEMM_CJK_FONT` → `analysis/fonts/*.{otf,ttf,ttc}`
→ 系统字体按候选名精确匹配（Noto Sans CJK SC / Source Han Sans / WenQuanYi / 微软雅黑 / 苹方 …）
→ 名字含 CJK 线索的模糊匹配。

拿到字体后还会**逐字符校验 glyph 覆盖**（`FT2Font.get_char_index`）：

- 缺汉字（U+2E80 以上）→ 整体退回英文标签 + stdout 警告（DejaVu 没有汉字可兜底，硬画必出方块）；
- 只缺标点（§ × · – “ ” † 等）→ 保留中文，靠字体链兜底：`rcParams["font.family"] = [CJK, "DejaVu Sans"]`
  （注意：必须是 **family 列表**才启用逐字形 fallback；`font.sans-serif` 列表实测不生效）；
- `axes.unicode_minus=False`，负号走 ASCII，避免 U+2212 缺字形。

本机（2026-09-13）**没有装任何 CJK 字体**，所以默认产出英文标签图 + 一条 stdout 警告：
`系统未探测到可用 CJK 字体 … 装字体后可得中文图：apt-get install fonts-noto-cjk，
或把 .otf 放到 analysis/fonts/，或 --font-file <path>`。
英文模式下 `models.json` 里带中文的 layer 名会自动从标题省略（同样为避免方块），并有警告。

---

## 6. 渲染自检结论（2026-09-13，demo 数据）

命令：`python3 analysis/plot.py --demo`（合成 CSV = `results/demo_bench_<UTC>.csv`，本次自检为
`demo_bench_20260913T054653Z.csv`，840 行 = 5 模型 × 7 step × 8 M × 3 协议；形状用
`models/models.json` 的定稿真形状，数值是编的；每次 `--demo` 会新生成一份，旧的可删）。

产出（**全部带 demo 前缀，非实测结果**）：

```
results/fig_demo_bw_vs_M_all.png              汇总 2×3
results/fig_demo_bw_vs_M_kimi_k3.png          results/fig_demo_bw_vs_M_qwen36.png
results/fig_demo_bw_vs_M_glm52.png            results/fig_demo_bw_vs_M_deepseek_v4_pro.png
results/fig_demo_bw_vs_M_minimax_m3.png
```

逐项目视检查（`view_image` 看过每张）：

| 检查项 | 结论 |
|---|---|
| 曲线不重叠成糊 | ✅ M≤32 段 7 条曲线间距 ≥150 GB/s；M=64/128 向 roofline 收敛属物理，marker+色深可分辨；`--logy` 可进一步拉开 |
| 图例清楚 | ✅ 单模型图图例在右侧外（9 项：roofline + 2 参考线 + 7 step，含协议名）；汇总图第 6 格同款图例，不压曲线 |
| log2 轴刻度 1/2/4/…/128 | ✅ `set_xscale('log', base=2)` + 固定刻度，无杂散 minor tick |
| 峰值线标注清楚 | ✅ 黑实线 4000 单模型图右端文字标注 + 图例；实测峰值虚线路径用 /tmp 假 `peak_bw.json`（best.bandwidth_gbps=3687.4）单独验证：青色虚线 + 图例 + 图注写明取值 key；缺失路径验证：不画线 + stdout/图注双警告 |
| 标题/图注不溢出、不互压 | ✅ 图注行距按「磅」计算（`footnote_layout`），1 行/2 行网格、7~8 行图注、中/英文四种组合都验过 |
| 形状差异可见（orchestrator 要求） | ✅ qwen36（8.4 MB 权重）平台只有 ~2.7 TB/s（launch 固定开销占比大），kimi/glm/dpsk（88~117 MB）到 ~3.5 TB/s；qwen36 的 S6 splitk 增益最大（CTA 数不足），与 models.json 的选型理由一致 |
| M=128 掉到 compute roof 以下 | ✅ 各模型 M=128 合成值 1.0~1.16 TB/s，低于各自 roof 上限 1.14~1.24 TB/s |
| 中文路径 | ⚠ 本机无 CJK 字体 → 默认英文（符合 CONTRACT「可用则中文」）。中文分支用 fontTools 现造的「全宽占位字体」验证：探测/glyph 校验/`family` 列表 fallback/中文排版度量全走通，ASCII 与标点由 DejaVu 兜底、零 missing-glyph 警告；真中文需装字体后重跑 |
| 鲁棒性 | ✅ 脏 CSV（坏行/缺 bandwidth/pass=false/重复行/协议大小写/缺标准协议/带宽与延迟不自洽）全部按预期警告并降级，不崩；`--logy --no-roofline --steps --models --no-summary`、无数据路径均 smoke 通过 |

已知限制：demo 的 S4/S5/S6 在 M≤32 段差距 <2%（真实情况也接近），打印小尺寸时靠 marker 区分；
汇总图第 6 格在模型数 = 3 的倍数时会被图例占用（自动改用最后一格，行为有日志）。

---

## 7. 对 CSV schema 的建议（**schema 仍以 CONTRACT §6 为准**，这里只提观察）

1. **`config` 列内部分隔符建议用 `;` 而不是 `,`**。`csv` 模块能正确引号包裹含逗号的字段，
   但下游 `awk -F,` / 肉眼检查会错位（我的 demo config 就含逗号，已用引号包裹，harness 请照做或改 `;`）。
2. **建议补 `latency_us_p30` 列**。§6 文字说 ISO 记录「p50/30」，但列定义只有 `latency_us_p50`；
   要么补列，要么把文字改成只记 p50，避免 harness 与契约各执一词。
3. **建议补 `clocks_sm_mhz` 列**。§6 要求「记录 clocks.sm 进结果 JSON」，但跨批次画图时时钟不可见，
   曲线不可比却无从察觉；进 CSV 一行成本极低（图注可自动带上）。
4. **建议补 `m_tile` 列**（或保证 `config` 里 `M_TILE=` 永远存在）。按 M_TILE 分组画第二张图
   （tile 阶梯 vs 带宽）是文章很可能要的视角，现在只能从 config 字符串里正则抠。
5. **`protocol` 取值建议固定大写 `ISO/B2B/PDL`** 并写进契约（我已做大小写归一 + 未知值警告）。
6. **`step_name` 请与 §3 表逐字一致**；我以 CSV 值优先、缺失回退内置表，两边不一致时图上会出现别名。
7. `bandwidth_gbps` 建议 ≥3 位小数、`latency_us_p50` ≥4 位：我的交叉校验容差 2%，
   低精度舍入在小 M（延迟 ~3µs）时会顶到容差边缘产生假警告。
