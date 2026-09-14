# models/ — 5 个模型的代表性 decode GEMM 形状

> shapes agent 产出。**write scope 只有两个文件**：`models/models.json` 与本文件。
> 接口冻结来源：`../CONTRACT.md` §2（模型与形状）+ §1（问题定义与 (N,K) 约定）+ §7（峰值参考线）。
> 本任务是**只读研究**：没有编译、没有跑 GPU、没有产生任何 results/。

---

## 1. 结论速览（每模型 1 个代表形状）

| id | 层名 | N | K | N/128 | K/128 | 权重(fp8) | 选取规则 |
|---|---|---:|---:|---:|---:|---:|---|
| `kimi_k3` | o_proj | **7168** | **12288** | 56 | 96 | 84.00 MiB | 规则1 o_proj |
| `qwen36` | o_weight (GDN/dense 输出投影) | **2048** | **4096** | 16 | 32 | 8.00 MiB | 规则1 o_proj |
| `glm52` | wo (final_o_projection) | **6144** | **16384** | 48 | 128 | 96.00 MiB | 规则1 o_proj |
| `deepseek_v4_pro` | wo_b (grouped 低秩 o_proj 的第二级) | **7168** | **16384** | 56 | 128 | 112.00 MiB | 规则1+2 o_proj 第二级 |
| `minimax_m3` | w_o (o_proj) | **6144** | **8192** | 48 | 64 | 48.00 MiB | 规则1 o_proj |

5 个主形状全部命中规则 1（o_proj：`N = hidden_size`，`K = attn 输出维 / o_proj 输入维`）。
**15 个形状（5 主 + 10 alt）的 N 和 K 全部是 128 的整数倍，pad 数量为 0**，见 §5。

---

## 2. (N,K) 约定与三个 harness 的存储写法换算（最容易踩的坑）

CONTRACT §1 冻结的口径是：

```
out[M,N] = act[M,K] @ W[N,K]^T      # W 是 [N,K]，K-contiguous
weight_scales[ceil(N/128), K/128]   # scale group = 128 个 K 元素
```

本目录下所有 (N,K) **已经归一到这个口径**。但 5 个 harness 源码用了 3 种不同的权重写法，
直接照抄源码里的 tuple 会得到转置的错误答案：

| harness | 源码写法 | 源码里的 tuple 是 | 到 (N,K) 的换算 |
|---|---|---|---|
| kimi_k3 / qwen36 / minimax_m3 | `F.linear(x, W)`（kimi 是等价封装 `linear(x, W, name)`） | `[N, K]`（out_features, in_features） | 直接读，无需换算 |
| deepseek_v4_pro | `fp8_gemm(x, W)`，文档串明确 `D = x @ W^T`（BM:L525-527），`n = W.shape[0]`（BM:L530） | `[N, K]` | 直接读，无需换算 |
| **glm52** | `x @ W`（无转置！BM:L796/L798/L1044） | **`[K, N]`**（in_features, out_features） | **必须转置读**：源码 `(16384, 6144)` → `(N,K) = (6144, 16384)` |

kimi_k3 的 `[N,K]` 约定由源码自己写死并自检：注释 `Weight shape reference (PyTorch Linear: W=[N,K], y = x @ W.T)`（BM:L114），
以及一个 numpy 小自检 `W=[N,K], x=[B,K] -> y=[B,N]`（BM:L3001-3005）。
deepseek 的 `_quantize_weight_fp8` 文档串也写明输入是 `[N, K]` 权重、scale 布局 `[ceil(N/128), ceil(K/128)]`（BM:L445-447），
与 CONTRACT §1 的 `weight_scales` 完全同构（`GRAN_K = 128`，BM:L111）。

---

## 3. 数据来源与取证方法

**权威源（唯一数字来源）**：`/home/admin/workspace/hengfeng_data/weave_new/DSA-learn/baselines/h100/<model>/full_attention_benchmark.py`

**辅助源（只做交叉核对，不作为任何数字的出处）**：
`/home/admin/workspace/hengfeng_data/weave_new/DSA-learn/baselines/h20/attention/model_arch_params.json`

禁令执行情况：**没有使用任何公开模型的记忆值**。每个 N/K 都是从 harness 里的常量表达式推导出来的，
并且尽量做到「四层互证」：

1. **常量定义行** — 例如 kimi `HIDDEN = 7168`（BM:L69）、`HD_KDA = H_KDA * D  # 12288`（BM:L73）
2. **权重构造行** — 例如 kimi `"o_proj": (HIDDEN, HD_KDA),  # [7168, 12288]`（BM:L126）
3. **harness 自己的静态 assert / self-check** — 这是最强的一层，因为它是源码作者对同一组数字的独立断言：
   - kimi_k3 BM:L2982-2997 + L3008-3012（含 `assert WEIGHT_SHAPES_KDA["o_proj"] == (7168, 12288)`）
   - deepseek_v4_pro BM:L3677-3683, L3688（`HIDDEN==7168`, `O_GROUPS==16`, `O_LORA==1024`, `H_Q==128`, `HEAD_D==512`）
   - minimax_m3 BM:L2822-2833 + L3021-3023（`PROJ_FUSED_DIM==9856`, `PROJ_MAIN_DIM==9216`, `w_fused.shape==(9856,HIDDEN)`）
   - glm52 / qwen36 没有形状 assert，因此这两个模型只用第 1/2/4 层 + 独立 fp32 参考路径互证
     （glm52 的 fp32 参考用同一组权重同形状重算：BM:L1678/L1686/L1756）
4. **decode 调用点** — 证明这个 GEMM 真的在 decode 路径上（见 §6 各模型小节）

辅助源 `model_arch_params.json` 与本次独立提取的结果**逐项一致**（kimi o_proj `[7168,12288]`、qwen36 两变体 o_proj `[2048,4096]`、
minimax o_proj `[6144, 64*128=8192]`、glm52 o_proj 记作 `[64*256=16384, 6144]`——注意该文件对 glm52 沿用了源码的 `[K,N]` 记法，
换算后与本表一致）。唯一需要提醒的是：辅助源里 deepseek_v4_pro **没有** o_proj 条目
（它的 `params` 里没有这一项），所以 deepseek 的输出投影完全由本次从 BM:L1477-1479 + L2305-2311 + L2583-2586 独立提取。

---

## 4. 完整形状表（主 + alt）

| # | id | 角色 | 层名 | N | K | N/128 (scale-group 数) | K/128 (k_block 数) | N/64 (BM=64 tile 数) | 权重字节(fp8) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `kimi_k3` | main | o_proj | 7168 | 12288 | 56 | 96 | 112 | 88080384 (84.00 MiB) |
| 2 | `kimi_k3` | alt1 | fused_qkvg (KDA 输入投影, 模型内最大单权重) | 49152 | 7168 | 384 | 56 | 768 | 352321536 (336.00 MiB) |
| 3 | `kimi_k3` | alt2 | q_b (MLA q_lora 上投影, 窄 K) | 18432 | 1536 | 144 | 12 | 288 | 28311552 (27.00 MiB) |
| 4 | `qwen36` | main | o_weight (GDN/dense 输出投影) | 2048 | 4096 | 16 | 32 | 32 | 8388608 (8.00 MiB) |
| 5 | `qwen36` | alt1 | qkvz_weight (GDN fused in_proj_qkvz) | 12288 | 2048 | 96 | 16 | 192 | 25165824 (24.00 MiB) |
| 6 | `qwen36` | alt2 | q_weight (dense query+output_gate 融合) | 8192 | 2048 | 64 | 16 | 128 | 16777216 (16.00 MiB) |
| 7 | `glm52` | main | wo (final_o_projection) | 6144 | 16384 | 48 | 128 | 96 | 100663296 (96.00 MiB) |
| 8 | `glm52` | alt1 | wq_b (q_lora 上投影, 大 N 窄 K) | 16384 | 2048 | 128 | 16 | 256 | 33554432 (32.00 MiB) |
| 9 | `glm52` | alt2 | wq_a (q_lora 下投影, 小 N) | 2048 | 6144 | 16 | 48 | 32 | 12582912 (12.00 MiB) |
| 10 | `deepseek_v4_pro` | main | wo_b (grouped 低秩 o_proj 的第二级) | 7168 | 16384 | 56 | 128 | 112 | 117440512 (112.00 MiB) |
| 11 | `deepseek_v4_pro` | alt1 | wq_b (q_lora 上投影, N=65536 超大 N) | 65536 | 1536 | 512 | 12 | 1024 | 100663296 (96.00 MiB) |
| 12 | `deepseek_v4_pro` | alt2 | wqkv_a_fp8 (fused q_a+kv 下投影, 小 N) | 2048 | 7168 | 16 | 56 | 32 | 14680064 (14.00 MiB) |
| 13 | `minimax_m3` | main | w_o (o_proj) | 6144 | 8192 | 48 | 64 | 96 | 50331648 (48.00 MiB) |
| 14 | `minimax_m3` | alt1 | w_qkv_idx_fused (sparse 单融合 QKV+index 投影) | 9856 | 6144 | 77 | 48 | 154 | 60555264 (57.75 MiB) |
| 15 | `minimax_m3` | alt2 | w_qkv_fused (dense 主 QKV 投影) | 9216 | 6144 | 72 | 48 | 144 | 56623104 (54.00 MiB) |

`K/128` 就是 CONTRACT §1 里每个输出元素要走的 k_block 数（`kKBlock = 128`，wgmma k32×4）。
注意跨度：`K/128` 从 12（kimi q_b / deepseek wq_b）到 128（glm52 wo / deepseek wo_b），差 10.7 倍；
`N/128` 从 16（qwen36 主形状 / glm52 wq_a / deepseek wqkv_a）到 512（deepseek wq_b），差 32 倍。
这个组合能同时压到「N 太小 CTA 不够」（S6 splitk 的动机）和「K 太长流水线填不满」两端。

---

## 5. 128 对齐门禁（规则 3）

规则 3 要求 `N % 128 == 0`（否则向上取整到 128 倍数并在 why 注明 pad）且 `K % 128 == 0`（否则换备选层）。

**结果：15/15 个形状全部通过，pad 数量 = 0。** 因此 `models.json` 里没有任何一条 `why` 提到 pad——
这不是漏写，是真的不需要。这不是刻意挑出来的巧合：**5 个模型的 o_proj 本身就天然 128 对齐**，
所以规则 1（首选 o_proj）与规则 3（必须 128 对齐）在 5/5 个模型上都不冲突，无需退化到规则 2。
alt 形状则是在「天然对齐的层」里挑的——凡是 N 不是 128 倍数的候选一律换层而不是 pad，被淘汰的都记录在 §7
（例如 kimi 的 `fused_qkv_a` N=2112 = 16.5×128、glm52 的 `wkv_a` N=576 = 4.5×128）。

**门禁是可复现的**：§12 第 6 步那段 python 会重新校验 `models.json` 的
`N % 128 == 0` / `K % 128 == 0` / 顶层与每层的 schema key 集合 / `alt` 数量 1-2 / `gemm.N == hidden`（规则 1），
任何一条不过就抛 AssertionError。生成时还额外校验过 `source.file` 存在且指向对应模型的 harness、
以及同一模型内 (N,K) 不重复。

---

## 6. 每模型细节（推导链 + 行号证据）

### 6.1 `kimi_k3` — Kimi-K3（hidden=7168，93 层 = 69 KDA + 24 MLA）

**常量链（源）**

| 常量 | 行号 |
|---|---|
| `HIDDEN = 7168` | BM:L69 |
| `H_KDA = 96 / H_MLA = 96` | BM:L70-71 |
| `D = 128` | BM:L72 |
| `HD_KDA = H_KDA*D = 12288 / HD_MLA = H_MLA*D = 12288` | BM:L73-74 |
| `MLA_QK = 128+64 = 192 / MLA_Q_LORA = 1536` | BM:L80, L82 |
| `NUM_KDA = 69 / NUM_MLA = 24 / NUM_LAYERS = 93` | BM:L86-88 |

**主形状证据链**

1. KDA `"o_proj": (HIDDEN, HD_KDA)` → `[7168, 12288]` — BM:L126
2. MLA `"o_proj": (HIDDEN, HD_MLA)` → `[7168, 12288]` — BM:L136
3. 自断言 `WEIGHT_SHAPES_KDA["o_proj"] == (7168, 12288)` — BM:L2990
4. 自断言 `WEIGHT_SHAPES_MLA["o_proj"] == (7168, 12288)` — BM:L2997
5. 自断言 `H_MLA*D == 96*128 == 12288  # g_proj/o_proj inner` — BM:L3012
6. 文件头注释 `O_proj : [7168, 12288]`（KDA 段 / MLA 段各一次） — BM:L20, L33

**decode 路径证据（证明这个 GEMM 每层每步必经）**

- KDA decode：`_kda_project_production` 的 decode 侧流分支 `use_side_stream = phase == "decode" and 0 < B*L <= 64` — BM:L838, L867
- KDA decode：`kda_full_forward` 里 `production_stage("kda_o_proj")` → `linear(normed, params["o_proj"], "o_proj")` — BM:L1019, L1070-1072
- MLA decode：`mla_decode_forward`（文档串 “One ordinary Q=1 … absorbed-MLA decode step”）→ `production_stage("mla_o_proj")` → `linear(gated_out, params["o_proj"], …)` — BM:L1527-1529, L1610-1611

**alt 形状与理由**

- `fused_qkvg` = `(4*HD_KDA, HIDDEN)` = **[49152, 7168]**
  - BM:L118（自断言 BM:L2982）；decode 调用点 BM:L879 / L885（`production_stage("kda_fused_qkvg")`）。理由：模型内**最大的单个权重**（336 MiB fp8），KDA 占 69/93 层、decode 每步必经；给出 `N/128 = 384` 的超大 N 工作点，是纯带宽极限与 CTA 数充足情形的对照。
- `q_b` = `(H_MLA*MLA_QK, MLA_Q_LORA)` = **[18432, 1536]**
  - BM:L132（自断言 BM:L2994 + `H_MLA*MLA_QK == 96*192 == 18432` BM:L3008）；decode 调用点 BM:L1562（`mla_decode_forward` 内 `production_stage("mla_q_b")`）。理由：覆盖 MLA 变体（24/93 层），且给出 `K/128 = 12` 的**窄 K** 工作点，与主形状的 `K/128 = 96` 形成 8 倍对比，专门压 S2(blockk)/S6(splitk) 在短 K 上的收益边界。

**考虑过但淘汰的候选（诚实记录）**

- `kv_b` = `[24576, 512]`（BM:L134）
  - **不在生产 decode 路径上**。`mla_decode_forward`（BM:L1527-1612）只用 `g_proj`/`fused_qkv_a`/`q_b`/`o_proj` 四个权重；`kv_b` 只出现在 prefill 生产路径（BM:L1406）、prefill 参考（BM:L1466）和 **decode 的独立 fp32 参考**（BM:L1637）里——absorbed-MLA decode 靠 `w_kc`/`w_vc` 吸收，根本不展开 kv_b。选它会违反“decode 路径”这个前提。
- `fused_qkv_a` = `[2112, 7168]`（BM:L130, 自断言 L2993）
  - **规则 3 淘汰**：N=2112 = 16.5×128，需要 pad 到 2176。虽然它确实在 MLA decode 路径上（BM:L1556），但既然有零 pad 的干净候选，就不引入 pad 形状。
- `b_proj` = `[96, 7168]`（BM:L119）
  - **规则 3 淘汰**：N=96 不是 128 的倍数（要 pad 到 128，浪费 25%）。
- `f_b_proj` = `[12288, 128]`（BM:L121）
  - 对齐合法但**退化**：K=128 只有一个 k_block，S2/S3 的多 stage 流水线与 SUBS=2 全部失效，无法体现文章主线的任何一步。
- `g_proj` = `[12288, 7168]`（BM:L135, 自断言 L2996, decode 调用点 BM:L1553）
  - 对齐合法、也在 decode 路径上，是合格的第三候选。没选它是因为 alt 上限为 2，而 `q_b` 的窄 K（K/128=12）比 `g_proj` 的 K/128=56 提供更多正交信息。

---

### 6.2 `qwen36` — Qwen3.6-35B-A3B（hidden=2048，40 层 = 30 GDN + 10 dense GQA）

**常量链（源）**

| 常量 | 行号 |
|---|---|
| `MODEL_NAME = "Qwen3.6-35B-A3B" / HIDDEN_SIZE = 2048` | BM:L37-38 |
| `GDN_H_QK=16, GDN_H_V=32, GDN_D=128` | BM:L44-46 |
| `GDN_QK_DIM = 16*128 = 2048` | BM:L47 |
| `GDN_V_DIM = 32*128 = 4096` | BM:L48 |
| `GDN_QKV_DIM = 2*2048 + 4096 = 8192` | BM:L49 |
| `DENSE_H_Q=16, DENSE_H_KV=2, DENSE_D=256` | BM:L53-55 |
| `DENSE_Q_DIM = 16*256 = 4096 / DENSE_KV_DIM = 2*256 = 512` | BM:L56-57 |
| `LAYER_COUNTS = {"gdn": 30, "dense": 10}` | BM:L61 |

**主形状证据链**

1. GDN `"o_weight": _bf16_weight(HIDDEN_SIZE, GDN_V_DIM)` → `[2048, 4096]` — BM:L282
2. dense `"o_weight": _bf16_weight(HIDDEN_SIZE, DENSE_Q_DIM)` → `[2048, 4096]` — BM:L323
3. `_bf16_weight(out_features, in_features)` → 返回 `randn(out, in)`，即 `[N, K]` — BM:L256-261

**decode 路径证据（证明这个 GEMM 每层每步必经）**

- GDN decode：`_forward_gdn_decode` → `nvtx_stage("qwen36.gdn.decode_step%d.output_projection")` → `F.linear(normalized.reshape(batch, GDN_V_DIM), layer["o_weight"])` — BM:L462, L523-529
- dense decode：`_forward_dense_decode` → `nvtx_stage("qwen36.dense.decode_step%d.output_projection")` → `F.linear(gated.reshape(batch, DENSE_Q_DIM), layer["o_weight"])` — BM:L1056, L1115-1119

**alt 形状与理由**

- `qkvz_weight` = `cat(qkv_weight[8192,2048], z_weight[4096,2048], dim=0)` = **[12288, 2048]**
  - BM:L267-268 + L287-288；源码注释直接写明 `in_proj_qkvz N=12288`（BM:L285）。decode 调用点 BM:L479（`decode_step%d.input_projection.qkvz`），prefill 对应 BM:L549。理由：GDN 层（30/40）decode 每步必经的**单个融合输入投影**，给出 `K/128 = 16` 的短 K + `N/128 = 96` 的中 N 组合。
- `q_weight` = `_bf16_weight(2*DENSE_Q_DIM, HIDDEN_SIZE)` = **[8192, 2048]**
  - BM:L318；注释说明每 head 产出 `[query(256) | output_gate(256)]` 交织，所以是 2×（BM:L317）。decode 调用点 BM:L1072（`dense.decode_step%d.input_projection.q_gate`）。理由：覆盖 dense 变体（10/40 层），形状与主 o_proj 正交（N/K 互换量级）。

**考虑过但淘汰的候选（诚实记录）**

- `ba_weight` = `[64, 2048]`（BM:L270+L269+L289-290）
  - **规则 3 淘汰**：N=64 = 0.5×128，要 pad 到 128，一半算力空转。它确实在 decode 路径上（BM:L483），但形状没有研究价值。
- `k_weight` / `v_weight` = `[512, 2048]`（BM:L319-320）
  - 对齐合法但太小（N/128 = 4，只有 4 个 tile），1 MiB 权重连一次 L2 都填不满，测不出带宽。
- dense 变体的 `o_weight`（BM:L323）
  - **没有单独列为 alt**：它和 GDN 的 `o_weight` 形状完全相同（都是 `[2048, 4096]`，因为 `GDN_V_DIM == DENSE_Q_DIM == 4096`），列出来是重复形状而不是不同层的信息。主形状一条即覆盖 40/40 层。

---

### 6.3 `glm52` — GLM-5.2（hidden=6144，78 层 = 21 full_index + 57 shared_index）

**常量链（源）**

| 常量 | 行号 |
|---|---|
| `HIDDEN = 6144 / Q_LORA = 2048 / H_Q = 64` | BM:L56-58 |
| `Q_NOPE = 192 / ROPE_D = 64 / Q_HEAD_D = Q_NOPE + ROPE_D = 256` | BM:L59-61 |
| `LATENT_D = 512 / MLA_D = 576 / V_HEAD_D = 256` | BM:L62-64 |
| `H_IDX = 32 / IDX_D = 128` | BM:L66-67 |
| `NUM_LAYERS = 78 / NUM_FULL_INDEX = 21 / NUM_SHARED_INDEX = 57` | BM:L70-72 |
| ``scaled_randn(*shape, fan_in)` → 返回 `randn(*shape)`，即 tuple 直接就是 tensor shape` | BM:L100-101, L91-92 |

**主形状证据链**

1. `"wo": scaled_randn(H_Q * V_HEAD_D, HIDDEN, fan_in=H_Q*V_HEAD_D)` → tensor shape `(16384, 6144)` — BM:L115
2. **转置换算**：调用处 `final_hidden = reconstructed.reshape(tokens, H_Q*V_HEAD_D) @ w["wo"]`，再 `.view(batch, q_len, HIDDEN)` — BM:L1044-1045
3. → `(tokens,16384) @ (16384,6144) = (tokens,6144)`，所以 **K = H_Q*V_HEAD_D = 64*256 = 16384，N = HIDDEN = 6144** — BM:L1044-1045
4. 独立 fp32 参考用同一 `w["wo"]` 同形状重算，交叉印证 — BM:L1756
5. stage 名 `final_o_projection` — BM:L1043

**decode 路径证据（证明这个 GEMM 每层每步必经）**

- `_forward_layer_one`（文档串：“Execute one prefill chunk or **exactly one ordinary-Q1 decode step**”，并硬校验 decode 时 `q_len != 1` 就抛错） — BM:L785-790
- `forward_layer`：`phase != "decode"` 走一次；decode 则 `q_len` 必须是 2，拆成 **两个 Q=1 步** 分别调 `_forward_layer_one` — BM:L1187-1202
- → wo GEMM 在 decode 下每个 Q=1 步各调一次，M = batch = 64 — BM:L1044

**alt 形状与理由**

- `wq_b` = `scaled_randn(Q_LORA, H_Q*Q_HEAD_D)` → `(2048, 16384)`，换算 **(N,K) = (16384, 2048)**
  - BM:L109；调用 `q_full = q_lora @ w["wq_b"]` BM:L798（stage `q_a_rms_q_b` BM:L795），fp32 参考 BM:L1686。理由：decode 每步必经的 q_lora 上投影，`N/128 = 128` 大 N + `K/128 = 16` 窄 K，和主形状（N/128=48, K/128=128）几乎完全互补。
- `wq_a` = `scaled_randn(HIDDEN, Q_LORA)` → `(6144, 2048)`，换算 **(N,K) = (2048, 6144)**
  - BM:L107；调用 `q_lora_raw = hidden @ w["wq_a"]` BM:L796，fp32 参考 BM:L1678。理由：给出 `N/128 = 16` 的**小 N** 工作点（和 qwen36 主形状同量级），是 S6 splitk 的第二个目标用例；同时它吃的是完整 hidden（K=6144），与 qwen36 的 K=4096 不同。

**考虑过但淘汰的候选（诚实记录）**

- `wkv_a` = `scaled_randn(HIDDEN, MLA_D)` → (6144, 576)，即 N=576（BM:L110, 调用 BM:L802）
  - **规则 3 淘汰**：N = 576 = 4.5×128，要 pad 到 640（+11%）。它在 decode 路径上，但既然规则 3 明确“不满足则换备选层”，就换成对齐干净的 `wq_a`/`wq_b`。
- `index_gate` → N=H_IDX=32（BM:L118, 调用 BM:L883）
  - **规则 3 淘汰**：N=32 = 0.25×128，pad 到 128 会有 75% 空转。
- `index_wk` → (N,K) = (128, 6144)（BM:L117, 调用 BM:L863）
  - 对齐合法但**退化**：N/128 = 1，整个 GEMM 只有 1 个输出 tile，128 个 SM 里只有 1 个干活。
- `index_wq_b` → (N,K) = (4096, 2048)（BM:L116, 调用 BM:L860）
  - 对齐合法，但只在 `state["variant"] == "full_index"` 分支里（BM:L856），即 **21/78 层**才有；且 `wq_b`(16384,2048) 已覆盖同样的 K=2048 窄 K 语义、层覆盖率高得多（78/78）。
- `w_kc` / `w_vc`（BM:L113-114）
  - **不是普通 GEMM**：它们是 3-D absorbed 视图 `(H_Q, Q_NOPE, LATENT_D)` / `(H_Q, LATENT_D, V_HEAD_D)`，通过 `torch.bmm` + `"mhd,hde->mhe"` einsum 使用（BM:L821-825, L1041, L1753），是 batched GEMM，不符合 CONTRACT §1 的单 `W[N,K]` 定义。

---

### 6.4 `deepseek_v4_pro` — DeepSeek-V4-Pro（hidden=7168，61 层 = 31 HCA + 30 CSA）

**常量链（源）**

| 常量 | 行号 |
|---|---|
| `HIDDEN = 7168 / H_Q = 128 / HEAD_D = 512` | BM:L63-65 |
| `QK_NOPE_D = 448 / QK_ROPE_D = 64 / V_HEAD_D = 512` | BM:L66-68 |
| `Q_LORA = 1536 / **O_GROUPS = 16 / O_LORA = 1024**` | BM:L69-71 |
| `IDX_H = 64 / IDX_D = 128 / HCA_LAYERS = 31 / CSA_LAYERS = 30` | BM:L72-79 |
| `GRAN_K = 128（per-128(K) scale group，与 CONTRACT §1 的 kKBlock=128 同值）` | BM:L111 |
| `HCA_COMP_OUT = 1024 / CSA_COMP_OUT = 2048 / IDX_COMP_OUT = 512` | BM:L84-88 |
| `静态自检 `HIDDEN==7168, H_Q==128, HEAD_D==512, Q_LORA==1536, O_GROUPS==16, O_LORA==1024`` | BM:L3677-3683 |

**主形状证据链**

1. `wo_b_raw = _randn(HIDDEN, O_GROUPS * O_LORA)` → `[7168, 16384]` — BM:L1479
2. `"wo_b_fp8": _quantize_weight_fp8(wo_b_raw)`（`_quantize_weight_fp8` 文档串：输入是 `[N, K]` 权重，scale `[ceil(N/128), ceil(K/128)]`） — BM:L1512, L445-457
3. `_fp8_gemm_prod` 文档串 `Production FP8 GEMM: D = x @ W^T`，且 `n = w_fp8_pair[0].shape[0]` — BM:L525-532
4. `_fp8_gemm_ref` 交叉印证：`(x_deq @ w_deq.t())` — BM:L547-555

**decode 路径证据（证明这个 GEMM 每层每步必经）**

- decode：`_layer_forward_decode`（“Two sequential ordinary-Q1 decode steps”，`assert q_len == 2`）→ `fp8_gemm = _fp8_gemm_ref if use_ref else _fp8_gemm_prod` — BM:L2368-2373, L2384
- decode 输出投影：`o_mid = grouped_oa(attn_flat, *common["wo_a_fp8"])` 然后 `output = fp8_gemm(o_mid, common["wo_b_fp8"])` — BM:L2583-2586
- prefill 对应位置：`with _attention_stage("output_proj"): output = fp8_gemm(o_mid, common["wo_b_fp8"])  # [q_len, HIDDEN]` — BM:L2305-2311

**alt 形状与理由**

- `wq_b` = `_randn(H_Q * HEAD_D, Q_LORA)` = **[65536, 1536]**
  - BM:L1475（`"wq_b_fp8"` BM:L1510）；decode 调用点 `q_full = fp8_gemm(q_lora, common["wq_b_fp8"])` BM:L2407（prefill BM:L2103，注释 `# [q_len, H_Q*HEAD_D]`）。理由：`N/128 = 512` 是本形状表里**最大的 N**，`K/128 = 12` 是最小的 K 之一，96 MiB 权重；这是 grid 维度极端充足 + K 极短的组合，和主形状（N/128=56, K/128=128）完全互补。
- `wqkv_a_fp8` = `cat([wq_a[1536,7168], wkv[512,7168]], dim=0)` = **[2048, 7168]**
  - BM:L1474 + L1476 + L1506-1508；decode 调用点 `qkv_a = fp8_gemm(x, common["wqkv_a_fp8"])` BM:L2403（prefill BM:L2098），随后 `q_lora = qkv_a[:, :Q_LORA]` / `kv_seed = qkv_a[:, Q_LORA:]` BM:L2404-2405 印证了 N 方向的 1536+512 拼接。理由：给出 `N/128 = 16` 的小 N 工作点（S6 splitk 第三个目标），且 K=7168 与既有工作一致。

**考虑过但淘汰的候选（诚实记录）**

- `wo_a` = `[O_GROUPS, O_LORA, per_group_d]` = `[16, 1024, 4096]`（BM:L1477-1478）
  - **不是单个稠密 GEMM**：它走 grouped einsum `"bhr,hdr->bhd"`（BM:L1404-1407），16 个 group 各是 (N=1024, K=4096)，并且激活侧还要先做 per-token-group fp8 量化（BM:L1392-1400）。拍平成 (16384, 4096) 需要伪造 grouping 语义，不符合 CONTRACT §1 的单 `W[N,K]` + `weight_scales[ceil(N/128), K/128]` 定义。它的权重字节（67108864 B）也确实小于 `wo_b`（117440512 B），所以规则 2 的“最大单个权重 GEMM”同样指向 `wo_b`。
- `w_proj` = `_randn(IDX_H, HIDDEN)` = `[64, 7168]`（BM:L1565, decode 调用 BM:L2200）
  - **规则 3 淘汰**：N=64 = 0.5×128。
- `idx_comp` = `[IDX_COMP_OUT=512, 7168]`（BM:L1566, decode 调用 BM:L2207-2210）
  - 对齐合法但只在 CSA 层（30/61）且 N/128=4，太小。
- `w_comp` = `[HCA_COMP_OUT=1024 | CSA_COMP_OUT=2048, 7168]`（BM:L1533 / L1555）
  - 对齐合法，但 HCA/CSA 两个变体形状不同、且 N 只有 8/16 个 tile；信息量不如 `wq_b`/`wqkv_a`。
- `idx_wq_b` = `[IDX_H*IDX_D=8192, Q_LORA=1536]`（BM:L1564, `idx_wq_b_fp8` L1583, decode 调用 BM:L2202-2203）
  - 对齐合法（N/128=64, K/128=12），是合格的第三候选。没选它是因为 alt 上限为 2，而它只在 CSA 层（30/61）出现，且它的 (N,K) 量级已被 `wq_b`(65536,1536) 以更高覆盖率（61/61）代表。

---

### 6.5 `minimax_m3` — MiniMax-M3（hidden=6144，60 层 = 57 sparse + 3 dense）

**常量链（源）**

| 常量 | 行号 |
|---|---|
| `HIDDEN = 6144 / H_Q = 64 / H_KV = 4` | BM:L185-187 |
| `H_IDX = 4 / H_IDX_K = 1 / D = 128` | BM:L188-190 |
| `DENSE_LAYERS = 3 / SPARSE_LAYERS = 57` | BM:L199-200 |
| `PROJ_MAIN_DIM = H_Q*D + 2*H_KV*D = 9216` | BM:L204 |
| `PROJ_IDX_DIM = H_IDX*D + H_IDX_K*D = 640` | BM:L206 |
| `PROJ_FUSED_DIM = 9216 + 640 = 9856` | BM:L208 |
| `静态自检 `PROJ_FUSED_DIM==9856 / PROJ_MAIN_DIM==9216 / PROJ_IDX_DIM==640 / w_fused.shape==(9856,HIDDEN) / w_dense.shape==(9216,HIDDEN)`` | BM:L2822-2833, L3021-3023 |

**主形状证据链**

1. `"w_o": randn(HIDDEN, H_Q * D, dtype=dt),  # [6144, 8192]`（源码自带注释） — BM:L733
2. H_Q*D = 64*128 = 8192；HIDDEN = 6144 — BM:L185-186, L190

**decode 路径证据（证明这个 GEMM 每层每步必经）**

- decode：`_trace_decode_staged`（“Stage-level trace for decode (sparse and dense)”）两个 Q=1 步循环 → `with record_function(f"o_proj_step{step}"): F.linear(attn.reshape(batch, H_Q*D), ctx["w"]["w_o"])` — BM:L2221-2226, L2283-2284
- prefill/timing：`with record_function("o_proj"): hidden_out = F.linear(attn_out.reshape(-1, H_Q*D), ctx["w"]["w_o"])` — BM:L2217-2218
- 另有 eager 计时路径 `hidden_out = F.linear(attn_out.reshape(-1, H_Q*D), weights["w_o"])` — BM:L1072, L1118

**alt 形状与理由**

- `w_qkv_idx_fused` = `cat(w_q, w_k, w_v, w_idx_q, w_idx_k, dim=0)` = **[9856, 6144]**
  - BM:L715-717（文档串 BM:L702 `w_qkv_idx_fused: [9856, HIDDEN]`，占位声明 BM:L740，常量 BM:L204/L206/L208，自断言 BM:L2822/L2828）。decode 调用点 BM:L2229（`record_function(f"projection_step{step}")`，sparse 分支）。理由：SGLang `_FusedQKVIndexProj` 把 2 个 GEMM 合成 1 个（BM:L29-31），**57/60 层**的 decode 每步必经；`N/128 = 77` 是一个非 2 的幂的奇数 tile 数，正好检验 kernel 对“不整除 64/128 CTA 网格”的处理（S4 的 128 CTAs 无 pad 话题）。
- `w_qkv_fused` = `cat(w_q, w_k, w_v, dim=0)` = **[9216, 6144]**
  - BM:L711-712（文档串 BM:L700，占位声明 BM:L738，自断言 BM:L2824/L2831）。decode 调用点 BM:L2234（dense 分支）。理由：dense 层（3/60）的同族形状，`N/128 = 72`；与 alt1 只差 index 的 640 行，是一组天然的“同 K 不同 N”对照，可以干净地量出 N 方向 tile 量化的边际成本。

**考虑过但淘汰的候选（诚实记录）**

- `w_q` = `[8192, 6144]`、`w_k`/`w_v` = `[512, 6144]`、`w_idx_q` = `[512, 6144]`、`w_idx_k` = `[128, 6144]`（BM:L730-736）
  - **不在被测路径上**：源码明确写 “w_q/w_k/w_v/w_idx_q/w_idx_k are retained ONLY for the independent FP32 reference forward (ref_layer_forward); they never enter the measured kernel”（BM:L708-709）。生产 decode 只用 fused 版本。
- `w_idx_fused` = `[640, 6144]`（BM:L713-714, L739）
  - 源码自己标注 `(legacy, unused in production)` / `— legacy, unused in production`（BM:L701, L739）。对齐合法（640 = 5×128）但不在生产路径上。

---

## 7. 规则 3 淘汰汇总（N 或 K 不是 128 倍数的层）

这些层**确实在 decode 路径上**，但被规则 3 挡掉了。列出来是为了让后来者知道“为什么没有它们”，
以及如果将来要引入 pad 形状，pad 量是多少。

| 模型 | 层 | 原始 (N,K) | 问题 | 若强行选用的 pad 后 N |
|---|---|---|---|---:|
| kimi_k3 | `fused_qkv_a` | (2112, 7168) | N = 16.5×128 | 2176 (+3.0%) |
| kimi_k3 | `b_proj` | (96, 7168) | N = 0.75×128 | 128 (+33.3%) |
| qwen36 | `ba_weight` | (64, 2048) | N = 0.5×128 | 128 (+100%) |
| glm52 | `wkv_a` | (576, 6144) | N = 4.5×128 | 640 (+11.1%) |
| glm52 | `index_gate` | (32, 6144) | N = 0.25×128 | 128 (+300%) |
| deepseek_v4_pro | `w_proj` (CSA indexer) | (64, 7168) | N = 0.5×128 | 128 (+100%) |

没有一个模型的**主 o_proj** 落在这张表里——5 个 o_proj 全部天然 128 对齐，所以规则 1 与规则 3 无冲突，
`models.json` 里也就没有任何 pad 注记。

---

## 8. decode 的 M 口径（给 kernel / harness agent）

5 个 harness 的 decode case **完全同构**：`batch = 64`，`q_len = 2`，并且都强制拆成**两个顺序的普通 Q=1 步**，
绝不是一次 Q=2 的 prefill 式调用。所以：

```
单次 decode GEMM launch 的真实 M = batch * 1 = 64      （不是 128！）
一个 decode case 会连续发 2 次这样的 launch
```

| 模型 | decode case 定义 | “两个 Q=1 步”的强制证据 | GEMM 的 M 来自 |
|---|---|---|---|
| kimi_k3 | `B=64 Q=2(=2xQ1)` BM:L41-42, L109 | `ordinary KDA decode production call must be Q=1` BM:L902；`ordinary decode must call two sequential Q=1 steps` BM:L1538；`Q=2 is two ordinary, stateful Q=1 steps` BM:L2528；BM:L2726, L2785 | `linear(normed[B,1,HD], o_proj)` BM:L1072 / `linear(gated_out, o_proj)` BM:L1611 |
| qwen36 | `batch=64, q_len=2` BM:L84-89 | `decode reporting horizon must contain exactly two Q=1 steps` BM:L466-467（GDN）/ L1062-1063（dense） | `normalized.reshape(batch, GDN_V_DIM)` BM:L527 / `gated.reshape(batch, DENSE_Q_DIM)` BM:L1119 |
| glm52 | `batch=64, q_len=2` BM:L85 | `_forward_layer_one decode requires exactly one query token` BM:L789-790；`forward_layer` 对 decode 循环 2 次 BM:L1192-1202 | `tokens = batch * q_len` = 64*1 BM:L791, 用于 BM:L1044 |
| deepseek_v4_pro | `batch=64, q_len=2` BM:L130 | `Two sequential ordinary-Q1 decode steps` BM:L2371；`assert q_len == 2` BM:L2384；`for step in range(q_len)` BM:L2397 | 每步 `x = data["x_steps"][step]` 形状 `[B, HIDDEN]` BM:L2398, 用于 BM:L2586 |
| minimax_m3 | `batch=64, q_len=2` BM:L212 | `for step in range(2)` BM:L2226 | `attn.reshape(batch, H_Q*D)` BM:L2284 |

**含义**：harness 只提供了 **M = 64** 这一个真实 decode 工作点。CONTRACT §1 的 M 阶梯
（`M_TILE ∈ {8,16,32,64,128}`，sweep `M = 1,2,4,8,16,32,64,128`）是 **kernel 侧的编译期阶梯**，
其中 M=64 与真实模型对齐，M=1..8 与 M=128 是为了画出完整曲线的外推点，不是从这 5 个 harness 读出来的。
写文章时请把这点讲清楚，别把 M=1 说成“某模型的真实 decode 批量”。

---

## 9. 冷权重 rotating set 提示（CONTRACT §6）

CONTRACT §6 要求 ISO/B2B 协议用旋转 buffer 集保证 working set ≥ 512 MB：
`sets = max(13, ceil(512MB / weight_bytes))`。按上表的权重字节（fp8 e4m3 = 1 B/元素）算出：

| # | id | 角色 | N×K | 权重字节 | sets | 旋转集总显存 |
|---:|---|---|---|---:|---:|---:|
| 1 | `kimi_k3` | main | 7168×12288 | 88080384 | 13 | 1.07 GiB |
| 2 | `kimi_k3` | alt1 | 49152×7168 | 352321536 | 13 | 4.27 GiB |
| 3 | `kimi_k3` | alt2 | 18432×1536 | 28311552 | 19 | 0.50 GiB |
| 4 | `qwen36` | main | 2048×4096 | 8388608 | 64 | 0.50 GiB |
| 5 | `qwen36` | alt1 | 12288×2048 | 25165824 | 22 | 0.52 GiB |
| 6 | `qwen36` | alt2 | 8192×2048 | 16777216 | 32 | 0.50 GiB |
| 7 | `glm52` | main | 6144×16384 | 100663296 | 13 | 1.22 GiB |
| 8 | `glm52` | alt1 | 16384×2048 | 33554432 | 16 | 0.50 GiB |
| 9 | `glm52` | alt2 | 2048×6144 | 12582912 | 43 | 0.50 GiB |
| 10 | `deepseek_v4_pro` | main | 7168×16384 | 117440512 | 13 | 1.42 GiB |
| 11 | `deepseek_v4_pro` | alt1 | 65536×1536 | 100663296 | 13 | 1.22 GiB |
| 12 | `deepseek_v4_pro` | alt2 | 2048×7168 | 14680064 | 37 | 0.51 GiB |
| 13 | `minimax_m3` | main | 6144×8192 | 50331648 | 13 | 0.61 GiB |
| 14 | `minimax_m3` | alt1 | 9856×6144 | 60555264 | 13 | 0.73 GiB |
| 15 | `minimax_m3` | alt2 | 9216×6144 | 56623104 | 13 | 0.69 GiB |

**两个要提醒 harness agent 的点：**

1. `qwen36` 主形状只有 8 MiB，需要 **64 个旋转集**才能到 512 MiB —— set 数远超其他模型，
   分配/释放策略要注意（64×8 MiB = 512 MiB，可以接受）。
2. `kimi_k3` 的 alt1 `fused_qkvg` 单个权重就有 **336 MiB**，`max(13, ceil(512/336)) = 13` 集 → **4.27 GiB**。
   在 GPU3 上跑这个形状前请先确认显存余量；建议主形状优先，alt 形状按需单独跑。

（另外注意：CONTRACT §6 的 `bandwidth_gbps = (N*K + M*K + 2*M*N) bytes / latency` 里权重项按 fp8 1 B 计，
与上表的权重字节口径一致。）

---

## 10. 用户称呼 ↔ 目录 id ↔ display 对照

CONTRACT §2 的“用户称呼”列与 harness **自报**的模型名不一致。`models.json` 的 `display` 字段一律采用
**harness 自己写进产物里的名字**（可指到行号、不编造），对照如下：

| CONTRACT §2 用户称呼 | 目录 id | `display`（本文件采用） | 自报出处 | hidden |
|---|---|---|---|---:|
| kimi k3 | `kimi_k3` | `Kimi-K3` | BM:L1742, L2131（`"model": "Kimi-K3"`）；`model_spec.json` `display_name` | 7168 |
| qwen3.8 max | `qwen36` | `Qwen3.6-35B-A3B` | BM:L37（`MODEL_NAME = "Qwen3.6-35B-A3B"`） | 2048 |
| glm5.3 | `glm52` | `GLM-5.2` | BM:L2185（`"model": "GLM-5.2"`）；docstring BM:L1；`model_spec.json` | 6144 |
| dpsk v4.1 flash | `deepseek_v4_pro` | `DeepSeek-V4-Pro` | BM:L3834, L3846（`"model": "DeepSeek-V4-Pro"`） | 7168 |
| minimax m3 | `minimax_m3` | `MiniMax-M3` | BM:L2335, L3082, L3100；`model_spec.json` `display_name` | 6144 |

`hidden` 列与 CONTRACT §2 表格给的 7168 / 2048 / 6144 / 7168 / 6144 **逐项一致**，
并且都由 harness 常量独立验证（BM:L69 / L38 / L56 / L63 / L185）。
画图 agent 若要用中文/用户口径的标题，请自行按本表映射，不要改 `models.json`。

---

## 11. `models.json` schema 说明

严格按 CONTRACT §2，**没有增加任何额外 key**（生成脚本里有 schema key 集合门禁）：

```
顶层            : models[], peak_bw{}
models[i]       : id, display, hidden, gemm{}, alt[], source{}
gemm            : layer, N, K, why
alt[j]          : layer, N, K            # 无 why，理由写在本文件 §6
source          : file, lines            # lines 是分号分隔的 BM:L 证据串
peak_bw         : spec_tbps, measured_note
```

- `source.file` 是**绝对路径**，指向权威 harness `.py`。
- `source.lines` 用 `BM:L<n>` 记法（沿用辅助源 `model_arch_params.json` 的 notation），
  覆盖：常量定义行、权重构造行、harness 自断言行、约定说明行、decode 调用点行、decode case 定义行、display 名出处行。
  主形状和 alt 形状的行号都在同一个 `lines` 串里，按分号分组。
- `peak_bw.spec_tbps = 4.0`：CONTRACT §7 指定的**主参考线**（HBM3 理论峰值，用户指定，不许替换）。
- `peak_bw.measured_note = "见 tools/peak_bw_probe"`：CONTRACT §7 的辅线（实测可达峰值）**不由本 agent 填数字**，
  这里只留指针。它由 peak-bw agent 的 `tools/peak_bw_probe.cu` 实测后写入 `results/peak_bw.json`。
  本文件写就时的状态：`tools/peak_bw_probe.cu` 已存在，`results/peak_bw.json` **尚未生成**
  （`results/` 为空）。按 CONTRACT §8，plot agent 读不到 `results/peak_bw.json` 时应不画虚线并告警。
  该文件属于别的 agent 的 write scope，本 agent 不读也不写它，所以这里只描述指针语义、不引用其内容。

---

## 12. 复现校验

形状提取是纯静态的，任何人可以这样复核（不需要 GPU）：

```bash
W=/home/admin/workspace/hengfeng_data/weave_new
B=$W/DSA-learn/baselines/h100

# 1) 看 kimi_k3 o_proj 的常量、构造、自断言三层
sed -n '69,74p;126p;136p;2990p;2997p;3012p' $B/kimi_k3/full_attention_benchmark.py

# 2) 看 glm52 的转置写法（关键坑）
sed -n '56,64p;107p;109p;115p;796p;798p;1044,1045p' $B/glm52/full_attention_benchmark.py

# 3) 看 deepseek 的低秩 o_proj 两级 + fp8 GEMM 约定
sed -n '63,71p;111p;445,447p;525,532p;1474,1479p;2305,2311p;2583,2586p;3677,3683p' \
    $B/deepseek_v4_pro/full_attention_benchmark.py

# 4) 看 minimax 的 w_o 与 fused 投影自断言
sed -n '185,208p;696,718p;725,740p;2283,2284p;2822,2833p' $B/minimax_m3/full_attention_benchmark.py

# 5) 看 qwen36 两个变体的 o_weight
sed -n '37,61p;256,261p;264,295p;314,324p;523,529p;1115,1119p' $B/qwen36/full_attention_benchmark.py

# 6) 校验 models.json 的 128 对齐与 schema
# 注意：heredoc 用 <<'EOF'（带引号）时 shell 不展开 $W，所以这里写绝对路径
python3 - <<'EOF'
import json
d=json.load(open('/home/admin/workspace/hengfeng_data/weave_new/DSA-learn/mk/h20/cutedsl/decode_gemm/models/models.json'))
assert set(d)=={'models','peak_bw'} and len(d['models'])==5
for m in d['models']:
    assert set(m)=={'id','display','hidden','gemm','alt','source'}
    assert set(m['gemm'])=={'layer','N','K','why'} and 1<=len(m['alt'])<=2
    for e in [m['gemm']]+m['alt']:
        assert e['N']%128==0 and e['K']%128==0, (m['id'],e)
    assert m['gemm']['N']==m['hidden'], m['id']   # 规则1：N = hidden_size
    print(m['id'], m['gemm']['N'], m['gemm']['K'], 'OK')
EOF
```

---

## 13. write scope 声明

本 agent 只写了：

- `models/models.json`（10610 bytes）
- `models/README.md`（本文件）

没有修改 `CONTRACT.md`、`kernels/`、`bench/`、`tools/`、`analysis/`、`results/`，也没有触碰只读的 `shared/` 参考实现或任何 baseline 源文件。
没有编译、没有占用 GPU。
