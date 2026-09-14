# `decode_gemm/kernels/` 设计说明

> 读者定位：准备把这套东西写成知乎文章的读者。所有数字都是本机（H20，CUDA 12.8，
> `tools/gpurun_dg.sh` 锁频 1830 MHz）真实测出来的，出处见 §13 与 `SMOKE_RESULTS.md`。
> 参考实现（只读 oracle）：`shared/swapab_m1_kernel_v2.cuh`（945 行，PDL 后 85.3% of 4 TB/s），
> 设计推导见 `shared/ANALYSIS_decode_gemm.md`。本目录是它的**干净重写**：
> N/K/M 全部 runtime 化、M_TILE 从 4 扩到 128、七个技术步骤全部收敛成同一个模板的开关。

---

## 0. 文件清单

| 文件 | 行数 | 职责 |
|---|---|---|
| `decode_gemm.cuh` | 1160 | 唯一的 kernel + `KernelConfig` + tensor map 构造 + host launcher（契约写 ≤ ~1100 行；S 路 split-K 泛化 +88 行，其中 ~60 行是注释，见 §15） |
| `weight_prepack.cuh` | 123 | raw `[N,K]` 与 tile-major packed 两种布局的定义、prepack kernel/launcher |
| `sm90_compat.cuh` | 134 | 从参考实现原样复制的 SM90 底层封装（`ld_shared`/`st_shared`/`make_smem_desc`/wgmma fence-commit-wait/`tma_copy_2d`/driver 错误检查），命名空间保持 `fp8_swapab_wgmma::sm90_compat` |
| `smoke_test.cu` | 636 | 完成门禁：50 个 case × 4 个形状（含 S∈{2,4,8} 的 12 个 S 路 split-K case），naive fp32 host 参考，rel_l2 ≤ 2e-3 |
| `build_smoke.sh` / `run_smoke.sh` | 17 / 18 | 编译（CONTRACT §0 的命令模板）/ 经 `tools/gpurun_dg.sh` 运行 |
| `SMOKE_RESULTS.md` | — | 37 case 全绿的结果表（rel_l2 + grid/block/smem）+ 覆盖矩阵；§5 是 S 路 split-K 追加的 13 case |
| `SMOKE_LOG_skS.txt` | — | S 路 split-K 门禁那次（14:11, BORROW GPU2）的原始输出：50 case 全绿 |
| `SMOKE_LOG.txt` | — | 全绿那次（06:39, CLEAN GPU0）的原始输出；`SMOKE_LOG_rerun.txt` / `SMOKE_LOG_v2.txt` 是另外两次独立复现（37 case 的 rel_l2 逐位相同），`SMOKE_LOG_quick.txt` 是修复前的失败现场 |
| `_dev/` | — | 排查用探针与预算表生成器，每个文件的用途和结论见 `_dev/README.md` |

CUTLASS 只用到 `cutlass/arch/barrier.h`、`cutlass/arch/reg_reconfig.h`、
`cute/arch/copy_sm90_desc.hpp`、`cute/arch/copy_sm90_tma.hpp` 四个头。

---

## 1. 问题：这是一个纯带宽问题

`out[M,N] = act[M,K] @ W[N,K]^T`，fp8 e4m3 输入、per-128(K) 反量化 scale、bf16 输出。
decode 场景 M ∈ {1..128}，N/K 是几千到一万六。

以 kimi_k3 的 `N=7168, K=12288`、M=1 为例：

| 量 | 值 |
|---|---|
| 权重读取 | 7168 × 12288 × 1 B = **88.1 MB** |
| 激活读取 | 1 × 12288 = 12 KB（可忽略） |
| 输出写出 | 7168 × 2 = 14 KB（可忽略） |
| 计算量 | 2·M·N·K = 176 MFLOP |
| 算术强度 | **2·M = 2 FLOP/Byte** |
| 理论下限 | 88.1 MB / 4.0 TB/s = **22.0 µs** |

H20 的 fp8 峰值约 282.7 TFLOP/s，ridge point 在 ~70 FLOP/B。AI = 2 差了一个半数量级，
所以 WGMMA 吞吐完全无关紧要，**唯一目标是让 88 MB 冷权重以尽可能接近 4 TB/s 的速度流过**。

这件事难在三点（`ANALYSIS_decode_gemm.md` §1.2 有完整测量）：

1. **kernel 太短**（十几 µs），启动 / 首个 TMA 的冷延迟 / 尾部排空 / host 往返都摊不掉；
2. **权重必须是冷的**（88 MB × 多份旋转 ≫ 60 MB L2），否则测出来的是 L2 带宽；
3. **并行度只能从 N 维来**：并发 DRAM 流数 ≈ CTA 数 = `ceil(N/BM) × gridDim.y`，
   实测 5 KB op 时 78 CTA→36%、154 CTA→60%、308 CTA→83% 的 4 TB/s 占比。

于是所有优化都在回答同一个问题：**怎样在十几微秒内、用有限的 CTA 数，把每个 TMA op
做得足够大、把流水线的头尾藏起来。**

---

## 2. 基石：为什么必须 swap-A/B

### 2.1 wgmma 的形状约束

`wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3` 的 A 侧**固定 64 行**。
若把激活放 A 侧（M=1），64 行里只有 1 行有效，tensor core 吞吐浪费 98%。
所以交换：

```
A 操作数 = 权重 tile   （64 wgmma 行 == 64 个输出行 n，实际有效 BM ≤ 64 行）
B 操作数 = 激活 tile   （8 wgmma 列 == 8 个激活行 m）
```

B 侧只有 8 行，看起来也浪费，但激活总量只有 M×K 字节（M=1 时 12 KB），
常驻 smem/L2，浪费的是"字节"而不是"带宽"——权重才是流量主体。

### 2.2 线程 → 输出元素映射（`decode_gemm.cuh:683-705`）

一个 math warp group（4 warp × 32 lane）覆盖 A 侧的 64 行：

| 量 | 公式 | 含义 |
|---|---|---|
| `row0` | `local_warp*16 + lane/4` | 该线程第 1 个输出行（n 方向，0..63） |
| `row8` | `row0 + 8` | 第 2 个输出行 |
| `logical_m_base` | `(lane & 3) * 2` | 该 lane 负责的 m（0,2,4,6） |
| `raw[mt][0..3]` | — | `(row0,m) (row0,m+1) (row8,m) (row8,m+1)` |

即每线程 4 个累加器 × `M_TILES = M_TILE/8` 个 n8 wgmma 组。

### 2.3 M_TILE 阶梯：v2 的 `LogicalM<=4` 是遗留保守断言

v2 写死 `static_assert(LogicalM >= 1 && LogicalM <= 4)`，但它的 B-tile 从来都是 n8，
`logical_m0=(lane&3)*2, logical_m1=+1` 本来就覆盖 m=0..7。所以本重写：

* **最小 tile 就是 8**：M=1..8 共用 `M_TILE=8`，用 runtime 谓词 `m < rt.M` 过滤
  （`:740-747` 把谓词压成两个 bitmask，M=1 时只有 `lane&3==0` 的 lane 参与 fold/store，
  与 v2 的 M=1 特化路径生成同样的指令）；
* **M_TILE > 8** 时每 k_block 发 `M_TILES` 组 n8 wgmma（`:792-799`），
  累加器变成 `raw[SUBS][M_TILES][4]` + `partial[M_TILES][4]`；
* 激活的 smem/TMA 天然支持：B 侧第 `mt` 组就是 subtile 内偏移 `mt*8*128` 字节
  （`:795`），TMA 一次拉 `box = {128, act_rows, SUBS}`，`act_rows = min(M_TILE, M)`。

---

## 3. 流水线骨架（S0）

```
CTA = (NUM_MATH_WG + 1) 个 warp group：前 NUM_MATH_WG 个做 math，最后 1 个只做 TMA producer
kStages = NUM_MATH_WG × STAGES_PER_WG 个 smem stage
stage      = (group % STAGES_PER_WG) * NUM_MATH_WG + math_wg      (:614, :752)
generation = group / STAGES_PER_WG
producer (:622-:641) : empty->wait((generation+1)&1) → TMA → full.arrive_and_expect_tx(bytes)
math     (:758-:866) : full->wait(generation&1)      → wgmma+fold → empty->arrive() (lane0)
```

要点（都照 v2 语义，不改）：

* **同一 warp group 的相邻 group 落在不同 stage**（stride = NUM_MATH_WG），
  所以 producer 的 4 个 warp 可以用 `stage % 4 == producer_warp` 平均分工（`:625-627`）；
* **`arrive_and_expect_tx` 在 TMA 之后发**：mbarrier 的 tx-count 是有符号的，
  晚一点 announce 不影响相位完成，却让"发 op"这一步不在关键路径上（`:648-650`）；
* **相位奇偶**：`mbarrier.try_wait.parity(p)` 的语义是"完成相位计数 & 1 != p"，
  所以全新 barrier 上 `wait(1)` 立即返回、`wait(0)` 阻塞到第一次完成 —— 这正是
  `empty->wait((gen+1)&1)` 与 `full->wait(gen&1)` 能对上的原因；
* **寄存器再配置**：producer `warpgroup_reg_dealloc<PROD_REGS>`（`:607`）、
  math `warpgroup_reg_alloc<MATH_REGS>`（`:666`）。这是把 168 个累加器寄存器
  塞进 384 线程 CTA 的前提；
* **padding tile 也要走完流水**：producer 对越界 tile `continue`（`:616`），
  math 侧用 `output_tile_active` 只关掉 store 与 scale 读取，所有下标都 clamp 到
  合法 tile/scale group（`:674-703`），保证既不越界也不死锁。

---

## 4. 七步技术：每步的「为什么」

### S1 — PDL（programmatic dependent launch）｜`:654-660`、`:1051-1055`

**为什么**：十几 µs 的 kernel 里有一块与数据量无关的固定开销（host event 往返 2–3 µs、
kernel 启动 ~1 µs、首个 TMA 的冷延迟 1.38 µs、尾部排空 ~1.5 µs）。
PDL 让**下一个 kernel 的 prologue**（descriptor prefetch、barrier init、scale 合并）
在上一个 kernel 还没排空时就开始跑，把这块开销折进重叠区。
v2 实测：batch 16.09 µs → PDL 12.90 µs，**−19.9%**，4 TB/s 占比 68.5% → 85.4%。

**实现**：两处。① producer 发完本 CTA 最后一条 TMA 后 `griddepcontrol.launch_dependents`
（`:654-660`）；② host 侧给 launch 挂 `cudaLaunchAttributeProgrammaticStreamSerialization`
（`:1051-1055`）。`PDL=false` 时一条 trigger 都不发（`:657` 的 `if constexpr`）。

**代价/风险**：一旦开了 PDL，相邻两次 launch **会真的重叠**，因此
"连发两次同一个 kernel 且写同一块 output" 在语义上是 data race（两边写同样的值，
实践中无害，但 harness 做正确性检查时应当单发、或让两次 launch 写不同 buffer）。
`smoke_test.cu` 故意连发两次并各自校验，用来抓"状态没复位"这类真 bug。

### S2 — blockK / `SUBS_PER_STAGE`｜`:634-642`（TMA box 第 3 维）、`:774-809`（math 侧）

**为什么**：冷数据的单流带宽强烈依赖**单个 TMA op 的字节数**：5 KB→38%、10 KB→60%、
20 KB→82%、40 KB→92%（4 TB/s 占比）。同时 barrier 事务数、循环迭代数也随 op 数线性变化。
`SUBS_PER_STAGE=2` 让一条 TMA 覆盖 2 个连续 k_block：
`BM=64` 时单 op 8 KB→16 KB，chunk 数与 barrier 事务数减半。
v2 归因（`ANALYSIS` §4A.2b）：这一步 −1.73 µs（p50），其中
**"同步/迭代减半" 占 40%、"单 op 变大" 占 60%**。

**实现**：一个 `k_block` = 128 B = 4 × wgmma k32；一个 chunk = `SUBS` 个 k_block。
math 侧 `raw[SUBS][M_TILES][4]`，每 sub 一个 `wgmma.commit_group`，
最后一次 `warpgroup_wait<0>()` 收全部（`:811-824`），然后按 sub 逐 k_block 做 scale fold。
事务字节 = `SUBS*(BM*128 + act_rows*128)`（`:551`）。

**硬约束**：`(K/128) % (SUBS × WGsPerOutputTile) == 0`（`:280-284`）。
甜区是单 op 10–16 KB：`SUBS=4` 配 `STG=4` 实测 21.31 µs，反而不如 `SUBS=2/STG=3` 的 18.61 µs
——op 变大换来的收益会被 smem 挤压掉的并发度吃光。

### S3 — prepack（tile-major packed 权重）｜`weight_prepack.cuh`、`:362-396`、`:634-642`

**为什么**：raw `[N,K]` 布局下，一个 CTA 要的数据是 BM 行、每行 `SUBS*128` 连续字节，
行与行相隔 K 字节。**布局改动本身不提速**（v1 22.56 µs vs v2 引擎+v1 形状 23.25 µs），
它的价值是让 S2 的"单 op 变大"成为可能：packed 之后
`(tile, 连续 k_block)` 在显存里相邻（stride = `BM*128`），
一个 CTA 的 `(tile, 全部 K)` 是一段连续的 `BM×K` 字节纯顺序流。

**布局**（`weight_prepack.cuh:14-31`，与 v2 完全一致）：

```
packed[((tile * num_k_blocks + kb) * BM + row_in_tile) * 128 + (k % 128)] = weight[tile*BM + row_in_tile][k]
```

`N` 不是 BM 的倍数时 pad 到 `padded_n = ceil(N/BM)*BM`，pad 行**写 0**
（`weight_prepack.cuh:98-101`）：math 侧会谓词掉，但 0 比未初始化 smem 更安全（不会 NaN）。

**raw 路径怎么做到同样的单大 op**（重要，见 §9）：把 K 拆成 `{128 字节, num_k_blocks}` 两维，
用 rank-4 map + 非单调 stride `{K, 128, N*K}` + `box {128, BM, SUBS, 1}`。
两种布局发出的 TMA 指令、smem 落点、wgmma 描述符**完全一样**，只有坐标含义不同
（`:634-642`）。所以 S2 在 raw 上也能拿到完整收益，S3 的收益就回归"省掉一次预处理"
与"更规整的 L2 访问"。

### S4 — tile_stage（BM / WG / STG 重配）｜纯 config，无独立代码

**为什么**：并发 DRAM 流数 = CTA 数。`BM=64`→96 CTA、`BM=32`→192 CTA（过订阅，24.99 µs）、
`BM=48`→128 CTA（**N=6144 整除、零 pad、落在甜区**，18.77 µs）。
同时把 math warp group 从 4 降到 2（线程 640→384）、`MATH_REGS` 112→80、`STG` 2→3：
流水线深度 3 是实测最优（depth1 54.1% → depth3 61.6% → depth7 59.5%）。
v2 归因：这一步 −1.90 µs（p50）/ −1.78 µs（PDL 口径），是 kernel 内最大单项。

**BM 怎么选（本次重写新增的整除性规则）**：优先选能整除 N 的 BM ∈ {32,40,48,56,64}，
其次选 pad 最小的。`N=7168` 不能被 48 整除（149.33）但能被 **56** 整除（=128 CTA，零 pad）；
`N=2048` 只能被 32/64 整除，选 **32**（64 CTA）并靠 S6 补 CTA。详见 §6.1。

### S5 — epi_overlap（第二次 trigger）｜`:882-888`

**为什么**：S1 的 trigger 在 producer 侧，覆盖的是"权重流已经全部发出"之后的时间；
但本 kernel 的**尾巴**（bf16 store，S6 时还有 fp32 workspace 往返）仍然独占 SM。
math 侧在 store 之前再发一次 `launch_dependents`，把下一个 kernel 的 prologue
提前到与我们的 epilogue 重叠。

**实现**：`if constexpr (kEpiOverlap) pdl_launch_dependents();`，位置在 mainloop 结束之后、
Phase A/B 之前（`:888`）。要求 `PDL=true`（`:174-175` static_assert），host 侧的
PDL attribute 由 `kPdl` 控制。

### S6 — splitk（CTA 级 K 切分）｜`:542-549`、`:944-991`

**为什么**：CTA 数 = `ceil(N/BM)`。小 N 模型（qwen36 `N=2048`）即使 BM=32 也只有 64 CTA
< 78 SM，**N 维已经切无可切**；此时沿 K 切一刀，`gridDim.y = 2`，并发流数翻倍。
这正是 CONTRACT 说的"SPLIT_K_CTA 的主场"。

**协议**（`:944-991`）：

1. CTA `y` 只跑 `[y*chunks/2, (y+1)*chunks/2)` 这段 chunk（`:542-549`），
   barrier/generation 各自从 0 开始，流水公式不变；
2. epilogue 先把 fp32 结果写进 `splitk_ws[y][m*N+n]`（`:956-961`）；
3. `__threadfence()`（release）→ 本 tile 的 named barrier（`:963-965`）；
4. 每 tile**一个** leader 线程 `atomicAdd(splitk_sem + output_tile, 1)`，
   把 `old == 1` 写进 smem flag（`:967-973`）→ 再一次 named barrier 广播（`:974`）；
5. 看到 flag 的那个 CTA 是"后到者"：`__threadfence()`（acquire）→ 读对方的
   `ws[1-y]` 相加 → `__float2bfloat16_rn` 写 output（`:976-985`）；
6. leader `atomicExch(sem, 0)` **自复位**（`:986`），下一次 launch 无需 host 清零
   （`smoke_test.cu` 每次跑完都回读 semaphore 断言全 0）。

**约束**：`(K/2)/(128*SUBS) % WGsPerOutputTile == 0` 且 `chunks/2 >= STAGES_PER_WG`
（`:285-292`）。

> **本节描述的是 2 路的历史形态。2 路已经泛化成 S 路（`SPLITK_FACTOR`，S∈{2,4,8}）：
> `gridDim.y = S`、ticket 判据变成 `previous == S-1`、reducer 累加其余 S-1 份 ws、
> `ws` 布局 `[S][M][N]`、自复位不变。协议、三笔账与实测甜区见 §15。**
**给 harness 的接口约定**：`splitk_ws` 需要 `2*M*N` 个 float；`splitk_sem` 按 CONTRACT 分配
`M*N` 个 int32 即可，但 kernel 只用前 `ceil(N/BM)` 个（每输出 tile 一个，不是每元素一个
——每元素 semaphore 在 M=128 时会有 78 万次 atomic，太贵）。

---

## 5. 模板参数逐个说明（`KernelConfig`，`:139-297`）

```cpp
template <int M_TILE, int OUTPUT_ROWS_PER_CTA, int NUM_MATH_WG, int MATH_REGS,
          int STAGES_PER_WG, int TILES_PER_CTA, int SUBS_PER_STAGE,
          bool PREPACK, bool PDL, bool EPI_OVERLAP, bool SPLIT_K_CTA,
          int PROD_REGS = 32, bool SINGLE_PRODUCER_WARP = false,
          bool PRELOAD_ACT = false, bool SCALE_IN_SMEM = (M_TILE <= 16)>
struct KernelConfig;
```

| 参数 | 含义 | 取值 / 约束 |
|---|---|---|
| `M_TILE` | runtime M 的编译期上界；`M_TILES = M_TILE/8` 组 n8 wgmma | 8/16/32/64/128 |
| `OUTPUT_ROWS_PER_CTA` (BM) | 一个输出 tile 的行数 = wgmma A 侧有效行数 | ≤64 且 %8==0；决定 CTA 数与 pad |
| `NUM_MATH_WG` | math warp group 数（另有 1 个 producer WG） | 1/2/4 |
| `MATH_REGS` | `warpgroup_reg_alloc` 的目标值 | %8==0，[24,256]，**受 §8 的硬上限约束** |
| `STAGES_PER_WG` | 每 WG 流水深度；`kStages = WG × STG` | 实测 3 最优（M_TILE=8） |
| `TILES_PER_CTA` | 每 CTA 的输出 tile 数；`WGsPerOutputTile = WG/TILES` 即 intra-CTA split-K | `WG % TILES == 0` |
| `SUBS_PER_STAGE` | blockK = `SUBS × 128`，一条 TMA / 一个 stage 覆盖的 k_block 数 | `(K/128) % (SUBS*WGsPerTile) == 0` |
| `PREPACK` | true = tile-major packed；false = raw `[N,K]` | 两种都必须正确（§9） |
| `PDL` | producer 侧 `launch_dependents` + host PDL attribute | S1 |
| `EPI_OVERLAP` | math 侧第二次 trigger | S5，需 `PDL=true` |
| `SPLIT_K_CTA` | split-K 开关（历史 bool，`true` 等价 `SPLITK_FACTOR=2`） | S6，需 `splitk_ws/sem` |
| `SPLITK_FACTOR` | **S 路宽度**：`gridDim.y = S`，每 CTA 跑 `1/S` 段连续 K | 第 16 个模板参数（追加在末尾、默认 0）；0=跟随 `SPLIT_K_CTA`，1=关，2/4/8=开；见 §15 |
| `PROD_REGS` | producer 的 `reg_dealloc` 目标 | 默认 32 |
| `SINGLE_PRODUCER_WARP` | 只用 warp0 发全部 TMA，其余 return | v2 实测在 BM48/SUBS2 无收益，默认 false |
| `PRELOAD_ACT` | 整段激活一次载入 smem（box 第 3 维 = num_k_blocks） | v2 实测 +0.18 µs，默认 false；CONTRACT 限定 `M_TILE=8 && PREPACK` |
| `SCALE_IN_SMEM` | prologue 预乘 `w_scale*a_scale` 进 smem（v2 路径）vs 主循环 `__ldg` 直读 | 默认 `M_TILE<=16`；M_TILE≥32 必须 false（§7） |

**数值路径两条完全等价**：`combined = w_scale * a_scale`（fp32 乘法，确定性），
每 k_block `partial = fmaf(combined, raw, partial)`，最后 `__float2bfloat16_rn`
（`:827-874`、`:936-942`）。smoke 里两条路径的 rel_l2 相同（6.912e-4）。

---

## 6. Canonical config 表

### 6.1 BM 的整除性选择（每形状）

BM 决定两件事：CTA 数 = `ceil(N/BM)`（并发 DRAM 流数），以及 pad 行数 = `ceil(N/BM)*BM - N`
（白读的权重字节）。规则：**优先选能整除 N 的 BM，其次选 pad 最小的；在整除的候选里挑
让 CTA 数落在 100–160 的那个**（v2 实测 192 CTA 过订阅反而慢 32%）。

| 形状 | N | /64 | /56 | /48 | /40 | /32 | S0–S3 用 | S4–S6 用 |
|---|---|---|---|---|---|---|---|---|
| kimi_k3 | 7168 | 112 ✓ | **128 ✓** | 149.3 (pad 32) | 179.2 (pad 32) | 224 ✓ | BM=64 | **BM=56**（128 CTA，零 pad） |
| dpsk_v4_pro | 7168 | 112 ✓ | **128 ✓** | 149.3 | 179.2 | 224 ✓ | BM=64 | **BM=56** |
| glm52 | 6144 | 96 ✓ | 109.7 (pad 16) | **128 ✓** | 153.6 (pad 16) | 192 ✓ | BM=64 | **BM=48**（128 CTA，零 pad） |
| minimax_m3 | 6144 | 96 ✓ | 109.7 | **128 ✓** | 153.6 | 192 ✓ | BM=64 | **BM=48** |
| qwen36 | 2048 | 32 ✓ | 36.6 (pad 24) | 42.7 (pad 16) | 51.2 (pad 32) | **64 ✓** | BM=64（32 CTA，欠订阅） | **BM=32**（64 CTA）+ **S6 split-K → 128 CTA** |

qwen36 是唯一"N 维切不满 78 个 SM"的形状：BM=32 也只有 64 CTA，所以它的收益主要来自 S6。
`BM=48 + N=7168` 会 pad 到 7200（150 tiles），smoke 里专门跑了一条来验证 pad 谓词
（`S5 epi_overlap bm48 pad`，rel_l2 = 6.989e-4 ✓）。

### 6.2 M_TILE=8 的 canonical 阶梯（每形状一行）

拓扑部分：S0/S1 用 `BM=64, WG=4, REG=112, STG=2, TILES=1, SUBS=1`；
S2/S3 只把 `SUBS` 提到 2；S4 起换 `BM=<形状选值>, WG=2, REG=80, STG=3, TILES=1, SUBS=2`。
开关部分：`PREPACK` 从 S3 起为 true，`PDL` 从 S1 起为 true，`EPI_OVERLAP` 从 S5 起，
`SPLIT_K_CTA` 只有 S6。`PROD_REGS=32`、`SCALE_IN_SMEM=true`（M_TILE=8）、
`SINGLE_PRODUCER_WARP=false`、`PRELOAD_ACT=false` 全程不变。

smem 为 M=1 时的实测字节数（M=8 时激活区 ×8，见 §7 公式）；`grid` 已含 S6 的 `y=2`。

| 形状 | step | BM | grid | threads | wSmem | actSmem | scaleSmem | partSmem | 总 smem |
|---|---|---|---|---|---|---|---|---|---|
| kimi_k3 (7168×12288) | S0/S1 | 64 | 112×1 | 640 | 69632 | 1024 | 6144 | 8192 | 85120 |
| | S2/S3 | 64 | 112×1 | 640 | 135168 | 2048 | 6144 | 8192 | 151680 |
| | S4/S5 | 56 | 128×1 | 384 | 90112 | 1536 | 6144 | 4096 | 101984 |
| | S6 | 56 | 128×2 | 384 | 90112 | 1536 | 6144 | 4096 | 102000 |
| dpsk_v4_pro (7168×16384) | S0/S1 | 64 | 112×1 | 640 | 69632 | 1024 | 8192 | 8192 | 87168 |
| | S2/S3 | 64 | 112×1 | 640 | 135168 | 2048 | 8192 | 8192 | 153728 |
| | S4/S5 | 56 | 128×1 | 384 | 90112 | 1536 | 8192 | 4096 | 104032 |
| | S6 | 56 | 128×2 | 384 | 90112 | 1536 | 8192 | 4096 | 104048 |
| glm52 (6144×16384) | S0/S1 | 64 | 96×1 | 640 | 69632 | 1024 | 8192 | 8192 | 87168 |
| | S2/S3 | 64 | 96×1 | 640 | 135168 | 2048 | 8192 | 8192 | 153728 |
| | S4/S5 | 48 | 128×1 | 384 | 77824 | 1536 | 8192 | 4096 | 91744 |
| | S6 | 48 | 128×2 | 384 | 77824 | 1536 | 8192 | 4096 | 91760 |
| minimax_m3 (6144×8192) | S0/S1 | 64 | 96×1 | 640 | 69632 | 1024 | 4096 | 8192 | 83072 |
| | S2/S3 | 64 | 96×1 | 640 | 135168 | 2048 | 4096 | 8192 | 149632 |
| | S4/S5 | 48 | 128×1 | 384 | 77824 | 1536 | 4096 | 4096 | 87648 |
| | S6 | 48 | 128×2 | 384 | 77824 | 1536 | 4096 | 4096 | 87664 |
| qwen36 (2048×4096) | S0/S1 | 64 | 32×1 | 640 | 69632 | 1024 | 2048 | 8192 | 81024 |
| | S2/S3 | 64 | 32×1 | 640 | 135168 | 2048 | 2048 | 8192 | 147584 |
| | S4/S5 | 32 | 64×1 | 384 | 53248 | 1536 | 2048 | 4096 | 61024 |
| | S6 | 32 | **64×2** | 384 | 53248 | 1536 | 2048 | 4096 | 61040 |

CONTRACT §3 的 M_TILE=8 阶梯（N=6144/K=7168，BM=48）已在 `SMOKE_RESULTS.md` 全部跑通。

### 6.3 M_TILE ≥ 16 的 canonical config

规则（CONTRACT §3）：`TILES_PER_CTA = NUM_MATH_WG`，即 `WGsPerOutputTile = 1`，
每 WG 独立一个输出 tile，**没有 intra-CTA split-K reduce**（否则 `partial` smem = `WG*64*M_TILE*4`
在 M_TILE=128 时要 64 KB）。拓扑统一为 `WG=2, TILES=2, STG=2`，BM 沿用 §6.1 的每形状选值：

| M_TILE | BM（按形状） | WG | TILES | REG | STG | SUBS | SCALE_IN_SMEM | 累加器寄存器 | CTA（kimi/qwen） | 总 smem（kimi, M=M_TILE） |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 56 / 32 | 2 | 2 | 112 | 2 | 2 | true | 24 | 64 / 32 | 102464 / 61504 |
| 32 | 56 / 32 | 2 | 2 | 168 | 2 | 2 | false | 48 | 64 / 32 | 94272 / 69696 |
| 64 | 56 / 32 | 2 | 2 | 168 | 2 | 2 | false | 96 | 64 / 32 | 127040 / 102464 |
| 128 | 56 / 32 | 2 | 2 | 168 | 2 | **1** | false | 128 | 64 / 32 | 98368 / 86080 |

（N=6144 的 glm52/minimax 把 BM 换成 48，smem 见 `_dev/config_table.md`：
M16 86080–102464、M32 86080、M64 118848、M128 94272。）

* **M_TILE=128 必须 `SUBS=1`**：累加器 = `(SUBS+1)*M_TILES*4`，SUBS=2 时要 192 个，
  超过 §8 的 168 硬上限；SUBS=1 时 128 个，刚好塞进 168（仍有 20 B spill，见 §8）。
* **`SCALE_IN_SMEM` 在 M_TILE≥32 必须 false**：scale 表 = `TILES*M_TILE*2*(K/128)*4` B，
  M_TILE=128/K=7168 时 = 57344 B，K=16384 时 = 131072 B，直接吃掉一半 smem。
  false 时 math 线程每 k_block 只 `__ldg` 2 个 w_scale（warp 内广播）+ `2*M_TILES` 个 a_scale
  （L1 常驻，`activation_scales` 总共才 `M*K/128*4` = 28 KB）。
* qwen36（N=2048）在 M_TILE≥16 只有 32 CTA，**建议一律叠 `SPLIT_K_CTA`** 到 64。
* 每步的开关叠加方式与 M_TILE=8 相同（S1 加 PDL、S3 加 PREPACK、S5 加 EPI_OVERLAP、S6 加 SPLIT_K_CTA）。
* 跑不通时的降级顺序：先 `STG 3→2`，再 `SUBS 2→1`，最后 `TILES=WG`（已是默认）。
  本次五个形状 × 全部 step × 全部 M_TILE 共 57 个组合**没有一个需要降级**
  （`_dev/config_table` 全部 `validate() == ok`，最大 smem 153728 B ≤ 227 KB）。

### 6.4 K 的整除性检查（`validate()` 会在 launch 前挡住）

`KB = K/128`。baseline 拓扑（WG=4,TILES=1 → WGsPerTile=4）要求 `KB % (4*SUBS) == 0`；
tuned 拓扑（WG=2,TILES=1 → 2）要求 `KB % (2*SUBS) == 0`；M_TILE≥16（WGsPerTile=1）要求 `KB % SUBS == 0`；
`SPLIT_K_CTA` 额外要求 `KB` 为偶数且 `(KB/2) % (WGsPerTile*SUBS) == 0`。

S 路 split-K 的整除性是 `KB % (SUBS*WGsPerTile*S) == 0`（§15.1）：`KB` 是 8 的倍数时
`SUBS=2/WGsPerTile≤2` 对 S∈{2,4,8} 恒成立；`WGsPerTile=4`（WG=4/TILES=1）时 S=8 会挂
（qwen36 `KB=32`：`32/(2*4*8)` 不整除），实测扫描里这一档如实记 skipped。

| 形状 | K | KB | KB%8 | KB/2 %8 | 结论 |
|---|---|---|---|---|---|
| kimi_k3 | 12288 | 96 | 0 | 0 | SUBS ∈ {1,2,4,8,...} 全可用 |
| dpsk_v4_pro / glm52 | 16384 | 128 | 0 | 0 | 同上 |
| minimax_m3 | 8192 | 64 | 0 | 0 | 同上 |
| qwen36 | 4096 | 32 | 0 | 0 | 同上（split-K 后每半 16 chunk，STG≤3 没问题） |
| CONTRACT smoke 形状 | 7168 | 56 | 0 | 0 | 同上 |

五个真实形状的 KB 都是 8 的倍数，所以 `SUBS=2 + WGsPerTile≤4` 恒合法。
若将来遇到 `KB % 8 != 0` 的形状（例如 K=4608 → KB=36），baseline 拓扑只能 `SUBS=1`
（36 % 4 = 0），tuned 拓扑可 `SUBS=2`（36 % 2 = 0）——`validate()` 会返回具体原因字符串，
harness 应把它记进 CSV 的 `config` 列而不是静默跳过。

---

## 7. smem 预算推导

分区顺序（`:559-582`，全部 16 B 对齐，权重区起点 1024 B 对齐——这是 SWIZZLE_128B
的地址相位要求）：

```
[0]                        weight   : kStages * SUBS * BM * 128 + 4096      (4096 = v2 的错相 padding)
[+w]                       act      : PRELOAD_ACT ? act_rows*K : kStages * SUBS * act_rows * 128
[+a]                       scales   : SCALE_IN_SMEM ? align16(TILES*M_TILE*2*(K/128)*4) : 0
[+s]                       barriers : align16(2*kStages*8 + (PRELOAD_ACT ? 8 : 0))
[+b]                       partials : WGsPerOutputTile>1 ? align16(WG*64*M_TILE*4) : 0
[+p]                       splitk   : SPLIT_K_CTA ? align16(TILES*4) : 0
```

其中 `act_rows = min(M_TILE, M)`（**runtime M 决定激活区大小**，M=1 时激活区只有
`kStages*SUBS*128` B，比按 M_TILE=8 分配省 8×；OOB 行由 TMA 零填充，不占 DRAM 流量）。
`kStages = NUM_MATH_WG * STAGES_PER_WG`。

三个主导项的量级（M_TILE=8, BM=48, WG=2, STG=3, SUBS=2, K=16384）：
weight 77824 + act 1536(M=1)/12288(M=8) + scales 8192 + partials 4096 = 91744 / 102496 B。
M_TILE=128 时反过来是激活主导：act = `kStages*SUBS*128*128` = 65536 B（SUBS=1, STG=2, WG=2），
weight 只有 28672 B —— 这也是"M_TILE=128 只能 SUBS=1"的第二个原因（smem）。

`KernelConfig::kFixedSmemBytes` 是与 N/K/M 无关的部分，编译期 `static_assert ≤ 227 KB`；
runtime 部分由 `dynamic_smem_bytes(M,K)` 算，`validate()` 再挡一次（`:260-275`、`:293-294`）。

---

## 8. 寄存器预算推导（含一个容易踩的硬上限）

**累加器需求**：`raw[SUBS][M_TILES][4]` + `partial[M_TILES][4]` = `(SUBS+1) * (M_TILE/8) * 4` 个寄存器。

| M_TILE | SUBS=1 | SUBS=2 |
|---|---|---|
| 8 | 8 | 12 |
| 16 | 16 | 24 |
| 32 | 32 | 48 |
| 64 | 64 | 96 |
| 128 | **128** | 192 ✗ |

**CTA 预算**：`(NUM_MATH_WG * MATH_REGS + PROD_REGS) * 128 ≤ 65536`（`:172-173` static_assert）。

**但还有一层更硬的上限，来自 `__launch_bounds__(kNumThreads, 1)`**：ptxas 给整个 CTA 分配的
寄存器数不会超过 `65536 / kNumThreads`（再向下取整到 warp 分配粒度）。实测（`_dev/regcheck2`）：

| NUM_MATH_WG | threads | `MATH_REGS` 请求 | ptxas 实际 `Used` |
|---|---|---|---|
| 1 | 256 | 232 | **232** ✓ |
| 2 | 384 | 168 / 200 / 232 / 240 | **168**（全部被压到 168） |
| 4 | 640 | 112 | **96** |

含义有两点，都写进文章：

1. **v2/CONTRACT 里 "WG=4 + REG=112" 实际跑的是 96 寄存器**。`warpgroup_reg_alloc<112>`
   只是向硬件多要了用不上的额度（4×128×112 + 128×32 = 61440 ≤ 65536，合法），
   ptxas 编译出来的代码只用 96。所以 harness 的 `--tune` 在 WG=4 下扫 `MATH_REGS ∈ {80,112,168}`
   会得到"112 与 168 完全相同"的结果——那不是测量噪声，是硬上限。
2. **M_TILE=128 撞上限**：需要 128 个累加器 + 描述符/地址/谓词（约 40 个），
   168 不够，实测 `24 B stack frame, 20 B spill stores, 52 B spill loads`。
   两条出路：① 保持 CONTRACT 的 canonical（`WG=2, REG=168`，接受 5 个寄存器的 spill，
   smoke 已通过、rel_l2 = 6.928e-4）；② 换 `WG=1, REG=232`（上限 256）→ **0 spill**，
   代价是 math 并行度减半、CTA 内只有 1 个 tile（smoke 里 `M128 S1 raw wg1` 用的就是它，
   0 spill、rel_l2 = 6.933e-4）。要跑分建议两个都测。

实测 ptxas 汇总（`_dev/regcheck.log`、`_dev/regcheck2`）：

| config | Used regs | spill st/ld |
|---|---|---|
| M_TILE=8, BM=64, WG=4, REG=112, SUBS=1/2（S0/S2） | 96 | 0 / 0 |
| M_TILE=8, BM=48/56/32, WG=2, REG=80, STG=3, SUBS=2（S4/S5/S6） | 80 | 0 / 0 |
| M_TILE=16, BM=64, WG=2, REG=112, TILES=2, SUBS=2 | 112 | 0 / 0 |
| M_TILE=32, BM=64, WG=2, REG=168, TILES=2, SUBS=2 | 168 | 0 / 0 |
| M_TILE=64, BM=48, WG=2, REG=168, TILES=2, SUBS=2 | 168 | 0 / 0 |
| M_TILE=128, BM=48, WG=2, REG=168, TILES=2, SUBS=1 | 168 | **20 / 52**（24 B stack） |
| M_TILE=128, BM=64, WG=1, REG=232, TILES=1, SUBS=1 | 232 | 0 / 0 |

---

## 9. TMA 布局：raw 与 packed 怎么做到"同一条指令"

两种布局都用 **rank-4 map + `box {128, BM, SUBS, 1}`**，smem 落点都是
`[SUBS][BM][128]`（每 128 B 一行，SWIZZLE_128B 按**绝对地址**做 `addr[4:6] ^= addr[7:9]`）：

| | globalDim | globalStride (B) | 坐标 |
|---|---|---|---|
| `PREPACK=true` | `{128, BM, KB, num_tiles}` | `{128, BM*128, KB*BM*128}` | `(0, 0, chunk*SUBS, tile)` |
| `PREPACK=false` | `{128, N, KB, 1}` | `{K, 128, N*K}` ← **非单调** | `(0, tile*BM, chunk*SUBS, 0)` |

raw 这一行的关键是**把 K 拆成 `{128 字节, KB 块}` 两维**，于是 dim2（stride 128）比
dim1（stride K）"更内层"，非单调 stride 是合法的（`cuTensorMapEncodeTiled` 接受，
逐字节验证通过，见 `_dev/probe_tma.cu` / `_dev/probe_act.cu`）。

**与 CONTRACT §4.4 字面写法的偏差（已实测确认必须偏离）**：契约写的是
"box `{128*SUBS, BM, 1, 1}` 于 raw `[N,K]` 上"。这个 box **编码不出来**：
SWIZZLE_128B 下内层 box 宽度不能超过 128 B 的 swizzle span，
`cuTensorMapEncodeTiled` 直接返回 `CUDA_ERROR_INVALID_VALUE`（`_dev/probe_tma.cu` 的 Encoding B）。
而且即使能编码，`[BM][SUBS*128]` 的行内 stride 是 256 B，K-major SW128 的 wgmma 描述符
也表达不了（描述符只能给 8 行组间的 SBO，组内行距固定 128 B）。
本实现用上面的 4D 拆维方案达到契约的**意图**（一次 TMA 拉 `SUBS*BM*128` 连续字节、
K 连续维内取 `SUBS*128`），并且 smem 落点与 packed 完全一致，math 侧代码零分支。

激活侧同理用 rank-3 map `{128, M, KB}` + stride `{K, 128}` + `box {128, act_rows, SUBS}`：
一条 TMA 覆盖 SUBS 个 k_block（v2 在 `LogicalM>1` 时要发 SUBS 条 2D TMA），
且 `act_rows = min(M_TILE, M)` 是 runtime 的 —— M=1 时每 subtile 只搬 128 B，
和 v2 的 M=1 特化路径字节数一致。OOB 行由 TMA 零填充并**计入事务字节**
（`_dev/probe_tma.cu` Encoding C 验证），所以 `arrive_and_expect_tx` 用 box 尺寸算。

---

## 10. 数值路径与精度

```
combined(kb, n_group, m) = weight_scales[n_group][kb] * activation_scales[m][kb]     // fp32 乘，确定性
raw(kb)                  = Σ_{k32} wgmma(weight_fp8, act_fp8)                        // fp32 累加，k32=0 时 scale_d=0
partial                 += fmaf(combined, raw(kb), partial)                          // 每 k_block 一次
out[m][n]                = __float2bfloat16_rn(partial)                              // 只有最后一步降精度
```

scale **不进 wgmma**（fp8 wgmma 没有 per-block scale），而是在 epilogue 前按 k_block 折进
fp32 累加器。这与 `SMOKE_RESULTS.md` 的 host 参考完全同构，实测
`rel_l2 = 6.5e-4 ~ 7.1e-4`（阈值 2e-3），残差来源是 wgmma 内部 k32 累加顺序
与 host 的顺序求和不同 + bf16 末位舍入。`SCALE_IN_SMEM` 两条路径给出**逐位相同**的结果
（同一个 fp32 乘法，只是算的时机不同），smoke 里两者 rel_l2 都是 6.912e-4。
padding 行（`n ≥ N`）与 `m ≥ M` 的 lane 全程被谓词掉；所有 scale 下标都 clamp 到合法范围
（`:695-703`），所以 pad tile 不会越界读，即使它的累加结果是垃圾。

---

## 11. Harness 怎么用（API）

```cpp
#include "decode_gemm.cuh"
#include "weight_prepack.cuh"
using namespace decode_gemm;

using Cfg = KernelConfig</*M_TILE=*/8, /*BM=*/48, /*WG=*/2, /*REG=*/80, /*STG=*/3,
                         /*TILES=*/1, /*SUBS=*/2, /*PREPACK=*/true, /*PDL=*/true,
                         /*EPI_OVERLAP=*/true, /*SPLIT_K_CTA=*/false>;

Problem p;                       // M/N/K + 5 个指针（+ splitk_ws/sem，仅 S6）
p.M = M; p.N = N; p.K = K;
p.activation = d_act; p.weight = d_w; p.weight_scales = d_ws;
p.activation_scales = d_as; p.output = d_out;

// 1) 形状合法性 + grid/block/smem（CSV 的 config 列直接用这个）
LaunchPlan plan = make_launch_plan<Cfg>(p);
if (plan.error) { /* 记录 plan.error，跳过 */ }

// 2) 预处理（timed region 之外）：PREPACK=true 时需要
size_t bytes = prepack::packed_weight_bytes(N, K, Cfg::kOutputRows);
prepack::launch_prepack_weight<Cfg::kOutputRows, Cfg::kOutputTilesPerCta>(d_raw, d_packed, N, K, stream);

// 3) tensor map 建一次、复用（cuTensorMapEncodeTiled 是 host driver 调用，别放进计时循环）
CUtensorMap w_tma = make_weight_tma<Cfg>(p);
CUtensorMap a_tma = make_activation_tma<Cfg>(p);

// 4) 计时循环里只调这个（内部会按需刷新 MaxDynamicSharedMemorySize，并挂 PDL attribute）
launch_with_maps<Cfg>(p, w_tma, a_tma, stream);
```

* `launch<Cfg>(p, stream)` = 建 map + `launch_with_maps`，方便单发正确性检查。
* 冷权重旋转：每个权重集各建一个 `w_tma`（map 里烧进了 device 指针）。
* `SPLIT_K_CTA`：`splitk_ws` 需 `2*M*N` float，`splitk_sem` 按 CONTRACT 给 `M*N` int32
  （kernel 只用前 `ceil(N/BM)` 个，见 §4-S6），**首次使用前清零一次**，之后 kernel 自复位。
* `PDL` 协议的计时：`launch_with_maps` 已经挂好 attribute，harness 只需 back-to-back 连发。
  注意开了 PDL 之后相邻 launch 会真重叠（§4-S1 的风险段）。

---

## 12. 那个 bug：根因与修法（文章可直接用的一段）

**现象**：37 个 smoke case 里 26 个 fail，全部集中在 `PREPACK=true`；raw 布局全绿。
日志上看着像"第一次 launch 通过、第二次 rel_l2≈1.2"（因为我的表格打印的是第一次的 rel_l2，
而 `SECOND-LAUNCH-FAIL` 的标注盖过了它），一度怀疑 PDL 让 prepack 与 GEMM 跨了
programmatic serialization、或 barrier 相位在第二次 launch 没复位。

**真正的根因**：`weight_prepack.cuh` 的 prepack kernel 把
"一行权重有多少个 uint4"算成了 `k / kUint4PerKBlock`（= K/8）而不是 `k / kBytesPerUint4`（= K/16）
——两个常量一个是"每 k_block 8 个 uint4"、一个是"每 uint4 16 字节"，名字都带 16/8，抄的时候串了。
于是 kernel 内部认为的 `total_vectors` 是 launcher 实际启动的线程数的 2 倍（后半部分权重从没被搬），
而 `row_global = linear / vectors_per_row` 又让前半部分**写到了错误的 (tile, k_block, row) 槽位**：
packed buffer 里 (tile0, kb0, row1) 的位置躺着 row1 的 kb1 数据。raw 路径不经过这个 kernel，所以全绿；
`_dev/debug_subs.cu` 用小形状 + 全 1 权重也没抓出来（布局置换在均匀数据下不可见）。

**定位手段**（比看代码快得多，两步就锁死）：① `_dev/debug_pack.cu` 用
"权重全 1、激活按 k_block 取 1/2/4/8"这种可手算的 pattern 跑 raw vs packed —— raw 0/6144 错、
packed 4396/6144 错，说明 kernel 主体没问题、问题在 packed 数据；② 同一个程序把 packed buffer
拷回 host，按 `packed[((tile*KB+kb)*BM+row)*128 + k%128]` 逐字节比对 raw，
`40820992/44040192` 字节不匹配，且第一条 mismatch 就是"row1 位置放着 kb1 的数据"——
置换形状直接指出了 `vectors_per_row` 差了 2 倍。

**修法**：一行，`vectors_per_row = k / kBytesPerUint4`，并在注释里写清两个常量的含义；
`launcher` 与 `kernel` 现在共用同一个 `total_vectors` 定义。修完 37/37 全绿，
packed 布局逐字节 0/44040192 错。

**教训**（文章里值得单独一段）：*布局类 bug 不会让 kernel 崩，只会让结果变成"另一个同样合法的
GEMM"*。用均匀数据做的 smoke test 对置换完全不敏感；能一眼定位的是"把中间 buffer 拷回 host
按公式逐字节比对"，而不是加 printf。

---

## 13. 遗留风险 / 未来优化

1. **M_TILE=128 有 20 B spill**（§8）。要么接受，要么切 `WG=1/REG=232`。
   更彻底的做法是把 `mt` 循环拆成两半（每次 8 组 n8 wgmma + fold），
   把 `raw` 的活跃寄存器砍半，代价是每 k_block 多一次 `warpgroup_wait`。未实现。
2. **激活 smem 在 M_TILE 大时是主导项，且在 WG 之间重复**。`TILES=WG` 时所有 WG 走同一串
   chunk，激活内容完全相同，理论上可以一份 smem 共享（省 1/(2×) 的激活 smem 与 TMA 流量）。
   但 stage 轮转公式 `stage = (group % STG)*WG + math_wg` 与"每 WG 独立 barrier"绑死了，
   共享需要重构成"激活 stage 与权重 stage 分离 + 多消费者 barrier"。未做，是下一步最大的空间。
3. **PDL 与正确性检查的冲突**（§4-S1）：连发两次同一 kernel 写同一 output 在 PDL 下是 race。
   `smoke_test.cu` 用它当"状态复位"探针（两次结果都必须对，因为写的是同一个值）；
   harness 若要严格验证 PDL 协议，应让相邻 launch 写不同 buffer。
4. **`splitk_sem` 的粒度**：我用"每输出 tile 一个 semaphore"而不是契约字面的"每元素一个"，
   M=128 时能省 78 万次 atomic。~~若将来 `gridDim.y > 2`（多路 split-K），`previous == 1`
   要改成 `previous == gridDim.y - 1`，并且"后到者"需要累加 y 份 ws~~ —— **已做，见 §15**。
   同时 §15.5 记录了泛化过程中挖出来的一个真 bug：PDL 连发时 semaphore 会被跨 launch 污染。
5. **`SINGLE_PRODUCER_WARP` 与 `PRELOAD_ACT`** 只是按 v2 语义实现并验证了正确性
   （smoke 各一条），没有性能数据；v2 的结论是两者都不划算。
6. **N 不是 8 的倍数**时（本次五个形状都不是这种情况）pad tile 的谓词已覆盖，
   但 `prepack` 会把 buffer 撑到 `padded_n*K`，harness 分配时要按 `padded_weight_bytes()` 来。
7. **未做**：cluster（`gridDim.y` 之外的 2-CTA TMA multicast）、`cp.async.bulk` 非 tensor 路径、
   以及 M_TILE≥16 时的 wgmma n 维合并（`m64n8k32` 换 `m64n16k32`/`m64n32k32` 可以把
   `M_TILES` 组 B 描述符合并，减少指令数）。这些都会改变数值路径或 smem 布局，属于下一版。

---

## 14. 复现

```bash
P=/home/admin/workspace/hengfeng_data/weave_new/DSA-learn/mk/h20/cutedsl/decode_gemm
cd $P/kernels
./build_smoke.sh          # nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -DNDEBUG
                          #   -I$P/kernels -I$CUTLASS smoke_test.cu -lcuda -lcublasLt
./run_smoke.sh            # 经 tools/gpurun_dg.sh（池 {0,1} + 跨项目 flock + 锁频 1830）
cat SMOKE_RESULTS.md      # 37 case 全绿
```

`run_smoke.sh` 里有一处历史 workaround：`tools/gpurun_dg.sh` **v1** 第 33 行把 `${m}MiB` 写成了
`$mMiB`，在 `set -u` 下会直接 abort，所以 wrapper 里 `export mMiB=MiB` 让那行能展开
（该文件不在本 agent 的 write scope，已报给 orchestrator；v2 已修好这行，export 变成无害的 no-op，
保留是为了兼容 v1）。GPU 选择完全交给 gpurun_dg.sh：CONTRACT §0 现行版本是
BORROW 池 {2,3} 优先（keeper 预留但物理空闲，带 2 s 看门狗，外来 compute 落卡即 `exit 75` 让路）
+ CLEAN 池 {0,1} 兜底，跨项目 flock 与锁频 1830 都在 wrapper 里，**本目录的脚本从不自己设
`CUDA_VISIBLE_DEVICES`、也从不 flock `.gpu3.lock`**。

S 路 split-K 的甜区扫描（§15）：
```bash
cd $P/bench
make BIN=bench_decode_sk OBJDIR=build_sk TUNE=0 -j12      # 只烘 canonical，~90 s
bash run_splitk_factor.sh                                 # qwen36 全 M + kimi_k3 M=1 对照
python3 splitk_factor_table.py \
        ../results/splitk_factor_qwen.csv --md             # 透视成表
# 想重现 semaphore 污染那个 bug（§15.5）：加 --sem-rotate 1
```

预算表与探针：`_dev/config_table`（打印 §6 的 smem/CTA 表）、`_dev/regcheck*`（ptxas 寄存器/spill）、
`_dev/probe_tma`+`_dev/probe_act`（TMA 编码合法性与 smem 落点逐字节验证）、
`_dev/debug_subs`+`_dev/debug_pack`（§12 的定位工具）。详见 `_dev/README.md`。

---

## 15. S 路 split-K（`SPLITK_FACTOR`）：三笔账 + 实测甜区

> splitk-S agent，2026-09-13。把 S6 的 CTA 级 split-K 从写死的 2 路泛化成 S∈{2,4,8} 路，
> 然后实测 qwen36（`N=2048, K=4096`，全项目最小的 N）的 S 甜区 + kimi_k3 的负对照。
> 数据：`results/splitk_factor_qwen.csv`（PDL, reps=3）、
> `results/splitk_factor_qwen_3proto.csv`（ISO/B2B/PDL 全量）、
> `results/splitk_factor_kimi_k3.csv` / `_3proto.csv`。门禁：`SMOKE_LOG_skS.txt`（50 case 全绿）。

### 15.1 接口与实现（`decode_gemm.cuh`）

`KernelConfig` 追加**第 16 个**模板参数 `int SPLITK_FACTOR = 0`。追加在最后并带默认值，
所以 CONTRACT §4 冻结的前 15 个参数位置一字不动，旧的实例化点（smoke_test 的 `Cfg<...>`、
bench 的 `VariantImpl<...>`、probe）不需要改就能编。

| `SPLIT_K_CTA` | `SPLITK_FACTOR` | `kSplitKFactor` (S) | `kSplitKCta` |
|---|---|---|---|
| false | 0（默认） | 1 | false |
| true | 0（默认） | **2**（= 旧行为） | true |
| — | 1 | 1（强制关） | false（与 `SPLIT_K_CTA=true` 同时给会 static_assert） |
| — | 2/4/8 | 2/4/8 | true |

改动点（S=1 时全部编译期折叠，**零开销**：`chunks_per_split`/`chunk_base` 是常量折叠，
epilogue 走 `if constexpr (!kSplitKCta)` 的老路，smem/寄存器预算一字未变）：

1. `gridDim.y = kSplitKFactor`（`make_launch_plan`，`:1091`）；
2. 每 CTA 处理 `[y*chunks/S, (y+1)*chunks/S)` 这段**连续** chunk（`:597-599`，注释 `:592-596`）。
   generation/stage 轮转公式没动 —— 它本来就只看 `local_chunk`，所以每个 split 各自从
   generation 0 重启，producer 的 `empty->wait((gen+1)&1)` 与 math 的 `full->wait(gen&1)`
   自然对上；
3. `splitk_ws` 布局 `[S][M][N]` fp32（harness 侧按 registry 里最大的 S 分配）；
4. ticket：每输出 tile 一个 leader 线程 `atomicAdd(sem+tile, 1)`，
   **`previous == S-1` 者为 reducer**（`:1049-1053`），累加其余 S-1 份 ws 后 `__float2bfloat16_rn`
   写 output（`:1058-1073`，peer 顺序按 `(y+s) % S` 轮转，避免所有 reducer 先撞 slab 0）；
5. `atomicExch(sem+tile, 0)` 自复位不变（`:1074`）——但它在 PDL 下有个坑，见 §15.5；
6. 整除性：`K % (128*SUBS*WGsPerTile*S) == 0`。K 是 runtime 值，所以由 `validate()`
   逐次把关（`:327-341`），另外给了一个编译期镜像 `Cfg::k_ok_for_splitk<K>()`
   （`:306-315`，另有 `num_ctas(n)`/`chunks_per_split(k)` 两个 host 侧助手 `:297-303`），smoke_test 用它把 6 个 S 路 config 的 K 钉在 `static_assert` 里。

**放宽了一条旧闸门**：原来要求 `chunks/S >= STAGES_PER_WG`（"填满流水线"），现在只要求
`chunks/S >= WGsPerOutputTile`（每 WG 至少一组）。理由：那是**性能**建议而不是正确性条件
（没用到的 stage 其 barrier 永远不会被 wait，不会死锁），而它恰好挡掉了本次最该测的角落
—— qwen36 `KB=32/SUBS=2` 时 S=8 每 CTA 只剩 2 个 chunk（< STG=3）。挡住它就等于
"用假设排除掉待验证的假设"。放开的代价如实体现在数据里（§15.2 第二笔账）。

**顺手修的一处 S 相关低效**：`SCALE_IN_SMEM` 的 prologue 原来把**整段 K** 的
`w_scale*a_scale` 都合并进 smem，而每个 CTA 只用得到自己那 `1/S` 段 —— 于是 prologue
成本不随 S 缩小、CTA 数又 ∝ S，全网格的 prologue 总量 ∝ S（S=8/N=2048 时 512 个 CTA
各合并 32 个 k_block，31/32 是白干的）。现在只填 `[kb_lo, kb_lo + chunks_per_split*SUBS)`
（`:767-805`），smem 表的**布局与大小不变**（仍按绝对 k_block 摆，只是稀疏填充），
S=1 时绝对下标恒等于原来的枚举下标 `i`，生成代码逐字相同（用 `Cfg::kSplitKCta ? … : i`
把这条等价性交给编译器折叠）。

### 15.2 三笔账

**账 1：reduce 流量 ∝ S·M（而且是延迟账，不是带宽账）**

ws 往返 = 写 S 份 + 读回 S-1 份 = `(2S-1)·M·N·4 B`：

| 场景 | 权重流量 | S=2 | S=4 | S=8 |
|---|---|---|---|---|
| qwen36 M=1 (N=2048) | 8.39 MB | 0.025 MB (0.3%) | 0.057 MB (0.7%) | 0.123 MB (1.5%) |
| qwen36 M=32 | 8.39 MB | 0.79 MB (9%) | 1.84 MB (22%) | 3.93 MB (**47%**) |
| kimi_k3 M=1 (N=7168) | 88.08 MB | 0.20 MB (0.2%) | 0.49 MB (0.6%) | 1.06 MB (1.2%) |

M=1 时流量占比小到可以忽略 —— 所以**M=1 上 split-K 变慢不是带宽问题**。真正贵的是
epilogue 那条**串行依赖链**：`ws store → __threadfence(membar.gl) → named barrier →
atomicAdd(L2 往返) → named barrier → smem flag → __threadfence → 读 S-1 份 ws(gmem 往返)
→ bf16 store`。实测它的长度：qwen36 M=1 在现役最优几何上 S=1→S=2 是 3.698→5.087 µs，
即 **+1.39 µs**；而整个 kernel 才 3.7 µs，所以 +38%。S 再大，链尾的"读 S-1 份"还线性变长
（S=4 是 +4.04 µs）。

**账 2：每 CTA 的流水线深度 = K/S/128/SUBS 个 chunk**

| 形状 | KB | SUBS | S=1 | S=2 | S=4 | S=8 | STG |
|---|---|---|---|---|---|---|---|
| qwen36 K=4096 | 32 | 2 | 16 | 8 | 4 | **2** | 3 |
| kimi_k3 K=12288 | 96 | 2 | 48 | 24 | 12 | 6 | 3 |

qwen36 在 S=4 时只剩 4 个 chunk（`WGsPerTile=2` → 每 WG 2 组），S=8 只剩 2 个（每 WG **1 组**）：
fill/drain 完全摊不掉，每个 CTA 变成"prologue → 一次 TMA → 一次 wgmma → epilogue"，
于是 CTA 数越多、每 CTA 越像纯开销。kimi_k3 的 S=8 还有 6 个 chunk（= STG×WGsPerTile），
所以它的 S=8 只慢 11.6% 而 qwen36 的 S=8 慢 122%~155%。**K 短的形状扛不住大 S。**

**账 3：wave 量化 = `ceil(CTA/78)` 与 `CTA/78` 的差**

| 几何（qwen36, N=2048） | CTA(S=1) | S=1 | S=2 | S=4 | S=8 |
|---|---|---|---|---|---|
| BM=56/WG=4/T=1（现役最优） | 37 | 0.47 wave | **0.95** | 1.90 | (不整除) |
| BM=32/WG=2/T=1（README §6.2 的 S6 几何） | 64 | 0.82 | 1.64 | 3.28 | 6.56 |
| BM=64/WG=1/T=1（M_TILE=16 现役） | 32 | 0.41 | 0.82 | 1.64 | 3.28 |
| BM=32/WG=2/**T=2**（M_TILE≥16 canonical） | 32 | 0.41 | 0.82 | 1.64 | 3.28 |

只有"S×CTA 落进 1 个 wave 之内"才不付量化税：37→74（0.95）是唯一干净的一档；
64→128（1.64）要跑 2 个 wave，每 wave 只做一半 K，理论上打平、实际上白付一遍
prologue/epilogue。这解释了为什么 BM=32 那条几何在 S=2 上是 **-0.8%（打平）** 而不是收益。

### 15.3 实测甜区（PDL 协议，reps=5，锁频 1830，冷权重 64 组 × 8 MiB 轮转）

括号里的 `%` 是相对**同一几何**的 S=1；负数 = split-K 更快。GB/s 是 CONTRACT §6 的 decode
口径（`(N*K + M*K + 2*M*N)/t`）。两条几何并列给出，因为 S 的甜区强依赖 S=1 时的 CTA 数：
base0 = 该形状该 M 的现役 canonical（tune override 赢家），base1 = 形状无关 canonical
（§6.2 的 S6 几何，BM 按整除 N / CTA 甜区选）。**粗体**是该行的 S=1 基线，
不一定是该 M 的最优（M=16/32 的 `BM32/WG2/T2` 行里 S=2 才是最快的）。

qwen36（`N=2048, K=4096`，数据源 `results/splitk_factor_qwen.csv`）：

| M | 几何 | CTA(S=1) | S=1（µs / GB/s） | S=2 | S=4 | S=8 |
|---|---|---|---|---|---|---|
| 1 | BM56/WG4/T1（**现役最优**） | 37 | **3.698 / 2271** | 5.087 / 1651 (+38%) | 7.739 / 1085 (+109%) | 不整除¹ |
| 1 | BM32/WG2/T1 | 64 | **4.486 / 1872** | 5.114 / 1642 (+14%) | 6.962 / 1206 (+55%) | 11.435 / 734 (+155%) |
| 2 | BM56/WG4/T1（**现役最优**） | 37 | **3.824 / 2198** | 5.903 / 1424 (+54%) | 8.964 / 938 (+134%) | 不整除¹ |
| 2 | BM32/WG2/T1 | 64 | **4.856 / 1731** | 5.873 / 1431 (+21%) | 8.086 / 1039 (+66%) | 12.755 / 659 (+163%) |
| 4 | BM56/WG4/T1（**现役最优**） | 37 | **3.869 / 2176** | 6.085 / 1384 (+57%) | 9.118 / 924 (+136%) | 不整除¹ |
| 4 | BM32/WG2/T1 | 64 | **4.881 / 1725** | 6.020 / 1399 (+23%) | 8.280 / 1017 (+70%) | 13.045 / 646 (+167%) |
| 8 | BM56/WG4/T1（**现役最优**） | 37 | **3.872 / 2183** | 6.264 / 1350 (+62%) | 9.322 / 907 (+141%) | 不整除¹ |
| 8 | BM32/WG2/T1 | 64 | **4.992 / 1694** | 6.153 / 1374 (+23%) | 8.364 / 1011 (+68%) | 13.155 / 643 (+164%) |
| 16 | BM64/WG1/T1（**现役最优**） | 32 | **6.730 / 1266** | 8.309 / 1025 (+23%) | 9.886 / 862 (+47%) | 13.573 / 628 (+102%) |
| 16 | BM32/WG2/**T2** | 32 | **15.468 / 551** | 10.909 / 781 (-29%) | 12.195 / 699 (-21%) | 15.548 / 548 (+1%) |
| 32 | BM56/WG1/T1（**现役最优**） | 37 | **8.821 / 981** | 8.611 / 1005 (-2%) | 10.106 / 856 (+15%) | 12.778 / 677 (+45%) |
| 32 | BM32/WG2/**T2** | 32 | **20.881 / 414** | 15.166 / 570 (-27%) | 16.890 / 512 (-19%) | 20.519 / 422 (-2%) |

¹ `BM56/WG4/T1` 的 `WGsPerTile=4`，S=8 时 `(K/8)/(128*SUBS)=2` 不是 4 的倍数 →
`K % (128*SUBS*WGsPerTile*S) == 0` 不成立，`validate()` 如实拒绝（扫描里记 skipped，不算失败）。
² kimi_k3 `N=7168 > 4096` 且 `splitk_worth_it(7168,56)==false`，按 §15.6 的闸门**没有实例化**
S≥4 的 override 变体（省编译时间）；同形状的 S=4/S=8 在 base1（`BM56/WG2/T1`）上有数据。

负对照 kimi_k3（`N=7168, K=12288`，M=1，数据源 `results/splitk_factor_kimi_k3.csv`）：

| 几何 | CTA(S=1) | S=1（µs / GB/s） | S=2 | S=4 | S=8 |
|---|---|---|---|---|---|
| BM56/WG1/T1（**现役最优**） | 128 | **26.164 / 3368** | 26.778 / 3290 (+2%) | 未展开² | 未展开² |
| BM56/WG2/T1 | 128 | **26.934 / 3271** | 26.410 / 3336 (-2%) | 27.486 / 3206 (+2%) | 30.865 / 2855 (+15%) |

**甜区结论**：
* qwen36 的 S 甜区是 **S=1**（M=1..32 全部）。跨几何跨 S 取最优，每个 M 的赢家都是 S=1；
  唯一例外是 M=32 的 8.611 vs 8.821（-2.4%，落在 ±3% 的复现噪声里，算打平）。
* S=2 只有在「S=1 的 CTA 数 ≤ 0.5 wave **且**该 config 本身已经被 CTA 数卡死」时才是净收益：
  `BM32/WG2/T2`（32 CTA）M=16/32 拿到 -29%/-27%。但那是把一个本来就慢 2.3× 的 config 拉回来
  一点（15.47→10.91 µs），仍远慢于同 M 的 S=1 最优几何（6.73 µs）——**"split-K 救活了一个
  坏 config" 不等于 "split-K 有用"**。
* **S=4/S=8 在 qwen36 上没有任何一档是净收益**（`BM32/WG2/T2` 的 -21%/-19% 同上；且 S=8
  已经回到 +1%/-2%，即"把 K 切成 8 段"省下的时间刚好被 CTA 数 ×8 的固定开销吃光）。
* kimi_k3（128 CTA、K=12288）：S=2 在 ±2% 内打平，S=4 +2%，S=8 +15% → 负对照成立
  （预测"≤0 收益" ✓）。K 长 3 倍、CTA 数已经 1.64 wave，所以它的 S=8 只慢 15%，
  而 qwen36 的 S=8 慢 155%~164% —— 这就是账 2（流水线深度）的直接读数。

**复现性**（同一二进制、同一张卡、间隔几分钟的 4 次独立运行）：44 个点里 43 个的 PDL p50
相互差 ≤1.2%；唯一的双稳点是 `M=1 / BM32/WG2/T1 / S=1`，在 **4.48 µs 与 5.16 µs 两个模态**
之间跳（reps=3 的两次都命中慢模，reps=5 与 ISO/B2B/PDL 全量那次都命中快模；p30 跟随 p50，
说明是整批 90 次 launch 一起换模态，不是个别离群值）。所以主表用 **reps=5** 那次；
reps=3 的两次原始文件都留着（任务口径 reps=3，两次都交）：
`results/splitk_factor_qwen_reps3run1.csv`、`results/splitk_factor_qwen_r3b.csv`；
ISO/B2B/PDL 全量在 `results/splitk_factor_qwen_3proto.csv`（§15.4 的协议梯度用它）。
这个双稳只影响 `BM32/WG2/T1` 的 S=1 基线（即"S=2 是 +14% 还是打平"），S≥4 的结论在两个模态下都不变。

### 15.4 与预测（"S=4 甜区、M=1 +15~30%"）的偏差解释

预测错在两处，都有数据支撑：

1. **把 PDL 的并发当成免费的**。PDL 协议下相邻两个 grid 是真的重叠（同一个 config
   B2B 7.24 µs → PDL 3.78 µs，≈2 个 grid 在飞），SM 早就被喂满了；split-K 想提供的
   "更多并发 DRAM 流"是**重复供给**，而它的代价（账 1 的串行链尾）照付。
   证据：同一个 config 的相对损失随协议并发度单调递增 ——
   qwen36 M=1 / BM56/WG4 / S=2：ISO **+6.3%** → B2B **+19.9%** → PDL **+34.5%**
   （三个数取自同一次 `splitk_factor_qwen_3proto.csv` 运行；reps=5 那次 PDL 是 +38%，同向）；
   S=4：+30.0% → +59.0% → +104.1%。也就是说 split-K 是 launch 重叠的**替代品**，不是补充品：
   已经在用 PDL 的流水里，它没有位置。
2. **把 reduce 当带宽账**。M=1 时 reduce 流量只占权重流量的 0.3%（账 1 表），
   按带宽算根本不该有 +37%。真正贵的是那串 `membar.gl + atomicAdd + gmem 往返`
   的**串行延迟 ~1.4 µs**，而 qwen36 整个 kernel 才 3.7 µs —— 分母太小。
   这也解释了为什么 K 大 12 倍的 kimi_k3（26 µs）上同样的链只值 +2.5%。

至于"S=4 是甜区"：只有当 S=1 的 CTA 数在 78/4 ≈ 20 附近、且 K 长到 S=4 后每 CTA 还剩
≥ STG×WGsPerTile 个 chunk 时才可能成立。本项目五个形状里没有一个落在这个窗口
（qwen36 是 CTA 数够但 K 太短，kimi/glm/dpsk/minimax 是 CTA 数本来就 ≥ 78）。
`configs.cuh` 的 `splitk_worth_it()` v3 判据（`ctas < 78 && ctas*2 >= 78`）
把 qwen36 现役几何（37 CTA → 74）判成 **false**，与本次实测一致，不需要改；
本次数据只是给它补上了 S≥4 的封杀理由（账 2 + 账 3）。

### 15.5 挖出来的真 bug：PDL 连发把 semaphore 弄脏（已修）

**症状**：S 扫描第一次跑出来，`BM32/WG2/T1` 几何的 S=2/4/8 三档 rel_l2 全是
**1.71e-01**（阈值 2e-3），而同一批里排在它们**前面**的 `BM56/WG4` 几何全绿；
单独只跑 S=8 时它也全绿（1.65e-03）。→ 不是 kernel 算错，是**状态被上一个 variant 污染**。

**根因**：`atomicExch(sem, 0)` 的自复位与 PDL 的重叠窗口打架。PDL 下 grid i+1 的 CTA
在 grid i 的所有 CTA 都发过 `launch_dependents` 之后就能起跑，而 `launch_dependents` 是
producer 发的（在 epilogue **之前**）；CTA 数越多、每 CTA 的 K 段越短（S 大时正是如此），
grid i+1 的 CTA 就越可能先冲到握手点。此时 grid i 的 reducer 还没 `atomicExch(0)`，
于是 grid i+1 的 `atomicAdd` 被随后的 exch 抹掉一格 → 该 tile 在 grid i+1 里
**选不出 reducer**（既不做 reduce、也不写 output），且 sem 停在非 0 → 之后**所有** launch
的握手全乱。

**证据**（给 bench 加了 sem 回读门禁后直接抓到，`--sem-rotate 1` 重现旧行为）：
```
SEM-DIRTY qwen36 M=1 S=4: 2 个 semaphore 没自复位（sem_rotate=1）   <- 自己那轮 PDL 批次留下的
SEM-DIRTY qwen36 M=1 S=2: 2 个 ...   -> rel_l2=1.58e-01 pass=N
SEM-DIRTY qwen36 M=1 S=4: 6 个 ...   -> rel_l2=1.58e-01 pass=N
SEM-DIRTY qwen36 M=1 S=8: 26 个 ...  -> rel_l2=1.58e-01 pass=N
```
**修法**（harness 侧，kernel 的自复位语义一字不动）：semaphore 跟着冷权重 set 一起**轮转**
——`bench_decode.cu` 把 `d_sem` 开成 `[sem_slots][sem_ints]`，`do_launch()` 按
`set_idx % sem_slots`（默认 16 槽）取槽。相邻 launch 必落不同槽，同一个槽要隔 16 次
launch 才复用，远超 PDL 的重叠窗口（≤3 次），所以每次 launch 看到的 sem 都是干净的 0。
修完 44 个扫描点全绿、`sem_dirty=0`（`--sem-rotate 16`，现在是默认）。

**两条必须写进文章的推论**：
1. **过去 `SPLITK=1` 的 PDL 数字可能偏乐观**：sem 脏了之后 reduce 被整块跳过，
   kernel 少干了活自然快。本次 S 扫描的数据全部是修好之后测的（每个点都回读 sem）。
2. smoke_test 抓不到这个 bug —— 它每次 launch 之间都 `cudaDeviceSynchronize()`，
   没有重叠窗口。能抓到它的是"**同一个进程里连续跑多个 split-K variant 的 PDL 批次**"，
   也就是 bench 本身。所以 bench 侧现在也有 sem 门禁（脏 → 该点计入 failed、退出码非 0）。

### 15.6 harness 侧接口变化（`bench/`）

* `ConfigSpec` 加 `int splitk_factor`（0=未指定）；有效值一律用 `sk_factor_of()` 读，
  它和 kernel 的 `kSplitKFactor` 是同一条归一化规则。`config` 串加 **`SK=<S>`** 字段，
  `SPLITK=` 改成打**有效**开关（`plot.py` 认 `SPLITK=1`，不能变）。
* `VariantImpl<..., int SKF=0>` 追加第 16 个模板参数；`ctas_fn` 改成问 `Cfg::num_ctas(N)`
  （= `gridDim.x * S`），CSV/JSON 里的 `num_ctas` 从此含 S。
* step6 的注册：canonical 每个候选 BM 注册 `{base(S=2), flip(S=1), S=4, S=8}` 四条；
  tune override（形状已知）按闸门 `sk_factor_expand(n,bm) = splitk_worth_it(n,bm) || n<=4096`
  再加 `{4,8}`（kimi/glm/dpsk/minimax 因此不展开，省实例化）；tune 网格的 S 维默认关，
  `make tune-bin TUNE_SKF=1 TUNE_SHAPE_N=2048 TUNE_SHAPE_K=4096 TUNE_GRID=s4 TUNE_STEPS=6`
  才展开（slot0=S2、slot1=S4、slot2=S8，不重复实例化）。
* 新 CLI：`--splitk-factor-sweep`、`--splitk-factors 1,2,4,8`、`--factor-out PATH`、
  `--sem-rotate N`。扫描每个 M 取两条基准几何（base0 = 该形状该 M 的现役 canonical，
  base1 = 形状无关 canonical），保证 S 之间是同几何可比；缺档记 skipped 不算失败。
  输出 = 主 CSV 的 16 列 + `splitk_factor`。
