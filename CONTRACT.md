# CONTRACT — decode_gemm 干净重写（接口冻结）

唯一接口来源。所有 agent 只写自己 write scope 内的文件，接口照本文件，不得擅自改动。
与 reference 源码冲突时以本文件为准。

## 0. 路径与环境

```
W   = /home/admin/workspace/hengfeng_data/weave_new
P   = $W/DSA-learn/mk/h20/cutedsl/decode_gemm          <- 本项目根
REF = $W/DSA-learn/mk/h20/cutedsl/shared               <- 参考实现（只读！）
CUTLASS = $W/weave_v2/deepgemm/third-party/cutlass/include
GPU   = 经 tools/gpurun_dg.sh v2 调度（用户 2026-09-13 裁决：卡用了才用，不许长期占）：
        BORROW 池 {2,3} 优先 —— 被 v19_swapab keeper 预留但物理空闲，授权借用；
          守卫 util<=5% 且除 keeper 白名单外无 compute pid；运行中 2s 看门狗，
          外来 compute 落卡 => abort 作废让路（exit 75）。keeper 不 kill 不碰。
        CLEAN 池 {0,1} 兜底 —— 严格守卫 util<=5%/mem<=2000MiB/完全无 compute pid。
        两池均抢 _lockbench_20260913/locks/gpu<N>.lock 跨项目 flock；跑前锁频 1830。
NVCC  = /usr/local/cuda/bin/nvcc 12.8，-std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a
```

编译命令模板（所有 build 脚本必须一致）：
```
nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -DNDEBUG \
  -I$P/kernels -I$CUTLASS <src> -lcuda -lcublasLt -o <out>
```

参考实现（只读）：
- `$REF/swapab_m1_kernel_v2.cuh`（945 行，终版 kernel，PDL 后 85.3%）
- `$REF/swapab_m1_weight_prepack_v2.cuh`（tile-major prepack）
- `$REF/fp8_swapab_wgmma/common/sm90_compat.cuh`（ld_shared/make_smem_desc/wgmma 包装，直接复制到 $P/kernels/）
- `$REF/bench_v2.cu`（harness 参考：冷权重旋转、三种计时协议、correctness）
- `$REF/ANALYSIS_decode_gemm.md`（设计推导，必读）

## 1. 问题定义（不许改）

decode GEMM：`out[M,N] = act[M,K] @ W[N,K]^T`，FP8 e4m3 输入 + per-128(K) 反量化 scale，bf16 输出。
- `act[M,K]` K-contiguous；`weight[N,K]` K-contiguous（raw 布局）或 tile-major packed（PREPACK 布局，见 §5）
- `weight_scales[ceil(N/128), K/128]` row-major；`activation_scales[M, K/128]` row-major
- `output[M,N]` bf16，N-contiguous
- swap-A/B：weight 走 wgmma A 侧（m64），activation 走 B 侧（n8×M_TILES）
- kKBlock = 128（scale group = 128 K 元素，wgmma k32×4 = 128）

### M 阶梯（compile-time M_TILE，runtime M ≤ M_TILE）
```
M_TILE ∈ {8, 16, 32, 64, 128}
sweep M = 1,2,4,8 -> M_TILE=8;  16->16;  32->32;  64->64;  128->128
```
v2 的 `LogicalM<=4` 是遗留保守断言：B-tile 恒为 n8，`logical_m0=(lane&3)*2, logical_m1=+1`
本来就覆盖 m=0..7。新 kernel 最小 tile 就是 8，M=1..8 共用 M_TILE=8（runtime 谓词过滤）。
M_TILE=16..128 时每 k_block 发 M_TILES=M_TILE/8 组 n8 wgmma，
raw[M_TILES][4] + partial[M_TILES][4]（M_TILE=128 -> 128 accum regs/thread）。

## 2. 模型与形状（models/models.json，由 shapes agent 产出）

从 `$W/DSA-learn/baselines/h100/<model>/full_attention_benchmark.py` 提取真实 decode 层 GEMM。
5 个模型（用户口径 -> 本地目录名）：

| 用户称呼 | 本地模型 | hidden |
|---|---|---|
| kimi k3 | kimi_k3 | 7168 |
| qwen3.8 max | qwen36 | 2048 |
| glm5.3 | glm52 | 6144 |
| dpsk v4.1 flash | deepseek_v4_pro | 7168 |
| minimax m3 | minimax_m3 | 6144 |

每个模型选 1 个代表 decode GEMM（N,K）+ 备选，schema：
```json
{"models":[{"id":"kimi_k3","display":"Kimi K3","hidden":7168,
  "gemm":{"layer":"<层名>","N":6144,"K":7168,"why":"<一行理由>"},
  "alt":[{"layer":"...","N":...,"K":...}],
  "source":{"file":"...py","lines":"L..."}}],
 "peak_bw":{"spec_tbps":4.0,"measured_note":"见 tools/peak_bw_probe"}}
```
要求：N 必须是 128 的倍数（kernel 按 128 scale-group 工作，pad 到 128 的倍数由 harness 做）。

## 3. 技术步骤（cumulative，文章主线，顺序不许改）

> 修订 2026-09-14：阶梯改为 6 步。原 S5「epi_overlap」经触发点消融证明与 S1 的早触发
> 冗余（per-CTA 信号先发先生效），移出主阶梯，仅保留为 `--pdl-placement` 消融；
> 原 S6 split-K 改编号 S5。kernels 侧 static_assert(EPI→PDL) 已放宽以支持 store-only 变体。

| step | 名字 | 相对上一步新增 | 说明 |
|---|---|---|---|
| S0 | baseline | — | swap-A/B 流水线：BM=64, WG=4, STG=2, SUBS=1, raw 布局, 无 PDL |
| S1 | pdl | PDL | launch_dependents 早触发 + host 侧 ProgrammaticStreamSerialization 属性 |
| S2 | blockk | SUBS=2 | 一次 TMA 拉 2 个 k_block（box 128→256），减少 op 数、加大每 op 字节 |
| S3 | prepack | PREPACK | tile-major packed 权重：每 (tile,k_block) 连续，TMA 全连续大块 |
| S4 | tile_stage | BM=48, WG=2, STG=3 | blockM/blockN 调优换更深流水线（128 CTAs 无 pad） |
| S5 | epi_overlap | EPI_OVERLAP | epilogue store 前发第二次 launch_dependents，让下个 kernel 的 prologue 与 store 重叠 |
| S6 | splitk | SPLIT_K_CTA | CTA 级 K-split（gridDim.y），gmem semaphore 两段 reduce；对小 N 模型/大 M 有用 |

M_TILE=8 的 canonical config 如上表（继承 v2 调优史）。
M_TILE≥16：kernel+harness agent 按下述规则自行定（写入 kernels/README.md 的 config 表）：
- TILES_PER_CTA = NUM_MATH_WG（每 WG 独立 tile，无 intra-CTA splitK reduce，避免 partial smem 爆炸）
- M_TILE=128：MATH_REGS=168, WG=2, STG=2, SUBS=1（smem/reg 双约束，实测再调）
- 每步都必须给出该 M_TILE 下可行的 config（smem ≤ 227KB，regs 预算满足），跑不通时降级 STG/SUBS 并在 README 记录

## 4. Kernel API（kernels/decode_gemm.cuh，由 kernel agent 产出）

```cpp
namespace decode_gemm {

template <
    int M_TILE,               // 8/16/32/64/128，runtime M <= M_TILE
    int OUTPUT_ROWS_PER_CTA,  // BM（blockN per CTA，输出行/CTA）
    int NUM_MATH_WG,          // 1/2/4
    int MATH_REGS,
    int STAGES_PER_WG,
    int TILES_PER_CTA,        // 每 CTA 输出 tile 数；WG 数必须整除
    int SUBS_PER_STAGE,       // blockK = SUBS * 128
    bool PREPACK,             // 权重 tile-major packed（true）/ raw N-major（false）
    bool PDL,                 // kernel 内 griddepcontrol 触发（producer 最后一次 TMA 后）
    bool EPI_OVERLAP,         // math 侧 store 前再触发一次 launch_dependents
    bool SPLIT_K_CTA,         // gridDim.y = 2，gmem reduce
    int PROD_REGS = 32,
    bool SINGLE_PRODUCER_WARP = false,
    bool PRELOAD_ACT = false> // 仅 M_TILE=8 且 PREPACK 允许
struct KernelConfig { /* 派生常量全部 constexpr，命名沿用 v2：kStages,kNumKChunks,
                         kWgsPerOutputTile,kGroupsPerMathWg,kDynamicSmemBytes... */ };

struct Problem {              // runtime 形状；N、K 是 host 值
  int M, N, K;
  const __nv_fp8_e4m3* activation;      // [M,K]
  const __nv_fp8_e4m3* weight;          // PREPACK ? packed : [N,K]
  const float* weight_scales;           // [ceil(N/128), K/128]
  const float* activation_scales;       // [M, K/128]
  __nv_bfloat16* output;                // [M,N]
  float* splitk_ws;                     // SPLIT_K_CTA 专用：[2][M,N]
  int* splitk_sem;                      // SPLIT_K_CTA 专用：[M*N] int32，host 保证清零后自复位
};

// host：建 tensor map（PREPACK/raw 两种布局）
template <typename Cfg> CUtensorMap make_weight_tma(const Problem&);
template <typename Cfg> CUtensorMap make_activation_tma(const Problem&);

// device kernel（__grid_constant__ CUtensorMap 传参，同 v2）
template <typename Cfg>
__global__ void fp8_decode_gemm(__nv_bfloat16* out, const float* w_scales,
                                const float* a_scales,
                                const __grid_constant__ CUtensorMap w_tma,
                                const __grid_constant__ CUtensorMap a_tma,
                                GemmRuntime rt);   // rt = {M,N,K,splitk_ws,splitk_sem}

// host launcher：算 grid/block/smem，设 PDL attribute，launch
template <typename Cfg>
cudaError_t launch(const Problem&, cudaStream_t);

}  // namespace decode_gemm
```

硬要求：
1. N、K 是 runtime（tensor map 运行时建）。M 是 runtime（≤M_TILE）。
   grid = ceil(ceil(N/BM)/TILES_PER_CTA) × (SPLIT_K_CTA?2:1)。
2. 所有技术是 compile-time 模板参数（clean code，无 runtime branch 热点）。
3. PDL 语义与 v2 完全一致：producer 发完最后一次 TMA 后 `griddepcontrol.launch_dependents`；
   EPI_OVERLAP=true 时 math 在 epilogue store 前再发一次。PDL=false 时不发射 trigger。
4. PREPACK=false 时权重走 4D tensor map（box {128*SUBS, BM,1,1} 于 raw [N,K] 上，K 连续维内取
   SUBS*128 字节）；PREPACK=true 时走 v2 的 tile-major packed 4D map（每 tile 的 SUBS 个 k_block 连续）。
   两种布局的正确性都必须验证。
5. SPLIT_K_CTA：gridDim.y=2，每 CTA 处理一半 K（K/2 必须整除 128*SUBS*WGsPerTile）；
   先到的 CTA 写 ws 并 atomicAdd(sem)，后到的做最终 reduce+bf16 store；sem 自复位（launch 间无需清零）。
6. 数值路径与 v2 相同：combined_scale = w_scale × a_scale，每 k_block fmaf 进 partial，
   最后 __float2bfloat16_rn 写出。M_TILE>8 时 combined_scales smem 过大（M=128 时 57KB+）——
   允许加一个 SCALE_IN_SMEM 开关，false 时 math 线程直接 __ldg w_scale、寄存器缓存 a_scale
   （每 k_block 一次 load），保证 M_TILE≥64 可跑。
7. smem ≤ 227KB、寄存器预算 (WG*MATH_REGS + PROD_REGS)*128 ≤ 65536，static_assert 把关。
8. 文件 ≤ ~1100 行，每个技术一段注释块说明「为什么」，Zhihu 读者要能看懂。
9. 提供 `kernels/smoke_test.cu`：M=1/8、N=6144、K=7168，7 个 step 的 canonical config 全实例化，
   vs naive fp32 host 参考，rel_l2 ≤ 2e-3，PREPACK 两种布局都测。编译+跑通是 kernel agent 的完成门禁。

## 5. 权重布局（kernels/weight_prepack.cuh）

- raw：`[N,K]` fp8，K-contiguous（外部 checkpoint 原生布局，零预处理）
- packed（v2 同款）：`[output_tile][k_block][BM rows × 128B]` 全连续；
  BM 非 128 倍数时 pad 到 tile 边界（kPaddedN）。prepack kernel 照 v2 `prepack_weight_u4` 泛化：
  模板参数 (BM, TILES_PER_CTA)，runtime N/K。
- harness 调 `launch_prepack_weight<Cfg>(src, dst, stream)` 完成转换（timed region 外）。

## 6. Harness（bench/bench_decode.cu，由 harness agent 产出）

CLI：
```
bench_decode --models all|kimi_k3,glm52,... --steps 0-6|all --ms 1,2,4,8,16,32,64,128 \
             [--tune] [--reps N] [--out results/]
```
每个 (model, step, M) 记录：
- correctness：GPU fp32 参考（cublasLt SGEMM on dequantized inputs），rel_l2 ≤ 2e-3
- 三种计时协议（全部记录）：
  - ISO：单发 eventSync，每次 L2 flush + 旋转权重集，p50/30
  - B2B：同 stream back-to-back ×90，旋转权重集
  - PDL：B2B + ProgrammaticStreamSerialization 属性（step≥1 才有意义；step0 也记录以对比）
- 文章曲线口径：**step≥1 用 PDL 协议，step0 用 B2B**（诚实：PDL 是 step1 引入的技术）
- 冷权重：旋转 buffer 集，working set ≥ 512MB（sets = max(13, ceil(512MB/weight_bytes))）
- 输出 `results/bench_<UTC时间戳>.csv`，列：
  `model,N,K,M,m_tile,step,step_name,protocol,latency_us_p50,latency_us_p30,bandwidth_gbps,pct_of_spec_peak,clocks_sm_mhz,rel_l2,pass,config`
  （config 字段内部用 `;` 分隔键值对，不用逗号；protocol 取值固定大写 ISO/B2B/PDL；
   bandwidth_gbps ≥3 位小数、latency ≥4 位小数）
  bandwidth_gbps = (N*K + M*K + 2*M*N) bytes / latency（decode 口径：权重为主）
  pct_of_spec_peak = bandwidth_gbps / 4000
- `--tune` 模式：对给定 (model,step,M) 扫 config 网格（BM∈{32,48,64}, WG∈{1,2,4},
  STG∈{2,3,4}, SUBS∈{1,2}, TILES∈{1,2}, MATH_REGS∈{80,112,168}），输出 best config 表
- 编译拆分：bench_decode.cu 按 M_TILE 拆 5 个 TU（bench_m8.cu ... bench_m128.cu）+ 公共 main，
  Makefile -j 并行编，避免单 TU 实例化爆炸
- 运行前检查 GPU3 无其他 compute app；记录 clocks.sm 进结果 JSON

## 7. 峰值参考线（画图用）

- **主参考线：理论峰值 4.0 TB/s（HBM3 spec）**——用户指定，不许替换
- 辅线（虚线）：实测可达峰值，由 `tools/peak_bw_probe.cu` 诚实测出：
  TMA bulk load（SM90_TMA_LOAD_2D/4D）+ 多级流水（≥4 stage 双缓冲以上）+ 旋转冷 buffer
  （working set ≥ 4GB，远超 60MB L2）+ ≥156 CTAs；扫 (op 字节数, CTA 数, stage 数) 取 best。
  禁止任何可能 L2 命中的写法；probe 必须自报 working set 大小与 L2 flush 证据。
  结果写 `results/peak_bw.json`（含每配置明细，best 值 + 方法说明）。

## 8. 画图（analysis/plot.py，由 plot agent 产出）

- 输入：results/bench_*.csv（取最新或 --csv 指定）
- 每模型一张子图（2×3 布局，第 6 格放汇总/legend 或留空）：
  x = M（1..128，log2 刻度），y = bandwidth GB/s，
  每个 step 一条曲线（S0..S6，颜色渐变），
  水平实线 = 4000 GB/s（理论峰值，标注 "HBM3 spec 4.0 TB/s"），
  水平虚线 = 实测可达峰值（从 results/peak_bw.json 读，缺失则不画并警告）
- 叠加 roofline 等效带宽上限曲线（标准量纲定义）：
  BW_cap(M) = min(4000, bytes(M) / t_floor(M))，bytes(M) = N*K + M*K + 2*M*N（与 y 轴同口径），
  t_floor(M) = 2*M*N*K / 282.7e12 s（H20 fp8 实测峰值，写死常数并注明来源）。
  注意：原式 `2*M*N*K*1e-9/t_us` 量纲错误（分子是 FLOP 非字节），作废。
- 输出 results/fig_bw_vs_M_<model>.png + 汇总 fig_bw_vs_M_all.png（300dpi，中文字体可用则中文）
- 无数据时用 --demo 生成假 CSV 自测渲染
