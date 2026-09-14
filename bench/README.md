# bench/ —— decode_gemm benchmark harness

CONTRACT §6 的实现。主程序 `bench_decode.cu` **不 include kernel 头**：所有
`decode_gemm::KernelConfig<...>` 实例化都在 5 个按 M_TILE 拆的 TU 里
（`bench_m8.cu` … `bench_m128.cu`），通过 `bench::VariantOps` 函数指针注册表调用，
所以 `make -j` 能真正并行（CONTRACT §6「编译拆分」）。

---

## 0. 30 秒上手

```bash
cd $P/bench
make -j8 TUNE=0                      # 只烘 canonical 七步，~30s（跑主 sweep 够用）
./run_sweep.sh --demo                # 小 sweep 自测（1 模型 x step0,1 x M=1,8）
./run_sweep.sh                       # 全量：5 模型 x 7 step x M=1..128
./run_sweep.sh --models kimi_k3 --steps all --ms 1,8,64
```

**所有 GPU 运行必须经过 `run_sweep.sh`（它内部调 `tools/gpurun_dg.sh`）**，
不要自己 `CUDA_VISIBLE_DEVICES=... ./bench_decode`：

* CONTRACT §0（2026-09-13 修订）：只允许池 **{0,1}**；GPU2/3 被 weave_v1
  `v19_swapab` 的 keeper 预留（RELEASE_KEEPER 机制），**不许碰/不许 kill/不许等**。
* `tools/gpurun_dg.sh` 负责：跨项目 flock（`_lockbench_20260913/locks/gpu<N>.lock`）
  \+ 争用守卫（util≤5% / mem≤2000MiB / 无 compute-app pid）+ 锁频 `-lgc 1830,1830`
  \+ 自动选卡 + 设 `CUDA_VISIBLE_DEVICES`。
* `bench_decode` 拿到卡后**自己再复核一遍**：用 PCI bus id 反查物理卡号，
  若落在 GPU2/3 直接退出码 3；若卡上有别的 compute app 也退出码 3
  （`--allow-shared` 可强跑，但数据会被污染）。
* 实际用的物理卡号写进 `results/bench_<ts>.json` 的 `gpu.physical_index`
  （以及 `gpu.gpurun_gpu` / `gpu.gpurun_target`）。
* 排队等卡超过 10 分钟，`run_sweep.sh` 会往 `results/sweep_<ts>.log` 打
  `WAITING <秒>s … | <各卡 util/mem>` 心跳，之后每 120s 一次。

---

## 1. CLI

```
bench_decode --models all|kimi_k3,glm52,...   # models/models.json 的 id
             --steps  all|0-6|0,1,3           # 技术阶梯 S0..S6
             --ms     1,2,4,8,16,32,64,128    # M 阶梯（M<=8 -> M_TILE=8）
             [--tune] [--tune-only] [--reps N] [--out results/]
             [--iso-samples N] [--b2b-batch N] [--demo] [--list]
             [--resume [PATH]] [--no-resume] [--seed N] [--verbose]
             [--no-correctness] [--check-launcher] [--force-api-launch]
             [--allow-shared] [--weight-cache-gb X] [--l2-flush-mb N]
             [--models-json PATH]
```

| flag | 作用 |
|---|---|
| `--demo` | 自测小 sweep：第 1 个模型 x step{0,1} x M{1,8}，reps=1 / ISO 10 发 / batch 30 |
| `--reps N` | ISO 轮数 **和** B2B/PDL 批次数（默认 3）。想要更细的 p30/p50 就加大它 |
| `--iso-samples` / `--b2b-batch` | 覆盖自动降档（K≥16384 时自动 20 / 50） |
| `--tune` | canonical 之外再扫 CONTRACT §6 的 config 网格 |
| `--tune-only` | 只扫网格 |
| `--resume [PATH]` | 断点续跑（见 §5） |
| `--check-launcher` | 额外用 kernel 自带 `launch<Cfg>` 复算 rel_l2，交叉验证 harness 自建的 tensor map / grid / smem 与 kernel launcher 一致 |
| `--force-api-launch` | 计时也走 kernel launcher（**PSS attr 不再受 harness 控制，B2B 与 PDL 会变等价**，仅排查用） |
| `--list` | 不碰 GPU，打印模型表 + 已实例化 variant 数 + canonical config |
| `--weight-cache-gb` | 冷权重 set 组的显存预算（默认 12 GiB，LRU 淘汰） |

---

## 2. 三个计时协议（全部用 CUDA event）

| 协议 | 做法 | p50/p30 来自 |
|---|---|---|
| **ISO** | 每次：256MB dummy 写 flush L2 → sync → `eventRecord` → **单发** → `eventRecord` → `eventSynchronize`；权重集逐个轮转 | `reps × iso_samples` 个单发样本 |
| **B2B** | 一个 event 窗口里**同 stream 连发 batch 次**（默认 90，K≥16384 降 50），轮转权重集，无 per-launch host sync | `reps` 个「窗口总时长 / batch」值 |
| **PDL** | 与 B2B 完全相同，只是每次 launch 多带一个 `cudaLaunchAttributeProgrammaticStreamSerialization` | 同 B2B |

三个协议**共用同一条 launch 路径**（harness 自己 `cudaLaunchKernelEx`，见 §6），
唯一差别就是那个 attribute，所以 B2B↔PDL 的差就是 PDL 的净收益。

**文章曲线口径**：`step≥1 用 PDL，step0 用 B2B`（PDL 是 S1 才引入的技术，
S0 的 kernel 不发 `griddepcontrol.launch_dependents`，PDL 行只是对照）。
harness 每点都会在 stdout 打 `curve(PDL|B2B)=…GB/s`。

> `cudaStreamSetAttribute(cudaLaunchAttributeProgrammaticStreamSerialization)`
> **在 CUDA 12.8 上返回 `invalid argument`**（`probe/probe_pdl_stream_attr.cu`
> 实测：mode1 无效、mode2 `cudaLaunchKernelEx` 有效，20 发 2000µs→512µs、
> 19/19 重叠）。所以 PDL 只能靠 per-launch 属性，这就是 harness 要自己 launch 的原因。

warmup：每个测量点先 10 发（timed region 外）。

---

## 3. 口径（诚实声明）

* **带宽分子** = `N*K + M*K + 2*M*N` bytes（权重 fp8 + 激活 fp8 + 输出 bf16；
  **不含 scale**，CONTRACT §6 明确）。`N` 用 logical N。
* `bandwidth_gbps = 分子 / latency_us / 1e-6 / 1e9`
* `pct_of_spec_peak = bandwidth_gbps / 4000`（CONTRACT §6 的公式：**比值**，不是百分数；
  4000 GB/s = HBM3 spec，主参考线，不许替换）
* **冷权重轮转**：`sets = max(5, ceil(512MiB / weight_bytes))`（orchestrator 2026-09-13
  把下限从 13 调到 5：dpsk 权重 117MB → 5 set 就够 512MB working set；qwen36 8MB → 64 set）。
  每个 set 是独立 buffer，PREPACK 布局的 set 各自跑一遍 `launch_prepack_weight`
  （在 timed region 外）。轮转保证 L2（60MB）里永远没有热权重。
* **L2 flush**：256MB `float4` 写（`--l2-flush-mb` 可调），只在 ISO 协议里做，
  在 event 窗口之外 + sync 之后才开始计时。
* **correctness**：权重/激活 dequantize 到 fp32 → `cublasLt` SGEMM
  (`CUBLAS_COMPUTE_32F`) → 与 kernel 的 bf16 输出比 `rel_l2 ≤ 2e-3`。
  参考只依赖 (model, M)，所以**每 (model,M) 算一次**（比 CONTRACT 要求的
  「每 (model,step,M) 一次」更省，结果一致）。
* **大 K 自动降档**：`K ≥ 16384` 时 ISO 样本 30→20、batch 90→50（orchestrator 补充），
  实际用的值写进每行明细与 JSON。
* `clocks_sm_mhz`：**每行**都写实测值（`nvidia-smi --query-gpu=clocks.sm`，
  每点强制读一次、行内 1s TTL 缓存）。gpurun_dg 锁 1830，正常应该恒为 1830；
  读到 `-1` 表示 nvidia-smi 没答上（plot 侧按脏数据降级）。

---

## 4. 输出文件

`--out`（默认 `<exe>/../results`，即 `$P/results`）下：

| 文件 | 内容 |
|---|---|
| `bench_<UTC>.csv` | **CONTRACT §6 的 16 列，列名/顺序一字不差**，plot agent 直接吃 |
| `bench_<UTC>_detail.csv` | 同 16 列 + 明细：`latency_us_min/mean,samples,reps,iso_samples,b2b_batch,pdl_batch,variant_kind,tune_index,rank,is_best,padded_n,weight_sets,weight_set_bytes,smem_bytes,num_ctas,num_threads,launch_mechanism,rel_l2_launcher` |
| `bench_<UTC>.json` | 运行摘要：GPU（含**实际物理卡号**）、clocks、独占检查、三协议定义、带宽口径、CSV schema、resume 状态、skipped 明细、tune best、耗时 |
| `tune_<UTC>.csv` / `_detail.csv` | `--tune` 的网格逐点结果（16 列 schema 相同） |
| `tune_best_<UTC>.csv` | `--tune` 的排名表（`rank,is_best,rank_protocol,…`），第 1 名就是 best config |
| `sweep_<UTC>.log` | `run_sweep.sh` 的完整日志（含 WAITING 心跳） |

主 CSV 的 16 列：

```
model,N,K,M,m_tile,step,step_name,protocol,latency_us_p50,latency_us_p30,bandwidth_gbps,pct_of_spec_peak,clocks_sm_mhz,rel_l2,pass,config
```

格式约定（orchestrator 2026-09-13 schema 增补）：

* `protocol` 固定大写 `ISO` / `B2B` / `PDL`，每个 (model,step,M) 三行
* `config` 内部用 **`;`** 分隔键值对，绝不出现逗号，例如
  `M_TILE=8;BM=56;WG=4;REG=112;STG=2;TILES=1;SUBS=1;PREPACK=0;PDL=0;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1`
  （被自动降级过会多一个 `;DOWNGRADED=1`）
* `latency_us_*` ≥4 位小数，`bandwidth_gbps` ≥3 位小数，`pct_of_spec_peak` 6 位，
  `rel_l2` 科学计数 6 位
* `pass`：`1` / `0` / `na`（`--no-correctness` 时是 `na`，`rel_l2=nan`）
* 测不出来的值是 `nan`（不是 0），plot 侧按脏数据降级

**追加列只出现在 `_detail` / `tune_best` 文件里**，主 CSV 严格 16 列，
免得 plot.py 逐列对齐时被绊倒。

---

## 5. 增量落盘 + 断点续跑

* **增量落盘**：每测完**一个协议**就 `write + flush` 一行（不是等一个点三协议全跑完）。
  被 kill / timeout 最多丢掉正在测的那一个协议。
* **断点续跑**：`--resume`（`run_sweep.sh` 默认带）启动时读 `--out` 里**最新的**
  `bench_*.csv`，把已有的 `(model,N,K,M,m_tile,step,protocol,config)` 组合记下来，
  本次只补缺口，并 **append 到同一个 CSV**（不写第二份表头）。
  * `latency_us_p50` 是 `nan`/空的旧行**不算完成**，会被重测。
  * 会校验 sibling `bench_<同ts>.json` 的 `kernel_backend`：**stub 的结果不会被
    真 kernel 的运行续跑**（反之亦然），不匹配就新开一份 CSV 并打 WARNING。
  * `--resume PATH` / `--resume-from PATH` 可显式指定要续的文件；`--no-resume` 强制新开。
  * `--tune` 时同样续跑 `tune_*.csv`（key 里带 config，所以网格逐 config 补）。
* 每次运行的 JSON 里有 `resume` 段：`source / rows_skipped / points_skipped /
  rows_written_this_run`，以及 `outputs.appended`。
* 整点跳过：三协议都有数时连 warmup / prepack / tensor map 都不做，续跑很快。

---

## 6. 与 kernel 的接口（联调点）

harness 只用 CONTRACT §4/§5 点名的符号，且全部在
`probe/probe_cfg_asserts.cu` 里做过编译期验证（`make probe-cfg`，几秒，不跑 GPU）：

| 用途 | 符号 |
|---|---|
| 实例化 | `decode_gemm::KernelConfig<M_TILE,BM,WG,REGS,STG,TILES,SUBS,PREPACK,PDL,EPI,SPLITK,PR,SPW,PLA,SISM>` |
| 形状可行性 | `Cfg::validate(M,N,K)`（返回 nullptr 或原因串；harness 直接把它写进 skipped） |
| 几何上报 | `Cfg::num_ctas_x(N)`、`Cfg::dynamic_smem_bytes(M,K)`、`Cfg::kNumThreads`、`Cfg::kFixedSmemBytes` |
| 权重布局 | `prepack::padded_n(N,BM)`、`prepack::packed_weight_bytes(N,K,BM)`、`prepack::launch_prepack_weight<BM,TILES>(src,dst,N,K,stream)` |
| tensor map | `make_weight_tma<Cfg>(Problem)` / `make_activation_tma<Cfg>(Problem)`（**timed region 外**预建，每个冷权重 set 一份） |
| launch | `make_launch_plan<Cfg>` + `ensure_smem_attribute<Cfg>` + `cudaLaunchKernelEx(fp8_decode_gemm<Cfg>, out, w_scales, a_scales, w_tma, a_tma, GemmRuntime{M,N,K,ws,sem})` |
| 交叉验证 | `launch<Cfg>(Problem, stream)`（`--check-launcher`） |

**为什么 harness 自己 launch 而不是直接调 `launch_with_maps<Cfg>`**：
kernel 的 `launch_with_maps` 在 `Cfg::kPdl` 为真时**恒开** PSS 属性，那样
B2B 协议也会被 PDL 加速，S0→S1 的收益就测不出来了。harness 用
`make_launch_plan<Cfg>` 拿到 kernel 自己算的 grid/block/smem，然后自己
`cudaLaunchKernelEx`，attr 由协议决定（ISO/B2B 关、PDL 开）。
`VariantOps::mechanism` 会把这个事实写进明细列 `launch_mechanism`：

* `kernel_ex` = harness 控制 attr（正常情况，三协议可比）
* `api` = 退回了 kernel launcher（`-DBENCH_NO_DIRECT_LAUNCH`），此时
  **B2B 与 PDL 等价，不能用来讲 PDL 收益**，JSON 里会标红说明

其它约定：

* `Problem.N` 传的是 **pad 到 128 倍数后的 N**（CONTRACT §2「pad 由 harness 做」）；
  CSV 的 `N` 列仍是 logical N，pad 情况写在 JSON 与 stdout WARNING 里。
  当前 5 个模型的 N 都是 128 的倍数，这条路径不会触发。
* `output` 按 `M × padded_n(BM=32)` 分配（比 `M×N` 大），即使 kernel 写满整个
  tile 也不会越界；比较时只取前 `N` 列、行距按 `Problem.N`。
* `weight_scales` 多分配 1 行、`activation`/`activation_scales` 多分配到
  `max(M_TILE,8)` 行并清零，避免 pad tile 的 scale 越界读。
* `splitk_ws` = `2 × M × padded_n` float、`splitk_sem` = `M × padded_n` int32，
  分配后清零（CONTRACT §4 note 5：host 保证清零，kernel 自复位）。
* `smem > 48KiB` 的 `cudaFuncSetAttribute` 由 kernel 的 `ensure_smem_attribute<Cfg>`
  负责，harness 在每个测量点 setup 时调一次。

---

## 7. config 表（`configs.cuh`）

**canonical BM 按形状选**（orchestrator 2026-09-13 补充）：候选 `{32,40,48,56,64}`，
先要求整除 N（pad=0），再要 output tile 数最贴近 128（v2 实测甜区），最后 BM 大者优先：

| N | 选中 BM | tiles | 备选顺序 |
|---|---|---|---|
| 7168 (kimi_k3 / deepseek_v4_pro) | **56** | 128 | 56 → 64 → 32 → 48 → 40 |
| 6144 (glm52 / minimax_m3) | **48** | 128 | 48 → 64 → 32 → 56 → 40 |
| 2048 (qwen36) | **32** | 64 | 32 → 64 → 40/48/56(pad) |

5 个候选 BM 的 canonical 都会实例化，运行时按 N 选（所以 `--list` 里
每个 M_TILE 是 `7 step × 5 BM = 35` 个 canonical variant）。

> ⚠ **这条几何规则已被 tune 数据部分否掉**（qwen36 的 BM=32 在 M=1 掉 10%，
> 见 §7.1）：v2 起 S4/S5/S6 的 BM 与其余旋钮由 `configs.cuh::kShapeOverrides`
> 按 (形状, M_TILE) 用实测 winner 覆盖，几何规则只作为 S0–S3 与兜底默认。

**M_TILE=8 的七步 = CONTRACT §3 原样**（REG 取 v2 终版：S0–S3=112，S4–S6=80）：

| step | 名字 | config（N=6144 时） |
|---|---|---|
| S0 | baseline | BM=48 WG=4 REG=112 STG=2 TILES=1 SUBS=1 raw，无 PDL |
| S1 | pdl | S0 + PDL |
| S2 | blockk | S1 + SUBS=2 |
| S3 | prepack | S2 + PREPACK |
| S4 | tile_stage | BM=48 WG=2 REG=80 STG=3 SUBS=2 PREPACK PDL |
| S5 | epi_overlap | S4 + EPI |
| S6 | splitk | S5 + SPLITK |

### ⚠ 待同步：M_TILE ≥ 16 的 canonical 表是**占位**（v2 由 kShapeOverrides 覆盖）

按 CONTRACT §3 末尾的规则填的（`TILES=WG`；M_TILE=128 → `REG=168/WG=2/STG=2`；
REG 按累加器需求 `≥ M_TILE+40` 选 80/112/168；S2 起 SUBS=2），
**等 `kernels/README.md` 的 config 表定稿后必须同步 `configs.cuh::canonical_base()`**。
预算不够时 `canonical_for()` 会自动降级 `SUBS → STG → WG/TILES` 并在 config 串里
打 `DOWNGRADED=1`（CONTRACT §3「跑不通时降级 STG/SUBS 并在 README 记录」）——
JSON 里能查到，出现了就要回来改表。

### --tune 网格

CONTRACT §6 的 `BM∈{32,48,64} WG∈{1,2,4} STG∈{2,3,4} SUBS∈{1,2} TILES∈{1,2}
REG∈{80,112,168}`，**外加 orchestrator 要求的 BM=40/56** → 540 项/step。
排名口径 = 文章口径（step≥1 用 PDL，step0 用 B2B）；rel_l2 不过 2e-3 的 config
不参与排名。

编译期闸门 `configs.cuh::feasible()` 镜像 kernel `KernelConfig` 的 static_assert
（寄存器预算、`kFixedSmemBytes ≤ 227KiB`、`2*TILES+1 ≤ 16`、`EPI→PDL`、
`PRELOAD_ACT` 限制、`M_TILE≥16 → TILES==WG`），再加一条 harness 自己的保护
`REG ≥ M_TILE+40`（累加器装不下会 spill/编不过）。**只有过闸门的 config 才会被
实例化**，所以 kernel 的 static_assert 不会被 harness 炸出来。与形状相关的约束
（SUBS/WGsPerTile 整除、act+scale smem）留给运行时 `Cfg::validate()`。

过闸门后的数量（每 step）：M8=472、M16=236、M32=236、M64=148、M128=60。
7 个 step 全烘 ≈ 8000 个实例化，很慢，所以有开关：

```bash
make TUNE=0                 # 只 canonical（~30s）        <- 主 sweep 用这个
make                        # canonical + TUNE_STEPS(默认 0,4,6) 的网格
make TUNE_STEPS=all         # CONTRACT §6 全量（7 step），约 20 分钟
make TUNE_STEPS=6           # 只烘 S6 的网格
make tune-bin               # 另出 bench_decode_tune（TUNE_STEPS=all），不动主二进制
```

如果 `--tune` 的某个 step 没烘进二进制，harness 不会静默跳过：会打
`该 step 的 tune 网格没烘进这个二进制（Makefile: TUNE=1 TUNE_STEPS=all…）`
并记进 JSON 的 `skipped`。

### 7.1 config 选择的经验规律（tune2/tune3 实测，2026-09-13）

`--tune` 扫完 5 形状 × M∈{1,8,16,32,64,128}（tune2 = M∈{1,16,64}，harness 侧；
tune3 = M∈{8,32,128}，orchestrator 侧）之后，**「tile 数贴近 128」这条纯几何规则
被数据否掉了**，canonical 必须按 (形状, M_TILE) 用实测 winner 覆盖
（`configs.cuh::kShapeOverrides`）。

每格 = 固定该 BM、扫遍其余旋钮后的最好成绩（PDL，GB/s，SUBS=2/PREPACK=1）：

| model | M | BM=32 | BM=48 | BM=56 | BM=64 | 全场最好 |
|---|---|---|---|---|---|---|
| kimi_k3 (N=7168) | 1 | 2795.0 | **3428.7** | 3402.4 | 3299.2 | BM48;WG2;REG80;STG4 |
| kimi_k3 | 16 | 1700.8 | 2304.6 | **2335.8** | 2328.9 | BM56;WG1;REG80;STG4 |
| kimi_k3 | 64 | 333.9 | 636.0 | 636.9 | **638.9** | BM64;WG2;REG168;STG3;TILES2 |
| deepseek_v4_pro (7168) | 1 | 3345.8 | 3402.5 | **3487.7** | 3472.6 | BM56;WG2;REG80;STG2 |
| glm52 (6144) | 1 | 3120.0 | 3373.4 | 3330.9 | **3431.0** | BM64;WG1;REG80;STG4 |
| glm52 | 64 | 285.5 | **543.7** | 542.2 | 541.9 | BM48;WG2;REG168;STG3;TILES2 |
| minimax_m3 (6144) | 1 | 3007.5 | 3065.5 | 3223.7 | **3259.7** | BM64;WG1;REG80;STG4 |
| qwen36 (N=2048) | 1 | 2225.6 | 2152.4 | **2291.5** | 2280.9 | BM56;WG4;REG96;STG3 |
| qwen36 | 16 | 1200.3 | 1206.0 | 1258.0 | **1267.0** | BM64;WG1;REG232;STG4 |
| qwen36 | 64 | 275.5 | 310.5 | 397.9 | **398.0** | BM64;WG1;REG232;STG4 |

**BM=32 在 30 个 (model,M) 格子里一次都没赢过**，而且 M 越大输得越惨
（kimi M=64：BM32 最好 333.9 vs BM64 638.9，差 1.9×）。

#### 小 N 上「更长的顺序流」比「更多 CTA」重要（qwen36 的 S4 降级现象）

v1 canonical（`results/bench_20260913T0711*.csv`）里 qwen36 是唯一在 S4 掉档的模型：

| step | config | PDL GB/s |
|---|---|---|
| S3 prepack | `BM=64;WG=4;REG=112;STG=2` | 2074.1 |
| S4 tile_stage | `BM=32;WG=2;REG=80;STG=3` | **1857.5（−10.4%）** |

S4 按几何规则把 BM 从 64 降到 32（2048/32 = 64 tiles，更「贴近 128」），
同时换成大 N 常用的 `WG=2/REG=80/STG=3` 束。tune 数据把这个 −10% 拆开：

* **不是噪声**：同一个 config 在 tune 窗口重测 = 1862.3 GB/s（差 0.3%）。
* **BM=32 本身只值 −2.9%**：`BM=32` 的最优搭配（STG=4）能到 2225.6，
  而全场最优 `BM=56;WG=4;REG=96;STG=3` = 2291.5。
* **真正的大头是流水线深度**：同样 BM=32，STG 2→3→4 = 1558.4 → 1862.3 → 2225.6
  （+43%）。N=2048、K=4096 意味着每 CTA 只有 32~64 行权重、K 方向只有几十个
  k-block，**STG 不够就没有足够的在途 TMA 请求盖住延迟**；同时 64 个 CTA 铺在
  78 个 SM 上本来就填不满机器，把 N 再切细（更多 CTA）换不来任何占用率，
  只是把每个 CTA 的顺序权重流切得更短。
* 反过来在大 N（kimi/dpsk，N=7168）上 BM=32 → 224 CTA，此时「多 CTA」也不是
  赢家（M=1 时 BM32 比 BM48 低 18.5%），因为每 CTA 的固定开销（prologue/epilogue、
  tensor-map、semaphore）被更少的字节摊薄。

一句话：**小 N（2048）上，M 小的时候拼的是「每个 CTA 的顺序流够长 + 在途 stage 够深」，
不是「CTA 够多」；等 M 增大，数学时间占比上升，BM 的选择被下面这条流量律主导。**

#### M 增大后 BM 变小 = 激活重读爆炸（口径提醒）

harness 报的 `bandwidth_gbps` 分子是**有用字节** `N*K + M*K + 2*M*N`（CONTRACT §6 口径），
但真实 DRAM 流量里激活会被**每个 CTA 列各读一次**：

```
traffic ≈ N*K  +  ceil(N/BM) * M*K  +  2*M*N
                 ^^^^^^^^^^^^^^^^^ 这一项不在分子里，且 ∝ M / BM
```

qwen36 M=64：BM=32 → 8.39MB 权重 + 64 CTA × 0.262MB 激活 = 25.4MB；
BM=64 → 8.39 + 32 × 0.262 = 17.0MB。流量比 1.49，实测带宽比 398.0/275.5 = 1.45
（差 3%，L2 命中让重读便宜了一点）。dpsk M=64 同理解释了 1.9× 的落差。
**所以「GB/s」不能读成 DRAM 效率**，它是有用字节口径；跨 BM 比较时小 BM 天然被
高估，这也是排名要用同一口径 + 直接看 latency 的原因（`csv_tool.py best` 按
bandwidth_gbps 排，同 (model,M,step) 内分子恒定，等价于按 latency 排）。

### 7.2 --tune 方法学（怎么跑、怎么落盘、怎么选 winner）

**二进制**：`make tune-bin TUNE_GRID=s4` → `bench_decode_tune`
（642 registered = 208 canonical + 434 tune）。
网格（orchestrator 2026-09-13 修订版，只扫 S4+ 的旋钮）：

```
BM ∈ {32,48,56,64} × (WG,TILES,REG) 组合 × STG ∈ {2,3,4}，SUBS=2 / PREPACK=1 固定，
EPI=0 / SPLITK=0（= S4 base；S5/S6 由 winner 叠加）
```

REG 受 kernel `__launch_bounds__` 的硬上限约束（**WG=4 → 最多 96 regs，WG=2 → 168，
WG=1 → 232**；kernel agent 实测：WG=4 时请求 112/168 都只拿到 96），
所以网格里 WG=4 只留 REG=96，不重复实例化 112/168。
过闸门后每 M_TILE 的 config 数：**M8=150、M16=91、M32≈91、M64=54、M128≈36**
（`skipped` = 形状相关的 `Cfg::validate()` 或 smem 预算不过，会记进 JSON）。

**协议 / 计时**：只跑 PDL（文章口径 step≥1 用 PDL），`--reps 2`，batch=90 连发；
correctness 用**单发**结果比对 cublasLt fp32 参考（PDL 连发写同一 output buffer
是语义 race，虽然写同值实践无害，但不能拿来做参考比对），rel_l2 > 2e-3 的 config
记 `pass=0`，不参与排名。

**落盘 / 断点**：每 config 一行，写完立刻 `flush`
（borrow 池的 2s 看门狗一旦看到外来 compute 就 abort(exit 75)，最多丢当前 config）。
`--tune-out results/tune_<model>.csv` 固定文件名 + purge&append，
所以分批跑 / abort 重跑都往同一个文件补缺口。

> ⚠ 踩过的坑（已修，2026-09-13）：`purge_incomplete_groups()` 原来把「组完整」
> 硬编码成 ISO+B2B+PDL 三行都在。tune 只跑 PDL 时每个 config 都被判成不完整 →
> 下一个 chunk 启动时把上一个 chunk 的行**全删了**（实测 `tune_qwen36.csv`
> 只剩最后一块 M=64 的 54 行，日志里 `purged 150 rows in 150 incomplete groups`）。
> 现在 purge 按 `--protocols` 请求的协议集判定完整度（PDL-only 就只要 PDL 一行），
> 主 CSV / tune CSV / ablation CSV 三条路径都吃这个修复。

**分块并行**（`bench/run_tune_campaign.sh A|B`）：5 模型分两路
（A=kimi/glm/dpsk，B=qwen/minimax），每 (model,M) 一个 chunk 单独调
`tools/gpurun_dg.sh`（flock 串行 + abort 自动重排 + `--tune-out` 续跑），
两路各吃一张卡互不干扰。成本极低：**每 chunk 1–4s GPU**（M=1 的 150 config 只要
1.1s），5 模型 × 3 个 M 的整轮 campaign ≈ 100s。

**选 winner**：

```bash
python3 bench/csv_tool.py best results/tune_best_v2.csv results/tune2_*.csv results/tune3_*.csv
```

按 bandwidth_gbps 最大取每 (model,M,step) 的 config；orchestrator 把它填进
`configs.cuh::kShapeOverrides`（每 model × m_tile 一行），
S5 = winner + `EPI=1`，S6 = winner + `EPI=1` + `SPLITK`（仅当
`ctas_no_split = ceil(N/BM) < 78` 且 `M ≥ 8`；qwen 实测 M=1/4 开 split-K 反降 24%、
M=16/64 升 35–41%，大模型 128 CTA 全关）。

**方法学上的两个 caveat**（文章里别过度解读）：

1. tune 只在 M∈{1,8,16,32,64,128} 上扫；M=2/4 与 M=1 共用 M_TILE=8 的 winner，
   没有单独验证过（这几档的曲线是 canonical 主 sweep 出的，不是 tune 出的）。
2. winner 是「单形状单 M」的最优，不是跨 M 的鲁棒最优。同一 M_TILE 内不同 M 的
   winner 可能不同（例：kimi M_TILE=8 在 M=1 是 BM48;WG2;REG80;STG4），
   `kShapeOverrides` 一行只能存一个，取的是该 M_TILE 覆盖档位里最有代表性的那档。


---

## 8. 文件清单

| 文件 | 作用 |
|---|---|
| `bench_decode.cu` | 主程序：CLI、models.json、host 量化数据、冷权重轮转、三协议计时、cublasLt 参考、CSV/JSON |
| `bench_m8.cu` … `bench_m128.cu` | 按 M_TILE 拆的实例化 TU（canonical + tune 网格），`make -j` 并行 |
| `configs.cuh` | step→canonical config 映射、BM 选择规则、tune 网格枚举、编译期可行性闸门（纯 constexpr） |
| `bench_common.cuh` | 主程序 ↔ 实例化 TU 的唯一耦合面（`VariantOps` 注册表、`DeviceBuffers`、`LaunchRequest`） |
| `variant_impl.cuh` | 把 `KernelConfig<...>` 包成 `VariantOps`（含 harness 自己控制的 PDL attr launch） |
| `kernel_api.h` | kernel 头选择器：默认 `kernels/decode_gemm.cuh`，`-DBENCH_USE_STUB` 时走 stub |
| `stub_kernel.h` | **仅流程自测**的假 kernel（见 §9） |
| `json_min.h` | 极简 JSON 读取器（只为 `models/models.json`，无第三方依赖） |
| `Makefile` | 并行编译 + 链接（编译命令严格按 CONTRACT §0）+ stub/tune-bin/probe 目标 |
| `run_sweep.sh` | 全流程：gpurun_dg 选卡 → 跑 → 收结果 → WAITING 心跳；`PLOT=1` 时尾部顺带出正式图 |
| `run_tune_campaign.sh` | `--tune` 分块驱动（A/B 两路分模型，每 (model,M) 一个 chunk，abort 自动重排） |
| `run_v2_extras.sh` | 等 v2 重编落地后自动跑：split-K 消融 → kimi M=1 方差 → 一致性核对 → 出图 |
| `csv_tool.py` | 离线 CSV 工具（不碰 GPU）：`merge` / `drop-steps` / `stats` / `variance` / `best` |
| `check_consistency.py` | v2 正式数据落地核对：16 列表头、21 行/(model,M)、pass、clocks=1830、PDL<B2B、带宽阶梯 |
| `make_plots.sh` | 出正式图：`results/bench_2*.csv` 按 mtime 时间序喂 `analysis/plot.py`（后者覆盖前者同 key 行） |
| `probe/probe_cfg_asserts.cu` | 编译期闸门一致性检查 + 直发 launch 符号存在性检查 |
| `probe/probe_pdl_stream_attr.cu` | 实测 PSS 只能靠 `cudaLaunchKernelEx`（不能靠 `cudaStreamSetAttribute`） |

---

## 9. stub 模式（`make stub`）—— 仅供流程自测

```bash
make stub -j8                       # -> bench_decode_stub（同一个源码树，-DBENCH_USE_STUB）
BIN=./bench_decode_stub ./run_sweep.sh --demo
```

`stub_kernel.h` 提供与真 kernel **同名同形**的 API，内部是 naive fp32 GEMM：

* 能自测的东西：CLI、models.json 解析、CSV/JSON schema、增量落盘 + 断点续跑、
  冷权重轮转 + prepack（packed 布局与 `kernels/weight_prepack.cuh` 逐字节一致）、
  tensor map 预建、三协议计时框架、PDL attr 是否真的生效（stub 在 kernel 开头就发
  `griddepcontrol.launch_dependents`，所以 PDL 会明显快过 B2B）、cublasLt 参考 +
  rel_l2 判定。
* **不能**用来看性能：没有 wgmma/TMA/流水线，慢几个数量级，数字无意义。
* SPLIT_K_CTA 被 stub 忽略（每 CTA 算全 K 直接写 bf16），输出仍正确；
  harness 侧 ws/sem 的分配与清零照跑。
* stub 的 `kDynamicSmemBytes` 用与 kernel 同一个估算式，所以 smem 上报链路也能验。
* JSON 的 `kernel_backend` 会写明是 stub，`--resume` 因此不会把 stub 结果当真 kernel 续跑。

---

## 10. 模型形状

运行时以 `models/models.json` 为准（shapes agent 定稿）：

| id | N | K | 权重(fp8) | sets = max(5, ceil(512MiB/W)) |
|---|---|---|---|---|
| kimi_k3 | 7168 | 12288 | 84 MiB | 7 |
| qwen36 | 2048 | 4096 | 8 MiB | 64 |
| glm52 | 6144 | 16384 | 96 MiB | 6 |
| deepseek_v4_pro | 7168 | 16384 | 112 MiB | 5 |
| minimax_m3 | 6144 | 8192 | 48 MiB | 11 |

`models.json` 缺失/损坏时才用内置兜底表（形状与上表相同）并打 WARNING；
JSON 的 `models_json.fallback_used` 会记下来。

显存预算：冷权重 set 组按 `(prepack,BM,TILES)` 缓存，LRU 上限 `--weight-cache-gb`
（默认 12 GiB，orchestrator 要求总占用别超 20 GiB）。canonical 主 sweep 每个模型
只需 2 组（raw + packed），tune 才会长到 10 组左右。

---

## 11. v2 extras：方差 / split-K 消融 / 一致性核对 / 出图

分工（orchestrator 2026-09-13）：**configs.cuh 的 kShapeOverrides + S4/S5/S6 重跑
由 orchestrator 做**；harness 侧负责下面四件事，全部由
`bench/run_v2_extras.sh` 串起来（它会先等 `bench/bench_decode` 被重编成 v2，
再开跑，避免用旧 canonical 测出跟正式曲线不同源的数字）：

```bash
bash bench/run_v2_extras.sh                 # 等 v2 重编 -> (1)(2)(3)(4) 全跑
NOW=1 bash bench/run_v2_extras.sh           # 不等，用现二进制立刻跑
ONLY=ablation|variance|check|plot bash bench/run_v2_extras.sh
tail -f /tmp/v2_extras.log
```

1. **split-K 消融** → `results/ablation_splitk.csv`（主 CSV 的 16 列 **+ `splitk_on`**）
   `--models kimi_k3,qwen36 --steps 6 --ms 1,8,64 --splitk-ablation`，
   on/off 各跑一次、三协议全跑 = 36 行。消融 run 的 canonical 主 CSV 落到
   `COLLECT=/tmp/v2extras_collect`，**不混进 `results/` 顶层的曲线数据**。
2. **方差** → `results/variance_kimi_m1.csv`
   kimi_k3 × M=1 × 全 7 step × PDL，3 个独立 batch（`--seed 1137/1274/1411`，
   批间隔 45s，不同时间窗），再用 `csv_tool.py variance` 合成
   每 (model,M,step,protocol,config) 的 mean/std/CV/min/max + 逐轮行 → 文章写 mean±std。
   每 batch 的原始 CSV 留在 `results/var_b{1,2,3}/`。
3. **一致性核对** → `results/consistency_report.txt`（`check_consistency.py`）
   16 列表头逐字校验、每 (model,M) 21 行（7 step × 3 协议，M_TILE=128 的
   `wg1reg232=1` 备选行与 S6 的 split-K on/off 对算 extra 单列）、`pass` 全 1、
   `clocks_sm_mhz` 全 1830、step≥1 的 PDL 延迟 < B2B（PSS attribute 真生效的判据）、
   latency ≥4 位小数 / bandwidth ≥3 位小数、bandwidth 阶梯表 + M=1 单调性。
   退出码 0=OK / 1=有硬错误（缺行、fail、时钟不对、表头不对）。
4. **出图** → `results/fig_bw_vs_M_<model>.png` + `fig_bw_vs_M_all.png`
   （`make_plots.sh`；`PREFIX=fig` 是默认，**不碰 orchestrator 的 `fig_v1_*` 存档图**）。

核对现状（v1 数据，2026-09-13 07:49）：840 行、40 个 (model,M) 组各 21 行、
`pass` 全 1、`clocks` 全 1830、PDL<B2B 违反 0 条 → `结论: OK`。
5 条 WARN 全是「M=1 阶梯非单调」，即 qwen36 的 S4 降级（§7.1）与各模型 S6 的
split-K 回退（v1 canonical 对大模型无条件开 split-K，v2 改成
`ctas_no_split<78 且 M≥8` 才开）。
