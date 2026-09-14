# tools/ — 峰值带宽探针与 GPU 调度

## peak_bw_probe.cu：诚实的「实测可达读带宽」

用途：给画图提供辅助虚线（主参考线是 HBM3 理论峰值 4.0 TB/s，见 CONTRACT §7）。

### 为什么不能随便写个 read bench

历史教训（shared/ANALYSIS_decode_gemm.md §3.1）：曾有 bench 报 4.06 TB/s（101.5%），
超物理。根因是 box 坐标每次迭代只前进 128 B，128 次迭代反复读同一个 ~48 KB 窗口，
**全部 L2 命中**——测的是 L2 不是 DRAM。本探针从设计上排除这类假象：

1. **read-once 旋转冷数据**：NUM_BUFS 个旋转 buffer，每次 launch 只读其中一个且推进索引，
   总 working set ≥ 4 GiB ≫ 60 MiB L2，任何 cache line 都活不到下次访问同一 buffer。
2. **no-rotate 消融自证**：不旋转、每次读同一 512 MiB buffer 的对照实验与主协议差 ~0.1%
   ——证明冷度来自「单次读体量 ≫ L2」这个事实，而不是旋转技巧本身。
3. **TMA + 深流水**：`cp.async.bulk.tensor.4d`（SWIZZLE_128B / L2_PROMOTION_256B）进
   STAGES 级 smem 流水线；1 个 producer warp 发 TMA，7 个 consumer warp 只做
   full.wait + 每 stage 一次 16 B smem 读 + empty.arrive，读结果 XOR 累加写回 gmem sink
   （防编译器消除）。
4. **单次 launch 体量 ~512 MiB ≈ 146 µs @3.5 TB/s** ≫ 启动开销，event 对只包单次 launch。
5. **L2 flush 消融**：256 MiB memset flush 放在 event 对之外，作为保守对照单独报告
   （真实 decode GEMM 不携带这种写流量，flush 的 write-back 会污染测量窗）。

### 结果（results/peak_bw.json）

| 项 | 值 |
|---|---|
| 理论峰值（画图主参考线） | 4.0 TB/s（HBM3 spec） |
| 实测可达 best | **3670.6 GB/s**（91.8% of spec） |
| best 配置 | op=80 KB × 624 CTA × 2 stage |
| 朴素 ld.global.nc.v4 对照 | 3612.7 GB/s |
| working set | 6.98 GiB |
| GPU / 锁频 | GPU0 / 1830 MHz |

扫描维度：每 TMA op 字节（5/10/20/40/80 KB）× CTA 数（78/156/312/624）× stage（2/4/8）。
每配置 ≥3 轮取中位。结论：H20 上「单次大块顺序流」的可达上限 ≈ 3.67 TB/s，
decode GEMM 的 85% of spec ≈ 93% of achievable——已经贴着模式天花板。

## gpurun_dg.sh：跨项目 GPU 安全调度

GPU2/3 被 weave_v1 v19_swapab 项目的 keeper 进程预留（RELEASE_KEEPER 机制），
协议明令不许碰/kill/等。空闲池只有 GPU0/1，且 lockbench 团队（_lockbench_20260913）也在用。
wrapper 做四件事：

1. 在池 {0,1} 找通过争用守卫的卡：util ≤5% 且 mem ≤2000 MiB 且无 compute-app pid
2. 抢 `_lockbench_20260913/locks/gpu<N>.lock` 的 flock（**跨项目**串行，不只是本项目内部）
3. 跑前重申 `nvidia-smi -lgc 1830,1830`（vBIOS 可锁最高频就是 1830）
4. 设 `CUDA_VISIBLE_DEVICES` 后运行，退出码透传

用法：`tools/gpurun_dg.sh <target_id> <cmd...>`。本项目一切 GPU 运行必须走它。
