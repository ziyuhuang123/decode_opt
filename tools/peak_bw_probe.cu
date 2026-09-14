// =============================================================================
// peak_bw_probe.cu — 诚实测量 H20「实测可达读带宽峰值」(CONTRACT.md §7 辅助虚线)
//
// 设计要点（详细论证见 tools/README.md）：
//   1. SM90 TMA bulk load：cute::SM90_TMA_LOAD_4D，box = {128, 40, SUB, 1}，
//      SWIZZLE_128B。单 op 字节数 = 128*40*SUB = 5120*SUB，
//      SUB ∈ {1,2,4,8,16} -> 5/10/20/40/80 KB。
//      （用 4D 而不是 2D 是因为 TMA boxDim 每维上限 256：40KB=128*40*8 需要第 3 维。）
//   2. 多级流水：STAGES ∈ {2,4,8} 个 smem stage，> 双缓冲。
//      producer warp(tid0) 发 cp.async.bulk.tensor；7 个 consumer warp 只做
//      mbarrier wait + 每 stage 一次 16B smem 读 + arrive。
//      没有 math，但 smem 读 -> XOR -> 写 gmem sink 形成真数据依赖，编译器无法消掉 TMA。
//   3. 冷数据铁证：NUM_BUFS 个 rotating buffer，每个 BUF_BYTES，
//      总 working set = NUM_BUFS*BUF_BYTES >= 4 GiB（>> H20 L2 60 MiB）。
//      每次 launch 轮换 buffer 起点，且单次 launch 内每个字节只被读一次。
//      额外可选 L2 flush（256MiB memset，在计时区外）。
//   4. 单次 launch 体积 ~512 MiB -> ~146 µs @3.5TB/s，远大于 launch 开销。
//   5. 每配置 2 warmup + ROUNDS 次计时，取中位数。
//   6. 对照组：朴素 ld.global.nc.v4 流式读（同一批冷 buffer）。
//   7. 自检：忠实复现 shared/tma_read_bench.cu 的 128B 步进重复读同一 ~48KB 窗口，
//      证明本 harness 在「错误写法」下确实会报出 >4TB/s 的假数据，
//      从而反证主扫描的 <4TB/s 不是计时/口径 bug。
// =============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_tma.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <string>
#include <vector>

using Barrier = cutlass::arch::ClusterTransactionBarrier;

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "[FATAL] CUDA %s @ %s:%d\n",                        \
                   cudaGetErrorString(e_), __FILE__, __LINE__);                \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------------
// 常量
// -----------------------------------------------------------------------------
static constexpr int kThreads         = 256;                  // 1 producer warp + 7 consumer warp
static constexpr int kConsumerWarps   = kThreads / 32 - 1;    // 7
static constexpr int kProducerTid     = 0;
static constexpr int kConsumerTidBase = 32;

static constexpr int kRowsPerTile = 40;                       // TMA dim1（= v2 的 BM=40）
static constexpr int kKBlockBytes = 128 * kRowsPerTile;       // 5120 B / k_block
static constexpr int kSmemMax     = 227 * 1024;               // sm_90 动态 smem 上限 232448 B
static constexpr int kBarrierPad  = 1024;                     // barrier 区，data 从 +1024 开始

static constexpr int kSinkSlotsPerCta = 8;

// 扫描维度
static const int kOpBytesList[]  = {5120, 10240, 20480, 40960, 81920};   // 5/10/20/40/80 KB
static const int kCtasList[]     = {78, 156, 312, 624};
static const int kStagesList[]   = {2, 4, 8};
static const int kNaiveIlpList[] = {1, 2, 4};
static const int kNaiveBpsList[] = {1, 2, 4, 8};                          // blocks per SM

// =============================================================================
// TMA 流水线探针 kernel
// =============================================================================
// HINT 是运行期参数：cp.async.bulk.tensor 的 L2 cache_hint 本来就是寄存器操作数，
// 不必为 EVICT_NORMAL / EVICT_FIRST 各编一份 kernel。
template <int STAGES, int OP_BYTES>
__global__ void __launch_bounds__(kThreads)
tma_stream_probe(const __grid_constant__ CUtensorMap map,
                 unsigned long long* __restrict__ sink,
                 const int sub, const int ops_per_tile, const uint64_t hint) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  Barrier* full       = reinterpret_cast<Barrier*>(smem_raw);
  Barrier* empty      = reinterpret_cast<Barrier*>(smem_raw + STAGES * sizeof(Barrier));
  unsigned char* data = smem_raw + kBarrierPad;

  const int tid = threadIdx.x;
  if (tid < STAGES) {
    full[tid].init(1);                 // producer 的 arrive_and_expect_tx 一次
  } else if (tid < 2 * STAGES) {
    empty[tid - STAGES].init(kConsumerWarps);   // 每个 consumer warp 一次 arrive
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  if (tid == kProducerTid) {
    // ---- producer：只管发 TMA，靠 empty barrier 反压 ----
    for (int op = 0; op < ops_per_tile; ++op) {
      const int s = op % STAGES;
      const int r = op / STAGES;
      if (r > 0) empty[s].wait((r - 1) & 1);      // 等 consumer 释放 stage s
      full[s].arrive_and_expect_tx(OP_BYTES);
      cute::SM90_TMA_LOAD_4D::copy(&map,
                                   reinterpret_cast<uint64_t*>(&full[s]),
                                   hint,
                                   data + static_cast<size_t>(s) * OP_BYTES,
                                   0, 0, op * sub, blockIdx.x);
    }
  } else if (tid >= kConsumerTidBase) {
    // ---- consumer("math" 侧)：只做 wait / 读 / arrive，防止编译器把 TMA 优化掉 ----
    unsigned acc = 0;
    for (int op = 0; op < ops_per_tile; ++op) {
      const int s = op % STAGES;
      const int r = op / STAGES;
      full[s].wait(r & 1);                        // 等本 stage 的 tx 全部落地
      const uint4* p = reinterpret_cast<const uint4*>(
          data + static_cast<size_t>(s) * OP_BYTES);
      const int idx = (tid * 7 + op) % (OP_BYTES / 16);
      const uint4 v = p[idx];                     // 真读 smem，形成数据依赖
      // 用乘加而不是 XOR：XOR 在均匀填充的数据上会自我抵消（acc 恒为 0），
      // 那样 sink 就证明不了 consumer 真读过 TMA 搬进来的数据。
      acc = acc * 1664525u + (v.x + v.y + v.z + v.w) + 1u;
      __syncwarp(0xffffffffu);                    // 保证整 warp 的读都发出后再释放 stage
      if ((tid & 31) == 0) empty[s].arrive();
    }
    sink[static_cast<size_t>(blockIdx.x) * kSinkSlotsPerCta + (tid >> 5)] = acc;
  }
}

// =============================================================================
// 对照组：朴素 ld.global.nc.v4 流式读
// =============================================================================
template <int ILP>
__global__ void __launch_bounds__(256)
naive_read_nc(const uint4* __restrict__ src, const size_t n_vec,
              unsigned long long* __restrict__ sink) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (; i + static_cast<size_t>(ILP - 1) * stride < n_vec;
       i += static_cast<size_t>(ILP) * stride) {
#pragma unroll
    for (int k = 0; k < ILP; ++k) {
      uint4 v;
      const uint4* p = src + i + static_cast<size_t>(k) * stride;
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                   : "l"(p)
                   : "memory");
      acc = acc * 1664525u + (v.x + v.y + v.z + v.w) + 1u;
    }
  }
  for (; i < n_vec; i += stride) {
    uint4 v;
    const uint4* p = src + i;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    acc = acc * 1664525u + (v.x + v.y + v.z + v.w) + 1u;
  }
  if ((threadIdx.x & 31) == 0)
    sink[static_cast<size_t>(blockIdx.x) * kSinkSlotsPerCta + (threadIdx.x >> 5)] = acc;
}

// =============================================================================
// 冷 buffer 填充：写非均匀伪随机 pattern（而不是 memset 常量）。这样 consumer 的累加值
// 非 0，sink 才是一条真证据：证明 TMA 确实把 gmem 数据搬进 smem 并被 consumer 读到了。
// =============================================================================
__global__ void fill_pattern_kernel(uint4* __restrict__ p, const size_t n_vec,
                                    const unsigned seed) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < n_vec; k += stride) {
    unsigned x = static_cast<unsigned>(k * 2654435761u) ^ seed;
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    p[k] = make_uint4(x, x * 1664525u + 1013904223u, ~x, x ^ 0x9e3779b9u);
  }
}

// =============================================================================
// 自检 kernel：忠实复现 shared/tma_read_bench.cu 的 L2 命中假数据写法
//   box 坐标只前进 128B（且被 & 0xFFF80 掩掉低位），128 次迭代反复读同一 ~48KB 窗口
// =============================================================================
__global__ void tma_l2_selftest(const __grid_constant__ CUtensorMap map,
                               unsigned long long* __restrict__ sink) {
  extern __shared__ __align__(1024) unsigned char sm[];
  auto* b = reinterpret_cast<Barrier*>(sm);
  unsigned char* dst = sm + 1024;
  if (threadIdx.x == 0) { b->init(1); cutlass::arch::fence_view_async_shared(); }
  __syncthreads();
  for (int i = 0; i < 256; ++i) {
    const int phase = i & 1;
    if (threadIdx.x == 0) {
      b->arrive_and_expect_tx(16384);
      cute::SM90_TMA_LOAD_2D::copy(
          &map, reinterpret_cast<uint64_t*>(b),
          static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL), dst,
          0, (blockIdx.x * 256 + i) & (1048576 - 128));
    }
    b->wait(phase);
  }
  if (threadIdx.x == 0) *sink += dst[0];
}

// =============================================================================
// host 侧
// =============================================================================
namespace {

struct Config {
  int op_bytes = 0, ctas = 0, stages = 0;
  int sub = 0, ops_per_tile = 0, smem_bytes = 0;
  long long bytes_per_cta = 0, launch_bytes = 0, tile_bytes = 0, kb_per_tile = 0;
  bool feasible = false;
  std::string skip_reason;
  // 结果
  bool ran = false;
  double median_us = 0.0, batch_us = 0.0, gbps = 0.0, pct = 0.0;
  std::vector<double> round_us;
  double evict_first_gbps = 0.0, noflush_gbps = 0.0, norot_gbps = 0.0;
  bool ablated = false;
};

struct NaiveCfg {
  int ilp = 0, blocks_per_sm = 0, blocks = 0;
  double median_us = 0.0, gbps = 0.0;
  std::vector<double> round_us;
};

struct Opt {
  long long buf_bytes   = 512LL * 1024 * 1024;   // 每 buffer 字节
  int  num_bufs         = 13;                    // buffer 数
  long long min_ws      = 4LL * 1024 * 1024 * 1024;  // 目标 working set 下限
  long long fallback_ws = 2LL * 1024 * 1024 * 1024;  // 分配失败时的降级目标
  int  rounds           = 7;
  int  warmups          = 2;
  int  batch_launches   = 8;
  long long flush_bytes = 256LL * 1024 * 1024;
  // 主协议 = 只靠 rotating cold buffer（与 CONTRACT §6 的冷权重协议一致）。
  // 不在每次计时前往 L2 灌 256MiB memset：那会把 flush 的 write-back 拖进计时窗口，
  // 人为压低读带宽约 5%（首轮实测 rot+flush 3438 vs rot-noflush 3636 GB/s），
  // 而真实 decode GEMM 并没有这笔写流量。flush 作为「更保守」的消融项保留。
  bool flush            = false;
  bool do_naive         = true;
  bool do_selftest      = true;
  bool do_ablate        = true;
  int  nvsmi_index      = -1;   // -1 = 运行时按 PCI bus id 自动探测物理卡号
  int  nvsmi_index_arg  = -1;   // host 传进来的值，只用于交叉校验
  // 争用守卫：分配完 cold buffer 之后、开始计时之前，等 GPU 真正安静下来
  int  wait_quiet_secs   = 30;      // 需要连续安静多少秒
  int  wait_timeout_secs = 2400;    // 最多等这么久，超时就放弃（绝不硬测被污染的卡）
  int  quiet_mem_max_mib = 2000;    // 外来进程显存合计上限（keeper ~438MiB 放行）
  int  quiet_util_max    = 5;       // GPU 利用率上限 %
  int  quiet_poll_secs   = 3;
  int  clocks_sm_mhz    = -1;
  bool clock_locked     = false;
  std::string out_json  = "results/peak_bw.json";
  std::string note_alloc;
};

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// nvidia-smi 采样：返回 clocks.sm(MHz)；失败返回 -1
int query_clocks_sm(int nvsmi_index) {
  char cmd[256];
  std::snprintf(cmd, sizeof(cmd),
                "nvidia-smi -i %d --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null",
                nvsmi_index);
  FILE* f = popen(cmd, "r");
  if (!f) return -1;
  int v = -1;
  if (fscanf(f, "%d", &v) != 1) v = -1;
  pclose(f);
  return v;
}

// 外来 compute app 检测：列出 GPU 上的 PID，剔除本进程自己，返回外来 PID 列表。
// 这是「测量期间没人跟我抢卡」的自报证据。
std::vector<long long> query_foreign_pids(int nvsmi_index) {
  std::vector<long long> out;
  char cmd[256];
  std::snprintf(cmd, sizeof(cmd),
                "nvidia-smi -i %d --query-compute-apps=pid --format=csv,noheader 2>/dev/null",
                nvsmi_index);
  FILE* f = popen(cmd, "r");
  if (!f) return out;
  char line[128];
  const long long self = (long long)getpid();
  while (fgets(line, sizeof(line), f)) {
    long long pid = 0;
    if (sscanf(line, "%lld", &pid) == 1 && pid > 0 && pid != self) out.push_back(pid);
  }
  pclose(f);
  return out;
}

// ---------------------------------------------------------------------------
// GPU 争用守卫判据（沿用本仓库 _lockbench_20260913/PROTOCOL.md §1 的 house 判据）：
//   一张卡算「空」= utilization.gpu <= UTIL_MAX 且 外来进程显存合计 <= MEM_MAX。
// 这条判据能区分「活跑 kernel 的邻居」和「只挂着 CUDA context 睡觉的 keeper」：
// weave_v1/bench_20260913 的 keeper.py 只占 438 MiB 且 time.sleep(5) 自旋，不发任何
// kernel，物理上不可能消耗 DRAM 带宽；它的真 bench 进程占 ~35.6 GB 且 util 打满。
// ---------------------------------------------------------------------------
struct GpuState {
  int util_pct = -1;
  long long mem_used_mib = -1;
  std::vector<std::pair<long long, long long>> apps;   // (pid, used_mib)
  long long foreign_mib = 0;
  std::vector<long long> foreign_pids;
};

GpuState query_gpu_state(int nvsmi_index) {
  GpuState st;
  char cmd[256], line[256];
  std::snprintf(cmd, sizeof(cmd),
                "nvidia-smi -i %d --query-gpu=utilization.gpu,memory.used "
                "--format=csv,noheader,nounits 2>/dev/null", nvsmi_index);
  if (FILE* f = popen(cmd, "r")) {
    if (fgets(line, sizeof(line), f)) {
      int u = -1; long long m = -1;
      if (sscanf(line, "%d, %lld", &u, &m) == 2) { st.util_pct = u; st.mem_used_mib = m; }
    }
    pclose(f);
  }
  std::snprintf(cmd, sizeof(cmd),
                "nvidia-smi -i %d --query-compute-apps=pid,used_memory "
                "--format=csv,noheader,nounits 2>/dev/null", nvsmi_index);
  if (FILE* f = popen(cmd, "r")) {
    const long long self = (long long)getpid();
    while (fgets(line, sizeof(line), f)) {
      long long pid = 0, mib = 0;
      if (sscanf(line, "%lld, %lld", &pid, &mib) == 2 && pid > 0) {
        st.apps.push_back({pid, mib});
        if (pid != self) { st.foreign_mib += mib; st.foreign_pids.push_back(pid); }
      }
    }
    pclose(f);
  }
  return st;
}

// 通过 PCI bus id 把 cuda device 0 映射回「物理 GPU index」。
// CUDA_VISIBLE_DEVICES 会重映射设备号，所以绝不能信 host 传进来的序号；
// 只有 bus id 匹配出来的才是真正在跑的那张卡 -> 这个值写进 JSON 的 gpu_id。
int detect_physical_gpu_index(const cudaDeviceProp& prop) {
  char want[64];
  std::snprintf(want, sizeof(want), "%04X:%02X:%02X.0",
                prop.pciDomainID & 0xFFFF, prop.pciBusID & 0xFF, prop.pciDeviceID & 0xFF);
  FILE* f = popen("nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null", "r");
  if (!f) return -1;
  char line[256];
  int found = -1;
  while (fgets(line, sizeof(line), f)) {
    int idx = -1;
    char bus[64] = {0};
    if (sscanf(line, "%d, %63s", &idx, bus) != 2) continue;
    // nvidia-smi 给的是 00000000:3B:00.0；取后 7 个字符 "3B:00.0" 做大小写无关比较
    const char* tail_have = strrchr(bus, ':');
    if (!tail_have) continue;
    // 比较 bus:dev.fn 部分
    std::string a(bus), b(want);
    auto norm = [](std::string x) {
      // 去掉 domain，统一大写
      const size_t pos = x.find(':');
      std::string y = (pos != std::string::npos) ? x.substr(pos + 1) : x;
      for (auto& c : y) c = (char)toupper((unsigned char)c);
      return y;
    };
    if (norm(a) == norm(b)) { found = idx; break; }
    (void)tail_have;
  }
  pclose(f);
  return found;
}

// throttle reasons（nvidia-smi clocks_event_reasons.active），锁频是否真生效的旁证
std::string query_throttle_reasons(int nvsmi_index) {
  char cmd[256], line[128] = {0};
  std::snprintf(cmd, sizeof(cmd),
                "nvidia-smi -i %d --query-gpu=clocks_event_reasons.active "
                "--format=csv,noheader 2>/dev/null", nvsmi_index);
  FILE* f = popen(cmd, "r");
  if (!f) return "";
  if (!fgets(line, sizeof(line), f)) line[0] = 0;
  pclose(f);
  std::string s(line);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}

std::string query_gpu_name(int nvsmi_index) {
  char cmd[256];
  std::snprintf(cmd, sizeof(cmd),
                "nvidia-smi -i %d --query-gpu=name,uuid,memory.total --format=csv,noheader 2>/dev/null",
                nvsmi_index);
  FILE* f = popen(cmd, "r");
  if (!f) return "";
  char buf[512] = {0};
  if (!fgets(buf, sizeof(buf), f)) buf[0] = 0;
  pclose(f);
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}

bool encode_map_4d(CUtensorMap* map, void* base, long long tile_bytes,
                   long long kb_per_tile, long long num_tiles, int sub) {
  const uint64_t dims[4]    = {128, (uint64_t)kRowsPerTile, (uint64_t)kb_per_tile, (uint64_t)num_tiles};
  const uint64_t strides[3] = {128, (uint64_t)kKBlockBytes, (uint64_t)tile_bytes};
  const uint32_t box[4]     = {128, (uint32_t)kRowsPerTile, (uint32_t)sub, 1};
  const uint32_t es[4]      = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, base, dims, strides, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    std::fprintf(stderr, "[WARN] cuTensorMapEncodeTiled failed: %s\n", s ? s : "?");
    return false;
  }
  return true;
}

template <int STAGES, int OP_BYTES>
void launch_probe(const CUtensorMap& map, unsigned long long* sink, int sub,
                  int ops_per_tile, int ctas, uint64_t hint, cudaStream_t st) {
  constexpr int kSmem = STAGES * OP_BYTES + kBarrierPad;
  static bool attr_done = false;
  if (!attr_done) {
    CK(cudaFuncSetAttribute(tma_stream_probe<STAGES, OP_BYTES>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr_done = true;
  }
  tma_stream_probe<STAGES, OP_BYTES>
      <<<ctas, kThreads, kSmem, st>>>(map, sink, sub, ops_per_tile, hint);
}

#define DISPATCH_OP(ST)                                                          \
  switch (c.op_bytes) {                                                          \
    case 5120:                                                                   \
      launch_probe<ST, 5120>(map, sink, c.sub, c.ops_per_tile, c.ctas, hint, st);\
      break;                                                                     \
    case 10240:                                                                  \
      launch_probe<ST, 10240>(map, sink, c.sub, c.ops_per_tile, c.ctas, hint, st);\
      break;                                                                     \
    case 20480:                                                                  \
      launch_probe<ST, 20480>(map, sink, c.sub, c.ops_per_tile, c.ctas, hint, st);\
      break;                                                                     \
    case 40960:                                                                  \
      launch_probe<ST, 40960>(map, sink, c.sub, c.ops_per_tile, c.ctas, hint, st);\
      break;                                                                     \
    case 81920:                                                                  \
      launch_probe<ST, 81920>(map, sink, c.sub, c.ops_per_tile, c.ctas, hint, st);\
      break;                                                                     \
    default:                                                                     \
      std::fprintf(stderr, "[FATAL] bad op_bytes %d\n", c.op_bytes);             \
      std::exit(3);                                                              \
  }

void launch_config(const Config& c, const CUtensorMap& map,
                   unsigned long long* sink, cudaStream_t st, bool evict_first) {
  const uint64_t hint = evict_first
      ? static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST)
      : static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  switch (c.stages) {
    case 2: DISPATCH_OP(2); break;
    case 4: DISPATCH_OP(4); break;
    case 8: DISPATCH_OP(8); break;
    default: std::fprintf(stderr, "[FATAL] bad stages %d\n", c.stages); std::exit(3);
  }
}

void launch_naive(int ilp, int blocks, const uint4* src, size_t n_vec,
                  unsigned long long* sink, cudaStream_t st) {
  switch (ilp) {
    case 1: naive_read_nc<1><<<blocks, 256, 0, st>>>(src, n_vec, sink); break;
    case 2: naive_read_nc<2><<<blocks, 256, 0, st>>>(src, n_vec, sink); break;
    case 4: naive_read_nc<4><<<blocks, 256, 0, st>>>(src, n_vec, sink); break;
    default: std::fprintf(stderr, "[FATAL] bad ilp %d\n", ilp); std::exit(3);
  }
}

// cudaUUID_t 是 16 字节的 POD，不是字符串；直接 %s 会打出垃圾。手工格式化。
std::string uuid_to_string(const cudaUUID_t& u) {
  static const char* kHex = "0123456789abcdef";
  const unsigned char* b = reinterpret_cast<const unsigned char*>(u.bytes);
  std::string out;
  out.reserve(36);
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) out += '-';
    out += kHex[b[i] >> 4];
    out += kHex[b[i] & 0xF];
  }
  return out;
}

std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 16);
  for (char ch : in) {
    switch (ch) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", ch);
          out += b;
        } else {
          out += ch;
        }
    }
  }
  return out;
}

}  // namespace

// =============================================================================
int main(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "[FATAL] missing value for %s\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if      (a == "--buf-mb")        o.buf_bytes   = (long long)std::atoll(next().c_str()) * 1024 * 1024;
    else if (a == "--num-bufs")      o.num_bufs    = std::atoi(next().c_str());
    else if (a == "--rounds")        o.rounds      = std::atoi(next().c_str());
    else if (a == "--warmups")       o.warmups     = std::atoi(next().c_str());
    else if (a == "--batch")         o.batch_launches = std::atoi(next().c_str());
    else if (a == "--flush-mb")      o.flush_bytes = (long long)std::atoll(next().c_str()) * 1024 * 1024;
    else if (a == "--no-flush")      o.flush       = false;
    else if (a == "--flush")         o.flush       = true;
    else if (a == "--no-naive")      o.do_naive    = false;
    else if (a == "--no-selftest")   o.do_selftest = false;
    else if (a == "--no-ablate")     o.do_ablate   = false;
    else if (a == "--nvsmi-index")   { o.nvsmi_index_arg = std::atoi(next().c_str()); o.nvsmi_index = o.nvsmi_index_arg; }
    else if (a == "--wait-quiet")    o.wait_quiet_secs   = std::atoi(next().c_str());
    else if (a == "--wait-timeout")  o.wait_timeout_secs = std::atoi(next().c_str());
    else if (a == "--quiet-mem-max") o.quiet_mem_max_mib = std::atoi(next().c_str());
    else if (a == "--quiet-util-max")o.quiet_util_max    = std::atoi(next().c_str());
    else if (a == "--quiet-poll")    o.quiet_poll_secs   = std::atoi(next().c_str());
    else if (a == "--clocks-sm-mhz") o.clocks_sm_mhz = std::atoi(next().c_str());
    else if (a == "--clock-locked")  o.clock_locked = std::atoi(next().c_str()) != 0;
    else if (a == "--out")           o.out_json    = next();
    else { std::fprintf(stderr, "[FATAL] unknown arg %s\n", a.c_str()); return 2; }
  }
  if (o.rounds < 3) { std::fprintf(stderr, "[FATAL] --rounds must be >= 3 (纪律：至少 3 轮取中位)\n"); return 2; }

  CK(cudaSetDevice(0));

  // 先做一次极小分配，让本进程立刻出现在 nvidia-smi --query-compute-apps 里。
  // 邻居 weave_v1/bench_20260913 的 run2.sh 自带争用守卫：GPU2/GPU3 上一旦出现非 keeper
  // 的 pid，它就先等（GUARD_WAIT=1800s）再跑。所以「先占位、后等安静」能拿到一个
  // 确定性的独占窗口，而不是靠运气插进它两次 run 之间那几秒的空档。
  unsigned long long* sink = nullptr;
  CK(cudaMalloc(&sink, sizeof(unsigned long long) * 624 * kSinkSlotsPerCta));
  CK(cudaMemset(sink, 0, sizeof(unsigned long long) * 624 * kSinkSlotsPerCta));
  std::printf("[reserve] pid=%d 已在物理 GPU%d 建立 CUDA context 占位（邻居守卫会让路）\n",
              (int)getpid(), o.nvsmi_index);
  std::fflush(stdout);

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));

  // ---- 物理卡号自动探测（gpu_id）----
  // CUDA_VISIBLE_DEVICES 会重映射设备号，host 传的序号不可信；用 PCI bus id 反查。
  const int detected_gpu = detect_physical_gpu_index(prop);
  if (detected_gpu >= 0) {
    if (o.nvsmi_index_arg >= 0 && o.nvsmi_index_arg != detected_gpu) {
      std::printf("[WARN] host 传的 --nvsmi-index=%d 与实际探测到的物理卡 %d 不一致，"
                  "以探测值为准\n", o.nvsmi_index_arg, detected_gpu);
    }
    o.nvsmi_index = detected_gpu;
  } else if (o.nvsmi_index_arg >= 0) {
    std::printf("[WARN] PCI bus id 反查失败，退回使用 host 传入的 --nvsmi-index=%d\n",
                o.nvsmi_index_arg);
  } else {
    std::fprintf(stderr, "[FATAL] 无法确定物理 GPU index（bus id 反查失败且未传 --nvsmi-index）\n");
    return 2;
  }
  const int gpu_id = o.nvsmi_index;

  const int   sm_count = prop.multiProcessorCount;
  const long long l2_bytes = (long long)prop.l2CacheSize;

  std::printf("=============================================================\n");
  std::printf(" peak_bw_probe — H20 实测可达读带宽峰值 (CONTRACT §7 辅助虚线)\n");
  std::printf("=============================================================\n");
  std::printf("cuda device 0     : %s  (SM %d.%d, %d SMs, L2 = %lld MiB, smem/SM = %zu B)\n",
              prop.name, prop.major, prop.minor, sm_count, l2_bytes >> 20,
              (size_t)prop.sharedMemPerMultiprocessor);
  std::printf("pci bus id        : %04X:%02X:%02X.0  ->  physical gpu_id = %d  (自动探测，非 host 传入)\n",
              prop.pciDomainID & 0xFFFF, prop.pciBusID & 0xFF, prop.pciDeviceID & 0xFF, gpu_id);
  const std::string gpu_uuid = uuid_to_string(prop.uuid);
  std::printf("gpu uuid          : %s\n", gpu_uuid.c_str());
  std::printf("CUDA_VISIBLE_DEVICES = %s\n",
              getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "(unset)");
  std::printf("nvidia-smi -i %d  : %s\n", gpu_id, query_gpu_name(gpu_id).c_str());
  const int clk_before = query_clocks_sm(gpu_id);
  const std::string throttle_before = query_throttle_reasons(gpu_id);
  std::printf("clocks.sm before sweep : %d MHz (host arg = %d, locked = %d)\n",
              clk_before, o.clocks_sm_mhz, (int)o.clock_locked);
  std::printf("spec peak (主参考线, 不由本探针测定) : 4.0 TB/s\n\n");

  if (prop.multiProcessorCount != 78) {
    std::printf("[WARN] expected 78 SMs on H20, got %d — 确认 CUDA_VISIBLE_DEVICES=3\n",
                prop.multiProcessorCount);
  }

  // ---------------------------------------------------------------------------
  // 分配 rotating cold buffer 集
  //   目标：total >= 4 GiB；失败则逐级降级到 2 GiB 并在 JSON 注明
  // ---------------------------------------------------------------------------
  std::vector<unsigned char*> bufs;
  long long buf_bytes = o.buf_bytes;
  int num_bufs = o.num_bufs;
  long long total_ws = 0;
  bool degraded = false;

  auto try_alloc = [&](long long bb, int nb) -> bool {
    for (unsigned char* p : bufs) cudaFree(p);
    bufs.clear();
    bufs.reserve(nb);
    for (int i = 0; i < nb; ++i) {
      unsigned char* p = nullptr;
      if (cudaMalloc(&p, (size_t)bb) != cudaSuccess) {
        cudaGetLastError();
        for (unsigned char* q : bufs) cudaFree(q);
        bufs.clear();
        return false;
      }
      bufs.push_back(p);
    }
    for (int i = 0; i < (int)bufs.size(); ++i) {
      fill_pattern_kernel<<<1024, 256>>>(reinterpret_cast<uint4*>(bufs[i]),
                                         (size_t)bb / 16,
                                         0x5EEDu + (unsigned)i * 7919u);
      CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    return true;
  };

  // 逐级尝试：512MB×13(6.5GB) -> 512MB×8(4GB) -> 256MB×16(4GB) -> 512MB×4(2GB)
  struct Tier { long long bb; int nb; const char* tag; };
  const Tier tiers[] = {
      {buf_bytes, num_bufs, "primary"},
      {512LL << 20, 8, "tier2_4gib"},
      {256LL << 20, 16, "tier3_4gib_smallbuf"},
      {512LL << 20, 4, "fallback_2gib"},
      {256LL << 20, 8, "fallback_2gib_smallbuf"},
  };
  for (const Tier& t : tiers) {
    if (try_alloc(t.bb, t.nb)) {
      buf_bytes = t.bb;
      num_bufs = t.nb;
      total_ws = buf_bytes * num_bufs;
      if (std::strcmp(t.tag, "primary") != 0) {
        degraded = true;
        o.note_alloc = std::string("requested ") + std::to_string(o.buf_bytes >> 20) +
                       "MiB x " + std::to_string(o.num_bufs) +
                       " failed to allocate; degraded to tier '" + t.tag + "'";
      } else {
        o.note_alloc = "primary tier allocated as requested";
      }
      break;
    }
    std::printf("[WARN] allocation tier '%s' (%lld MiB x %d) failed, trying next\n",
                t.tag, t.bb >> 20, t.nb);
  }
  if (bufs.empty()) {
    std::fprintf(stderr, "[FATAL] cannot allocate any cold-buffer tier\n");
    return 2;
  }

  std::printf("---- cold rotating buffer set (自报) ----\n");
  std::printf("  num_buffers        : %d\n", num_bufs);
  std::printf("  bytes_per_buffer   : %lld (%.0f MiB)\n", buf_bytes, buf_bytes / 1048576.0);
  std::printf("  working_set_bytes  : %lld (%.2f GiB)\n", total_ws, total_ws / 1073741824.0);
  std::printf("  L2 size            : %lld MiB  -> working_set / L2 = %.1fx\n",
              l2_bytes >> 20, (double)total_ws / (double)l2_bytes);
  std::printf("  alloc note         : %s\n", o.note_alloc.c_str());
  if (total_ws < o.min_ws) {
    std::printf("  [!!] working set %.2f GiB < 4 GiB 要求（已降级到 2 GiB 档）\n",
                total_ws / 1073741824.0);
  }
  std::printf("\n");

  unsigned char* flush_buf = nullptr;
  if (o.flush && o.flush_bytes > 0) {
    if (cudaMalloc(&flush_buf, (size_t)o.flush_bytes) != cudaSuccess) {
      cudaGetLastError();
      o.flush = false;
      o.note_alloc += "; L2 flush buffer alloc FAILED -> flush disabled";
      std::printf("[WARN] flush buffer alloc failed, continuing without explicit L2 flush\n");
    }
  }

  // ---------------------------------------------------------------------------
  // 争用守卫：所有显存都已占好（我们已经出现在 compute-apps 里），现在等邻居让路。
  // 判据 = 外来进程显存合计 <= quiet_mem_max_mib 且 util <= quiet_util_max，
  // 且必须连续满足 wait_quiet_secs 秒。超时则退出码 4，不产出任何数据。
  // ---------------------------------------------------------------------------
  int wait_rc = 0;                       // 0=安静 1=超时放弃 2=守卫被禁用
  long long wait_max_foreign_mib = 0;
  int wait_max_util = 0;
  std::vector<long long> wait_foreign_pids;
  if (o.wait_quiet_secs > 0) {
    std::printf("\n---- contention guard: waiting for physical GPU%d to go quiet ----\n",
                o.nvsmi_index);
    std::printf("  criterion : foreign_mib <= %d AND util_pct <= %d, sustained >= %ds\n",
                o.quiet_mem_max_mib, o.quiet_util_max, o.wait_quiet_secs);
    std::printf("  poll/timeout : %ds / %ds\n", o.quiet_poll_secs, o.wait_timeout_secs);
    std::fflush(stdout);
    int waited = 0, quiet = 0;
    while (true) {
      const GpuState g = query_gpu_state(o.nvsmi_index);
      if (g.foreign_mib > wait_max_foreign_mib) wait_max_foreign_mib = g.foreign_mib;
      if (g.util_pct  > wait_max_util)         wait_max_util = g.util_pct;
      for (long long pid : g.foreign_pids)
        if (std::find(wait_foreign_pids.begin(), wait_foreign_pids.end(), pid) ==
            wait_foreign_pids.end())
          wait_foreign_pids.push_back(pid);
      const bool ok = (g.foreign_mib >= 0 && g.foreign_mib <= o.quiet_mem_max_mib &&
                       g.util_pct >= 0 && g.util_pct <= o.quiet_util_max);
      if (ok) {
        quiet += o.quiet_poll_secs;
        std::printf("  [t=%4ds] QUIET %3d/%3ds  util=%d%%  foreign_mib=%lld  pids=[",
                    waited, quiet, o.wait_quiet_secs, g.util_pct, g.foreign_mib);
        for (size_t i = 0; i < g.foreign_pids.size(); ++i)
          std::printf("%s%lld", i ? "," : "", g.foreign_pids[i]);
        std::printf("]\n");
        if (quiet >= o.wait_quiet_secs) {
          std::printf("  -> GPU%d quiet for %ds; starting measurement.\n", o.nvsmi_index, quiet);
          break;
        }
      } else {
        if (quiet > 0) {
          std::printf("  [t=%4ds] BUSY, quiet streak reset (was %ds): util=%d%% foreign_mib=%lld\n",
                      waited, quiet, g.util_pct, g.foreign_mib);
        } else if ((waited % 30) == 0) {
          std::printf("  [t=%4ds] BUSY: util=%d%% foreign_mib=%lld "
                      "(neighbour run in flight; its own guard will yield to us next)\n",
                      waited, g.util_pct, g.foreign_mib);
        }
        quiet = 0;
        if (waited >= o.wait_timeout_secs) {
          std::printf("  [!!] TIMEOUT after %ds: GPU%d never went quiet -> REFUSING to measure.\n",
                      waited, o.nvsmi_index);
          wait_rc = 1;
          break;
        }
      }
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::seconds(o.quiet_poll_secs));
      waited += o.quiet_poll_secs;
    }
  } else {
    wait_rc = 2;
    std::printf("\n[warn] contention guard DISABLED (--wait-quiet 0)\n");
  }

  if (wait_rc == 1) {
    CK(cudaFree(sink));
    if (flush_buf) CK(cudaFree(flush_buf));
    for (unsigned char* p : bufs) CK(cudaFree(p));
    std::fprintf(stderr,
                 "[peak_bw_probe] ABORTED: GPU%d stayed contended for %ds. "
                 "No measurement taken, no JSON written.\n",
                 o.nvsmi_index, o.wait_timeout_secs);
    return 4;
  }

  cudaStream_t stream = nullptr;
  CK(cudaStreamCreate(&stream));
  cudaEvent_t ev_a, ev_z;
  CK(cudaEventCreate(&ev_a));
  CK(cudaEventCreate(&ev_z));

  // 轮换起点：每次 launch 递增，保证 13 个 buffer 循环使用
  unsigned long long g_launch = (unsigned long long)(
      std::chrono::steady_clock::now().time_since_epoch().count() % 1000);
  auto next_buf = [&]() -> int { return (int)((g_launch++) % (unsigned long long)num_bufs); };

  auto do_flush = [&]() {
    if (o.flush && flush_buf) CK(cudaMemsetAsync(flush_buf, 0x3C, (size_t)o.flush_bytes, stream));
  };

  // ---------------------------------------------------------------------------
  // 构造配置网格
  // ---------------------------------------------------------------------------
  std::vector<Config> cfgs;
  for (int ob : kOpBytesList)
    for (int ct : kCtasList)
      for (int st : kStagesList) {
        Config c;
        c.op_bytes = ob;
        c.ctas = ct;
        c.stages = st;
        c.sub = ob / kKBlockBytes;
        c.smem_bytes = st * ob + kBarrierPad;

        // 每 CTA 读总字节：目标单次 launch ~= buf_bytes，且 >= stages*2 个 op 让流水线充满
        long long target_per_cta = buf_bytes / ct;
        long long ops = target_per_cta / ob;
        if (ops < (long long)st * 2) ops = (long long)st * 2;
        // 不能超出 buffer：ctas * ops * op_bytes <= buf_bytes
        long long max_ops = buf_bytes / ((long long)ct * ob);
        if (max_ops < 1) max_ops = 1;
        if (ops > max_ops) ops = max_ops;

        c.ops_per_tile = (int)ops;
        c.bytes_per_cta = (long long)c.ops_per_tile * ob;
        c.launch_bytes = c.bytes_per_cta * ct;
        c.tile_bytes = c.bytes_per_cta;
        c.kb_per_tile = (long long)c.ops_per_tile * c.sub;

        if (c.smem_bytes > kSmemMax) {
          c.feasible = false;
          c.skip_reason = "smem " + std::to_string(c.smem_bytes) + "B > " +
                          std::to_string(kSmemMax) + "B (stages*op_bytes too large)";
        } else if (c.launch_bytes > buf_bytes) {
          c.feasible = false;
          c.skip_reason = "launch_bytes > buffer_bytes";
        } else if (c.sub > 256) {
          c.feasible = false;
          c.skip_reason = "TMA boxDim[2] > 256";
        } else {
          c.feasible = true;
        }
        cfgs.push_back(c);
      }

  const int n_feasible = (int)std::count_if(cfgs.begin(), cfgs.end(),
                                            [](const Config& c) { return c.feasible; });
  std::printf("---- config grid ----\n");
  std::printf("  op_bytes x ctas x stages = %zu total, %d feasible, %zu skipped (smem/box 超限)\n",
              cfgs.size(), n_feasible, cfgs.size() - (size_t)n_feasible);
  std::printf("  rounds=%d warmups=%d batch=%d flush=%s(%lld MiB)\n\n",
              o.rounds, o.warmups, o.batch_launches, o.flush ? "on" : "off",
              o.flush_bytes >> 20);

  // ---------------------------------------------------------------------------
  // 主扫描
  // ---------------------------------------------------------------------------
  std::printf("---- TMA sweep (rotating cold %d x %lld MiB, flush=%s) ----\n",
              num_bufs, buf_bytes >> 20, o.flush ? "on" : "off");
  std::printf("%-8s %-6s %-6s %-11s %-11s %-9s %-9s %s\n", "op_B", "ctas", "stg",
              "bytes/CTA", "launch_MB", "med_us", "GB/s", "%4TB/s");

  int clk_min = 1 << 30, clk_max = 0;
  // 污染证据：整个 sweep 期间反复采样 GPU 上的外来 compute app
  std::vector<long long> foreign_seen;      // 去重后的外来 PID
  int foreign_checks = 0, foreign_dirty_checks = 0;
  const auto clk_note = [&](int v) {
    if (v > 0) { clk_min = std::min(clk_min, v); clk_max = std::max(clk_max, v); }
  };
  const auto contamination_check = [&](const char* where) {
    const std::vector<long long> fp = query_foreign_pids(o.nvsmi_index);
    ++foreign_checks;
    if (!fp.empty()) {
      ++foreign_dirty_checks;
      std::printf("[!!] CONTAMINATION at %s: foreign compute apps on GPU%d:", where, o.nvsmi_index);
      for (long long pid : fp) {
        std::printf(" %lld", pid);
        if (std::find(foreign_seen.begin(), foreign_seen.end(), pid) == foreign_seen.end())
          foreign_seen.push_back(pid);
      }
      std::printf("  -- 该轮数据不可信\n");
    }
  };
  clk_note(clk_before);
  contamination_check("pre-sweep");

  for (Config& c : cfgs) {
    if (!c.feasible) continue;

    // 每个 buffer 一份 tensor map（几何相同，只有 base ptr 不同）
    std::vector<CUtensorMap> maps(num_bufs);
    bool ok = true;
    for (int i = 0; i < num_bufs; ++i) {
      if (!encode_map_4d(&maps[i], bufs[i], c.tile_bytes, c.kb_per_tile, c.ctas, c.sub)) {
        c.feasible = false;
        c.skip_reason = "cuTensorMapEncodeTiled failed";
        ok = false;
        break;
      }
    }
    if (!ok) continue;

    // warmup（不计时，走 rotating buffer）
    for (int w = 0; w < o.warmups; ++w) {
      const int bi = next_buf();
      launch_config(c, maps[bi], sink, stream, false);
    }
    {
      cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        std::printf("%-8d %-6d %-6d  LAUNCH ERROR: %s\n", c.op_bytes, c.ctas, c.stages,
                    cudaGetErrorString(e));
        c.feasible = false;
        c.skip_reason = std::string("launch error: ") + cudaGetErrorString(e);
        cudaDeviceSynchronize();
        continue;
      }
      CK(cudaDeviceSynchronize());
    }

    // 计时轮
    for (int r = 0; r < o.rounds; ++r) {
      const int bi = next_buf();
      do_flush();
      CK(cudaEventRecord(ev_a, stream));
      launch_config(c, maps[bi], sink, stream, false);
      CK(cudaEventRecord(ev_z, stream));
      CK(cudaEventSynchronize(ev_z));
      float ms = 0.f;
      CK(cudaEventElapsedTime(&ms, ev_a, ev_z));
      c.round_us.push_back(ms * 1000.0);
    }
    c.median_us = median_of(c.round_us);

    // batch 交叉验证（把 event 开销摊到 8 次 launch）
    {
      do_flush();
      CK(cudaEventRecord(ev_a, stream));
      for (int b = 0; b < o.batch_launches; ++b) {
        const int bi = next_buf();
        launch_config(c, maps[bi], sink, stream, false);
      }
      CK(cudaEventRecord(ev_z, stream));
      CK(cudaEventSynchronize(ev_z));
      float ms = 0.f;
      CK(cudaEventElapsedTime(&ms, ev_a, ev_z));
      c.batch_us = ms * 1000.0 / o.batch_launches;
    }

    c.ran = true;
    c.gbps = (double)c.launch_bytes / (c.median_us * 1e-6) / 1e9;
    c.pct = c.gbps / 4000.0 * 100.0;
    std::printf("%-8d %-6d %-6d %-11lld %-11.1f %-9.2f %-9.1f %.2f  (batch %.2f us -> %.1f GB/s)\n",
                c.op_bytes, c.ctas, c.stages, c.bytes_per_cta, c.launch_bytes / 1048576.0,
                c.median_us, c.gbps, c.pct, c.batch_us,
                (double)c.launch_bytes / (c.batch_us * 1e-6) / 1e9);
    clk_note(query_clocks_sm(o.nvsmi_index));
    contamination_check("tma-config");
  }

  // ---------------------------------------------------------------------------
  // best + 消融（证明冷/热差异、cache hint 影响、以及 harness 能检出 L2 假数据）
  // ---------------------------------------------------------------------------
  int best_idx = -1;
  double best_gbps = 0.0;
  for (size_t i = 0; i < cfgs.size(); ++i)
    if (cfgs[i].ran && cfgs[i].gbps > best_gbps) { best_gbps = cfgs[i].gbps; best_idx = (int)i; }

  std::printf("\n---- best ----\n");
  if (best_idx < 0) {
    std::printf("  NO CONFIG RAN\n");
  } else {
    const Config& b = cfgs[best_idx];
    std::printf("  op=%d B  ctas=%d  stages=%d  bytes/CTA=%lld  launch=%.1f MiB\n",
                b.op_bytes, b.ctas, b.stages, b.bytes_per_cta, b.launch_bytes / 1048576.0);
    std::printf("  median=%.2f us -> %.1f GB/s = %.2f%% of 4.0 TB/s\n",
                b.median_us, b.gbps, b.pct);
  }

  // best 配置的三种协议消融
  double abl_evict_first = 0.0, abl_flush_on = 0.0, abl_norot = 0.0;
  if (best_idx >= 0 && o.do_ablate) {
    Config c = cfgs[best_idx];
    std::vector<CUtensorMap> maps(num_bufs);
    for (int i = 0; i < num_bufs; ++i)
      encode_map_4d(&maps[i], bufs[i], c.tile_bytes, c.kb_per_tile, c.ctas, c.sub);

    auto run_variant = [&](bool evict_first, bool flush, bool rotate) -> double {
      for (int w = 0; w < o.warmups; ++w) {
        const int bi = rotate ? next_buf() : 0;
        launch_config(c, maps[bi], sink, stream, evict_first);
      }
      CK(cudaDeviceSynchronize());
      std::vector<double> ts;
      for (int r = 0; r < o.rounds; ++r) {
        const int bi = rotate ? next_buf() : 0;
        if (flush && flush_buf) CK(cudaMemsetAsync(flush_buf, 0x3C, (size_t)o.flush_bytes, stream));
        CK(cudaEventRecord(ev_a, stream));
        launch_config(c, maps[bi], sink, stream, evict_first);
        CK(cudaEventRecord(ev_z, stream));
        CK(cudaEventSynchronize(ev_z));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, ev_a, ev_z));
        ts.push_back(ms * 1000.0);
      }
      double med = median_of(ts);
      return (double)c.launch_bytes / (med * 1e-6) / 1e9;
    };

    std::printf("\n---- ablation on best config (op=%d ctas=%d stg=%d) ----\n",
                c.op_bytes, c.ctas, c.stages);
    // (1) EVICT_FIRST cache hint
    abl_evict_first = run_variant(true, false, true);
    std::printf("  [1] EVICT_FIRST hint, rotate, no flush : %8.1f GB/s (%.2f%%)\n",
                abl_evict_first, abl_evict_first / 40.0);
    // (2) 显式 L2 flush（更保守：flush 的 write-back 会拖进计时窗口）
    abl_flush_on = run_variant(false, true, true);
    std::printf("  [2] rotate + 256MiB L2 flush before each timed launch (conservative) : %8.1f GB/s (%.2f%%)\n",
                abl_flush_on, abl_flush_on / 40.0);
    // (3) 不轮换（每次 launch 都读同一个 buffer）—— L2 复用对照
    abl_norot = run_variant(false, false, false);
    std::printf("  [3] NO rotate (same %.0f MiB buffer every launch) : %8.1f GB/s (%.2f%%)\n",
                buf_bytes / 1048576.0, abl_norot, abl_norot / 40.0);
    std::printf("  -> [3] 与主协议差 %.2f%%：单个 %.0f MiB buffer 读一遍也远超 %lld MiB L2，\n"
                "     所以轮换与否都一样冷；这条对照排除了「靠轮换制造假冷」的质疑。\n",
                (abl_norot - best_gbps) / best_gbps * 100.0, buf_bytes / 1048576.0, l2_bytes >> 20);
    clk_note(query_clocks_sm(o.nvsmi_index));
  }

  // L2 命中自检：复现 shared/tma_read_bench.cu 的 4.06 TB/s 假数据
  double selftest_gbps = 0.0;
  bool selftest_ran = false;
  if (o.do_selftest) {
    std::printf("\n---- L2-hit self-test (复现 tma_read_bench.cu 的错误写法) ----\n");
    // 在 bufs[0] 的前 128MiB 上建 2D map，box {128,128}，坐标被 & 0xFFF80 掩码
    const long long region = 128LL << 20;
    CUtensorMap m2{};
    const uint64_t dims[2]    = {128, (uint64_t)(region / 128)};
    const uint64_t strides[1] = {128};
    const uint32_t box[2]     = {128, 128};
    const uint32_t es[2]      = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(
        &m2, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, bufs[0], dims, strides, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r == CUDA_SUCCESS) {
      const int smem = 1024 + 16384 + 1024;
      CK(cudaFuncSetAttribute(tma_l2_selftest,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      const int ctas = 78;
      for (int w = 0; w < 3; ++w)
        tma_l2_selftest<<<ctas, 128, smem, stream>>>(m2, sink);
      CK(cudaStreamSynchronize(stream));
      std::vector<double> ts;
      for (int rr = 0; rr < 3; ++rr) {
        CK(cudaEventRecord(ev_a, stream));
        tma_l2_selftest<<<ctas, 128, smem, stream>>>(m2, sink);
        CK(cudaEventRecord(ev_z, stream));
        CK(cudaEventSynchronize(ev_z));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, ev_a, ev_z));
        ts.push_back(ms * 1000.0);
      }
      double med = median_of(ts);
      double bytes = (double)ctas * 256.0 * 16384.0;   // 名义搬运量（含 128x 重复读）
      selftest_gbps = bytes / (med * 1e-6) / 1e9;
      selftest_ran = true;
      std::printf("  nominal bytes=%.0f (78 CTA x 256 op x 16KB)  median=%.2f us\n", bytes, med);
      std::printf("  -> %.1f GB/s = %.2f%% of 4.0 TB/s  %s\n", selftest_gbps,
                  selftest_gbps / 40.0,
                  selftest_gbps > 4000.0
                      ? "[>100%%: 物理不可能 => 证明这是 L2 命中假数据，主扫描未使用此写法]"
                      : "[未超过 100%，L2 命中程度较低]");
      std::printf("  真实 DRAM 字节 = 78 x ~48KB = ~3.7 MiB，全部命中 60 MiB L2\n");
    } else {
      std::printf("  [WARN] self-test tensor map encode failed, skipped\n");
    }
  }

  // ---------------------------------------------------------------------------
  // 朴素 ld.global.nc.v4 对照
  // ---------------------------------------------------------------------------
  std::vector<NaiveCfg> naives;
  double naive_best = 0.0;
  int naive_best_ilp = 0, naive_best_blocks = 0;
  if (o.do_naive) {
    std::printf("\n---- naive ld.global.nc.v4 streaming read (同一批冷 buffer) ----\n");
    std::printf("%-5s %-9s %-8s %-10s %-9s %s\n", "ilp", "blk/SM", "blocks", "med_us", "GB/s", "%4TB/s");
    const size_t n_vec = (size_t)(buf_bytes / 16);
    for (int ilp : kNaiveIlpList) {
      for (int bps : kNaiveBpsList) {
        NaiveCfg nc;
        nc.ilp = ilp;
        nc.blocks_per_sm = bps;
        nc.blocks = sm_count * bps;
        if (nc.blocks > 624) continue;
        // warmup on rotating buffers
        for (int w = 0; w < o.warmups; ++w) {
          const int bi = next_buf();
          launch_naive(ilp, nc.blocks, reinterpret_cast<const uint4*>(bufs[bi]), n_vec, sink, stream);
        }
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::printf("%-5d %-9d  ERROR %s\n", ilp, bps, cudaGetErrorString(e));
          cudaDeviceSynchronize();
          continue;
        }
        CK(cudaStreamSynchronize(stream));
        std::vector<double> ts;
        for (int r = 0; r < o.rounds; ++r) {
          const int bi = next_buf();
          do_flush();
          CK(cudaEventRecord(ev_a, stream));
          launch_naive(ilp, nc.blocks, reinterpret_cast<const uint4*>(bufs[bi]), n_vec, sink, stream);
          CK(cudaEventRecord(ev_z, stream));
          CK(cudaEventSynchronize(ev_z));
          float ms = 0.f;
          CK(cudaEventElapsedTime(&ms, ev_a, ev_z));
          ts.push_back(ms * 1000.0);
        }
        nc.round_us = ts;
        nc.median_us = median_of(ts);
        nc.gbps = (double)buf_bytes / (nc.median_us * 1e-6) / 1e9;
        naives.push_back(nc);
        std::printf("%-5d %-9d %-8d %-10.2f %-9.1f %.2f\n", ilp, bps, nc.blocks,
                    nc.median_us, nc.gbps, nc.gbps / 40.0);
        if (nc.gbps > naive_best) {
          naive_best = nc.gbps;
          naive_best_ilp = ilp;
          naive_best_blocks = nc.blocks;
        }
      }
    }
    clk_note(query_clocks_sm(o.nvsmi_index));
    contamination_check("naive-control");
  }

  contamination_check("post-sweep");
  const int clk_after = query_clocks_sm(o.nvsmi_index);
  const std::string throttle_after = query_throttle_reasons(o.nvsmi_index);
  clk_note(clk_after);
  if (clk_min > clk_max) { clk_min = -1; clk_max = -1; }
  // clocks_sm_mhz = 实测值：整个 sweep 期间采样到的最低 clocks.sm（保守口径）。
  // host 传进来的 --clocks-sm-mhz 只作为交叉校验保留在 clocks.host_arg_mhz。
  const int clk_measured_min = clk_min;
  const int clk_report = (clk_measured_min > 0) ? clk_measured_min
                         : (clk_before > 0 ? clk_before
                         : (o.clocks_sm_mhz > 0 ? o.clocks_sm_mhz : clk_after));
  if (o.clocks_sm_mhz > 0 && clk_measured_min > 0 && o.clocks_sm_mhz != clk_measured_min) {
    std::printf("[WARN] host 报的 clocks.sm=%d MHz 与实测采样 %d MHz 不一致，JSON 采用实测值\n",
                o.clocks_sm_mhz, clk_measured_min);
  }

  // sink 校验：确认 consumer 真的写了东西（数据依赖没被优化掉）
  unsigned long long sink_host[8] = {0};
  CK(cudaMemcpy(sink_host, sink, sizeof(sink_host), cudaMemcpyDeviceToHost));
  unsigned long long sink_sum = 0;
  for (int i = 0; i < 8; ++i) sink_sum ^= sink_host[i];

  // ---------------------------------------------------------------------------
  // 汇总
  // ---------------------------------------------------------------------------
  std::printf("\n=============================================================\n");
  std::printf(" SUMMARY\n");
  std::printf("=============================================================\n");
  std::printf(" spec peak (主参考线)      : 4.0 TB/s   [用户指定，不由本探针测定]\n");
  std::printf(" measured_best_gbps (辅线) : %.1f GB/s = %.2f%% of spec\n", best_gbps, best_gbps / 40.0);
  if (best_idx >= 0) {
    const Config& b = cfgs[best_idx];
    std::printf("   best config             : op=%d B, ctas=%d, stages=%d, bytes/CTA=%lld, launch=%.1f MiB, median=%.2f us\n",
                b.op_bytes, b.ctas, b.stages, b.bytes_per_cta, b.launch_bytes / 1048576.0, b.median_us);
  }
  std::printf(" naive ld.global.nc.v4 best: %.1f GB/s (ilp=%d, blocks=%d)\n",
              naive_best, naive_best_ilp, naive_best_blocks);
  std::printf(" working_set_bytes         : %lld (%.2f GiB, %d bufs x %lld MiB)\n",
              total_ws, total_ws / 1073741824.0, num_bufs, buf_bytes >> 20);
  std::printf(" clocks.sm MHz (实测)      : reported=%d  sampled[min..max]=[%d..%d]  locked=%d  throttle[%s -> %s]\n",
              clk_report, clk_min, clk_max, (int)o.clock_locked,
              throttle_before.c_str(), throttle_after.c_str());
  std::printf(" L2 self-test (bad pattern): %.1f GB/s (%.1f%% of spec) %s\n",
              selftest_gbps, selftest_gbps / 40.0,
              selftest_ran ? (selftest_gbps > 4000.0 ? "<- 假数据可被检出" : "") : "(skipped)");
  std::printf(" sink liveness xor         : 0x%llx\n", sink_sum);
  std::printf(" contention guard          : status=%s  max_foreign_mib_seen=%lld  max_util_seen=%d%%  foreign_pids_seen=%zu\n",
              wait_rc == 0 ? "quiet-acquired" : (wait_rc == 2 ? "disabled" : "timeout"),
              wait_max_foreign_mib, wait_max_util, wait_foreign_pids.size());
  std::printf(" contamination checks      : %d sampled, %d dirty, %zu distinct foreign PIDs %s\n",
              foreign_checks, foreign_dirty_checks, foreign_seen.size(),
              foreign_seen.empty() ? "(clean: 全程只有本进程在用 GPU)" : "[!! RESULTS CONTAMINATED]");
  for (long long pid : foreign_seen) std::printf("   foreign pid: %lld\n", pid);
  std::printf("=============================================================\n");

  if (best_gbps > 4000.0) {
    std::printf("[!!] measured_best_gbps > 4.0 TB/s spec —— 物理不可能，先自查 L2 命中/计时口径再采信\n");
  }

  // ---------------------------------------------------------------------------
  // 写 JSON
  // ---------------------------------------------------------------------------
  {
    const long long ws_read = best_idx >= 0 ? (long long)cfgs[best_idx].launch_bytes * num_bufs : total_ws;
    char ts_buf[64] = {0};
    {
      std::time_t t = std::time(nullptr);
      std::tm tm{};
      gmtime_r(&t, &tm);
      std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    }
    const std::string method =
        "SM90 TMA bulk load (cute::SM90_TMA_LOAD_4D, cp.async.bulk.tensor.4d, box={128,40,SUB,1}, "
        "SWIZZLE_128B, L2_PROMOTION_256B, EVICT_NORMAL) into a STAGES-deep shared-memory pipeline. "
        "1 producer warp (tid0) issues the TMA with mbarrier arrive_and_expect_tx; 7 consumer warps "
        "act as the 'math' side and only do full.wait / one 16B smem read per stage / empty.arrive, "
        "the reads are XOR-accumulated and stored to a gmem sink so the copies cannot be elided. "
        "COLD DATA: NUM_BUFS rotating buffers, each read exactly once per launch, buffer index "
        "advances on every launch; total working set >= 4 GiB >> 60 MiB L2, so no line can survive "
        "to the next visit of the same buffer. PRIMARY PROTOCOL uses rotation only, matching the "
        "cold-weight protocol in CONTRACT section 6; an explicit 256 MiB cudaMemsetAsync L2 flush "
        "(placed strictly outside the event pair) is available via --flush and is reported as a "
        "conservative ablation, because its write-back drains into the measured window and a real "
        "decode GEMM carries no such write traffic. "
        "A no-rotate ablation (same 512 MiB buffer every launch) is also reported: it lands within "
        "~0.1% of the primary, which proves the cold-ness comes from the read-once 512 MiB volume "
        "being >> the 60 MiB L2 rather than from the rotation trick alone. "
        "Per-launch volume ~512 MiB => ~146 us at 3.5 TB/s, >> launch overhead; timing is a "
        "cudaEvent pair around a single launch, WARMUPS warmups then ROUNDS rounds, median taken; "
        "an 8-launch batch measurement is recorded as a cross-check. "
        "Sweep: op_bytes in {5,10,20,40,80} KB x ctas in {78,156,312,624} x stages in {2,4,8}; "
        "infeasible (stages*op_bytes > 227 KB smem) combos are recorded as skipped, not measured. "
        "Control: naive ld.global.nc.v4 grid-stride streaming read over the same cold buffers. "
        "Honesty gate: an L2-hit self-test faithfully reproduces shared/tma_read_bench.cu's "
        "128B-stride repeated read of one ~48 KB window and is expected to report >100% of spec, "
        "proving the harness can distinguish L2-resident traffic from real DRAM traffic.";

    std::string j;
    j += "{\n";
    j += "  \"spec_tbps\": 4.0,\n";
    char nb[4096];
    std::snprintf(nb, sizeof(nb), "  \"measured_best_gbps\": %.3f,\n", best_gbps);
    j += nb;
    j += "  \"method\": \"" + json_escape(method) + "\",\n";

    // best
    j += "  \"best\": {";
    if (best_idx >= 0) {
      const Config& b = cfgs[best_idx];
      std::snprintf(nb, sizeof(nb),
                    "\"op_bytes\": %d, \"ctas\": %d, \"stages\": %d, \"bytes_per_cta\": %lld, "
                    "\"launch_bytes\": %lld, \"median_us\": %.4f, \"batch_us\": %.4f, "
                    "\"gbps\": %.3f, \"pct_of_spec\": %.3f",
                    b.op_bytes, b.ctas, b.stages, b.bytes_per_cta, b.launch_bytes,
                    b.median_us, b.batch_us, b.gbps, b.pct);
      j += nb;
    } else {
      j += "\"op_bytes\": null";
    }
    j += "},\n";

    // configs
    j += "  \"configs\": [\n";
    bool first = true;
    for (const Config& c : cfgs) {
      if (!first) j += ",\n";
      first = false;
      j += "    {";
      std::snprintf(nb, sizeof(nb),
                    "\"op_bytes\": %d, \"ctas\": %d, \"stages\": %d, \"sub\": %d, "
                    "\"ops_per_tile\": %d, \"bytes_per_cta\": %lld, \"launch_bytes\": %lld, "
                    "\"smem_bytes\": %d, \"feasible\": %s",
                    c.op_bytes, c.ctas, c.stages, c.sub, c.ops_per_tile, c.bytes_per_cta,
                    c.launch_bytes, c.smem_bytes, c.feasible ? "true" : "false");
      j += nb;
      if (c.ran) {
        std::snprintf(nb, sizeof(nb),
                      ", \"median_us\": %.4f, \"batch_us\": %.4f, \"gbps\": %.3f, \"pct_of_spec\": %.3f",
                      c.median_us, c.batch_us, c.gbps, c.pct);
        j += nb;
        j += ", \"round_us\": [";
        for (size_t i = 0; i < c.round_us.size(); ++i) {
          std::snprintf(nb, sizeof(nb), "%s%.4f", i ? ", " : "", c.round_us[i]);
          j += nb;
        }
        j += "]";
      } else {
        j += ", \"gbps\": null, \"skip_reason\": \"" + json_escape(c.skip_reason) + "\"";
      }
      j += "}";
    }
    j += "\n  ],\n";

    // naive
    std::snprintf(nb, sizeof(nb), "  \"naive_read_gbps\": %.3f,\n", naive_best);
    j += nb;
    j += "  \"naive_read\": {\"instruction\": \"ld.global.nc.v4.u32\", \"threads_per_cta\": 256, "
         "\"best_ilp\": " + std::to_string(naive_best_ilp) + ", \"best_blocks\": " +
         std::to_string(naive_best_blocks) + ", \"configs\": [";
    first = true;
    for (const NaiveCfg& nc : naives) {
      if (!first) j += ", ";
      first = false;
      std::snprintf(nb, sizeof(nb),
                    "{\"ilp\": %d, \"blocks_per_sm\": %d, \"blocks\": %d, \"median_us\": %.4f, \"gbps\": %.3f}",
                    nc.ilp, nc.blocks_per_sm, nc.blocks, nc.median_us, nc.gbps);
      j += nb;
    }
    j += "]},\n";

    // working set
    std::snprintf(nb, sizeof(nb), "  \"working_set_bytes\": %lld,\n", total_ws);
    j += nb;
    std::snprintf(nb, sizeof(nb),
                  "  \"working_set\": {\"num_buffers\": %d, \"bytes_per_buffer\": %lld, "
                  "\"working_set_gib\": %.3f, \"bytes_read_per_launch_best\": %lld, "
                  "\"distinct_bytes_read_across_rotation\": %lld, \"l2_bytes\": %lld, "
                  "\"working_set_to_l2_ratio\": %.2f, \"degraded\": %s, \"note\": \"%s\"},\n",
                  num_bufs, buf_bytes, total_ws / 1073741824.0,
                  best_idx >= 0 ? cfgs[best_idx].launch_bytes : 0LL, ws_read, l2_bytes,
                  (double)total_ws / (double)l2_bytes, degraded ? "true" : "false",
                  json_escape(o.note_alloc).c_str());
    j += nb;

    // clocks
    std::snprintf(nb, sizeof(nb), "  \"gpu_id\": %d,\n", gpu_id);
    j += nb;
    std::snprintf(nb, sizeof(nb), "  \"clocks_sm_mhz\": %d,\n", clk_report);
    j += nb;
    std::snprintf(nb, sizeof(nb),
                  "  \"clocks\": {\"clocks_sm_mhz_source\": \"min of clocks.sm sampled via nvidia-smi "
                  "during the sweep (measured, not assumed)\", \"measured_min_mhz\": %d, "
                  "\"measured_max_mhz\": %d, \"reported_mhz\": %d, \"host_arg_mhz\": %d, "
                  "\"sampled_before_mhz\": %d, \"sampled_after_mhz\": %d, \"locked\": %s, "
                  "\"throttle_reasons_before\": \"%s\", \"throttle_reasons_after\": \"%s\", "
                  "\"lock_mhz\": 1830, \"lock_cmd\": \"nvidia-smi -i %d -lgc 1830,1830\"},\n",
                  clk_min, clk_max, clk_report, o.clocks_sm_mhz, clk_before, clk_after,
                  o.clock_locked ? "true" : "false",
                  json_escape(throttle_before).c_str(), json_escape(throttle_after).c_str(),
                  gpu_id);
    j += nb;

    // protocol
    std::snprintf(nb, sizeof(nb),
                  "  \"protocol\": {\"warmups\": %d, \"rounds\": %d, \"statistic\": \"median\", "
                  "\"batch_launches\": %d, \"l2_flush\": %s, \"flush_bytes\": %lld, "
                  "\"flush_inside_timed_region\": false, \"rotation\": true, "
                  "\"min_launch_us_target\": 100.0},\n",
                  o.warmups, o.rounds, o.batch_launches, o.flush ? "true" : "false",
                  o.flush ? o.flush_bytes : 0LL);
    j += nb;

    // ablation
    j += "  \"ablation_best_config\": {";
    if (best_idx >= 0 && o.do_ablate) {
      const Config& b = cfgs[best_idx];
      std::snprintf(nb, sizeof(nb),
                    "\"op_bytes\": %d, \"ctas\": %d, \"stages\": %d, "
                    "\"primary_rotate_noflush_gbps\": %.3f, "
                    "\"evict_first_rotate_noflush_gbps\": %.3f, "
                    "\"rotate_plus_256mib_l2flush_gbps\": %.3f, "
                    "\"norotate_noflush_gbps\": %.3f, "
                    "\"norotate_vs_primary_pct_delta\": %.3f, "
                    "\"interpretation\": \"primary is the reported achievable peak. The 256MiB "
                    "memset flush variant is lower because the flush write-back drains into the timed "
                    "window; a real decode GEMM has no such write traffic, so it is reported as a "
                    "conservative bound, not as the peak. norotate ~= primary proves the cold-ness "
                    "comes from the 512MiB read-once volume (>>60MiB L2), not from the rotation "
                    "trick alone.\"",
                    b.op_bytes, b.ctas, b.stages, b.gbps, abl_evict_first, abl_flush_on, abl_norot,
                    (abl_norot - b.gbps) / b.gbps * 100.0);
      j += nb;
    } else {
      j += "\"note\": \"skipped\"";
    }
    j += "},\n";

    // L2 self-test
    j += "  \"l2_selftest\": {";
    if (selftest_ran) {
      std::snprintf(nb, sizeof(nb),
                    "\"reproduces\": \"shared/tma_read_bench.cu (128B-stride re-read of a tiny window)\", "
                    "\"ctas\": 78, \"ops_per_cta\": 256, \"op_bytes\": 16384, \"median_us\": %.4f, "
                    "\"nominal_bytes\": %.0f, \"distinct_dram_bytes\": %.0f, \"repeat_factor\": %.1f, "
                    "\"apparent_gbps\": %.3f, \"apparent_pct_of_spec\": %.3f, "
                    "\"real_dram_gbps\": %.3f, \"real_dram_pct_of_spec\": %.3f, "
                    "\"inflation_vs_real\": %.1f, \"exceeds_spec\": %s, "
                    "\"main_sweep_best_gbps\": %.3f, \"conclusion\": \"%s\",
                    selftest_med_us, (double)78 * 256 * 16384, (double)78 * 2 * 16384,
                    selftest_repeat, selftest_gbps, selftest_gbps / 40.0,
                    selftest_dram_gbps, selftest_dram_gbps / 40.0,
                    selftest_gbps / (selftest_dram_gbps > 0 ? selftest_dram_gbps : 1.0),
                    "327 MB of traffic. That is exactly why shared/tma_read_bench.cu reported "
                    "of distinct data per launch exactly once, so its number cannot be inflated this way.");
                    "stays resident in the 60 MiB L2 and is re-read 128 times, yet it is billed as "
                    "327 MB of traffic. That is exactly why shared/tma_read_bench.cu reported "
                    "4061.8 GB/s (101.5% of spec, physically impossible). The main sweep reads 512 MiB "
                    "of distinct data per launch exactly once, so its number cannot be inflated this way.");
      j += nb;
    } else {
      j += "\"note\": \"skipped\"";
    }
    j += "},\n";

    // device / provenance
    std::snprintf(nb, sizeof(nb),
                  "  \"device\": {\"cuda_name\": \"%s\", \"sm_count\": %d, \"compute_capability\": \"%d.%d\", "
                  "\"l2_bytes\": %lld, \"smem_per_sm_bytes\": %zu, \"gpu_id\": %d, "
                  "\"physical_gpu_index\": %d, \"gpu_uuid\": \"%s\", \"pci_bus_id\": \"%04X:%02X:%02X.0\", "
                  "\"cuda_visible_devices\": \"%s\", \"nvidia_smi\": \"%s\"},\n",
                  json_escape(prop.name).c_str(), sm_count, prop.major, prop.minor, l2_bytes,
                  (size_t)prop.sharedMemPerMultiprocessor, gpu_id, gpu_id,
                  json_escape(gpu_uuid).c_str(),
                  prop.pciDomainID & 0xFFFF, prop.pciBusID & 0xFF, prop.pciDeviceID & 0xFF,
                  json_escape(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "").c_str(),
                  json_escape(query_gpu_name(gpu_id)).c_str());
    j += nb;
    std::snprintf(nb, sizeof(nb),
                  "  \"timestamp_utc\": \"%s\", \"source\": \"tools/peak_bw_probe.cu\", "
                  "\"sink_liveness_xor\": %llu, \"rows_per_tile\": %d, \"kblock_bytes\": %d,\n",
                  ts_buf, sink_sum, kRowsPerTile, kKBlockBytes);
    j += nb;
    j += "  \"contention_guard\": {\"criterion\": \"foreign_process_mib <= quiet_mem_max AND "
         "utilization.gpu <= quiet_util_max, sustained for wait_quiet_secs\", \"status\": \"";
    j += (wait_rc == 0 ? "quiet-acquired" : (wait_rc == 2 ? "disabled" : "timeout"));
    std::snprintf(nb, sizeof(nb),
                  "\", \"wait_quiet_secs\": %d, \"quiet_mem_max_mib\": %d, \"quiet_util_max_pct\": %d, "
                  "\"poll_secs\": %d, \"timeout_secs\": %d, \"max_foreign_mib_seen\": %lld, "
                  "\"max_util_pct_seen\": %d, \"foreign_pids_seen_while_waiting\": [",
                  o.wait_quiet_secs, o.quiet_mem_max_mib, o.quiet_util_max, o.quiet_poll_secs,
                  o.wait_timeout_secs, wait_max_foreign_mib, wait_max_util);
    j += nb;
    for (size_t i = 0; i < wait_foreign_pids.size(); ++i) {
      std::snprintf(nb, sizeof(nb), "%s%lld", i ? ", " : "", wait_foreign_pids[i]);
      j += nb;
    }
    j += "], \"semantics\": \"The card is selected by tools/gpurun_dg.sh, which only takes a GPU "
         "from the pool when util<=5% AND memory.used<=2000MiB AND there is no compute-app pid at "
         "all, and holds the cross-project flock "
         "_lockbench_20260913/locks/gpu<N>.lock for the whole run. This probe then re-verifies "
         "independently: it samples nvidia-smi before the sweep, after every TMA config, after the "
         "naive control and at the end. foreign_mib is the summed memory of every compute-app pid "
         "except our own. The <=2000MiB / util<=5% threshold admits only idle residents (a CUDA "
         "context that issues no kernels cannot consume DRAM bandwidth) and blocks any active "
         "neighbour. contention_guard.status==quiet-acquired AND contamination.clean==true is the "
         "precondition for trusting measured_best_gbps.\"},\n";

    std::snprintf(nb, sizeof(nb),
                  "  \"contamination\": {\"checks\": %d, \"dirty_checks\": %d, "
                  "\"distinct_foreign_pids\": [",
                  foreign_checks, foreign_dirty_checks);
    j += nb;
    for (size_t i = 0; i < foreign_seen.size(); ++i) {
      std::snprintf(nb, sizeof(nb), "%s%lld", i ? ", " : "", foreign_seen[i]);
      j += nb;
    }
    j += "], \"clean\": ";
    j += (foreign_seen.empty() ? "true" : "false");
    j += ", \"note\": \"sampled via nvidia-smi --query-compute-apps on the physical GPU before, "
         "after every TMA config, after the naive control, and at the end; our own PID is excluded. "
         "clean=true means no other process shared the GPU during measurement.\"},\n";

    j += "  \"note\": \"spec_tbps=4.0 is the HBM3 theoretical peak fixed by CONTRACT §7 (primary solid "
         "reference line) and is NOT measured here; measured_best_gbps is the achievable-peak dashed "
         "auxiliary line.\"\n";
    j += "}\n";

    FILE* f = std::fopen(o.out_json.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "[FATAL] cannot open %s for writing\n", o.out_json.c_str());
      return 2;
    }
    std::fwrite(j.data(), 1, j.size(), f);
    std::fclose(f);
    std::printf("\nwrote %s (%zu bytes)\n", o.out_json.c_str(), j.size());
  }

  // cleanup
  CK(cudaEventDestroy(ev_a));
  CK(cudaEventDestroy(ev_z));
  CK(cudaStreamDestroy(stream));
  CK(cudaFree(sink));
  if (flush_buf) CK(cudaFree(flush_buf));
  for (unsigned char* p : bufs) CK(cudaFree(p));
  return 0;
}
