// SPDX-License-Identifier: MIT
//
// bench/stub_kernel.h —— **仅供 harness 流程自测**，不是真 kernel。
// 真 kernel 在 kernels/decode_gemm.cuh；make（默认）走真的，`make stub` 走这里。
//
// 它提供与 kernels/decode_gemm.cuh *同名同形* 的 API 子集，所以
// bench/variant_impl.cuh 只有一条代码路径，stub 与真 kernel 都能编：
//   KernelConfig<15 params> / Problem / GemmRuntime / LaunchPlan
//   make_launch_plan / ensure_smem_attribute / launch_with_maps / launch
//   make_weight_tma / make_activation_tma / fp8_decode_gemm
//   prepack::padded_n / packed_weight_bytes / launch_prepack_weight<BM,TILES>
//
// 与真 kernel 的差别（stub 的自测边界，见 bench/README.md）：
//   * naive fp32 GEMM，无 wgmma/TMA/多级流水 -> 性能数字无意义（慢几个数量级）
//   * 数值路径一致（fp8->float，per-k_block combined_scale，fmaf 累加，
//     __float2bfloat16_rn 写出）-> correctness/rel_l2 管线可自测
//   * packed 布局与 kernels/weight_prepack.cuh 完全一致（tile-major
//     [tile][k_block][BM rows x 128B]，pad 行填 0）-> prepack/paddedN/冷权重
//     set 大小这条链路是真跑过的
//   * PDL：kPdl 时在 kernel 开头就发 griddepcontrol.launch_dependents，用来验证
//     harness 的 PSS attr 计时链路真的生效（真 kernel 的 trigger 在 producer
//     最后一次 TMA 之后，重叠量小得多，所以 stub 的 PDL 数字会好看过头）
//   * SPLIT_K_CTA：stub 忽略（每个 CTA 算全 K 直接写 bf16），输出仍正确；
//     harness 侧 ws/sem 分配与清零照跑
//   * tensor map 是 128B 不透明块：stub 把 (base 指针, 形状, 布局) memcpy 进去，
//     device 侧再读回来（真 kernel 用 cuTensorMapEncodeTiled + TMA 指令）
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstring>

namespace decode_gemm {

inline constexpr uint32_t kKBlock = 128;
inline constexpr uint32_t kWgmmaRows = 64;
inline constexpr uint32_t kRegisterFileSize = 65536;
inline constexpr uint32_t kMaxDynamicSmemBytes = 227 * 1024;
inline constexpr uint32_t kSmemPaddingBytes = 4096;

template <class T>
constexpr T ceil_div(T a, T b) { return (a + b - 1) / b; }
inline constexpr uint32_t align_up_u32(uint32_t v, uint32_t a) {
  return (v + a - 1) / a * a;
}

// ------------------------------------------------------------ KernelConfig --
template <int M_TILE_, int OUTPUT_ROWS_PER_CTA_, int NUM_MATH_WG_, int MATH_REGS_,
          int STAGES_PER_WG_, int TILES_PER_CTA_, int SUBS_PER_STAGE_, bool PREPACK_,
          bool PDL_, bool EPI_OVERLAP_, bool SPLIT_K_CTA_, int PROD_REGS_ = 32,
          bool SINGLE_PRODUCER_WARP_ = false, bool PRELOAD_ACT_ = false,
          bool SCALE_IN_SMEM_ = (M_TILE_ <= 16)>
struct KernelConfig {
  static_assert(M_TILE_ >= 8 && M_TILE_ <= 128 && (M_TILE_ & (M_TILE_ - 1)) == 0,
                "M_TILE must be one of 8/16/32/64/128");
  static_assert(OUTPUT_ROWS_PER_CTA_ > 0 && OUTPUT_ROWS_PER_CTA_ <= kWgmmaRows &&
                    OUTPUT_ROWS_PER_CTA_ % 8 == 0, "BM must be a multiple of 8 in (0,64]");
  static_assert(NUM_MATH_WG_ == 1 || NUM_MATH_WG_ == 2 || NUM_MATH_WG_ == 4, "WG must be 1/2/4");
  static_assert(TILES_PER_CTA_ > 0 && NUM_MATH_WG_ % TILES_PER_CTA_ == 0, "WG % TILES != 0");
  static_assert((NUM_MATH_WG_ * MATH_REGS_ + PROD_REGS_) * 128 <= kRegisterFileSize,
                "register budget exceeded");
  static_assert(!EPI_OVERLAP_ || PDL_, "EPI_OVERLAP requires PDL");
  static_assert(!PRELOAD_ACT_ || (M_TILE_ == 8 && PREPACK_), "PRELOAD_ACT: M_TILE=8 + PREPACK only");
  static_assert(2 * TILES_PER_CTA_ + 1 <= 16, "too many named barrier ids");

  static constexpr uint32_t kMTile = static_cast<uint32_t>(M_TILE_);
  static constexpr uint32_t kOutputRows = static_cast<uint32_t>(OUTPUT_ROWS_PER_CTA_);
  static constexpr uint32_t kNumMathWgs = static_cast<uint32_t>(NUM_MATH_WG_);
  static constexpr uint32_t kMathRegisters = static_cast<uint32_t>(MATH_REGS_);
  static constexpr uint32_t kProducerRegs = static_cast<uint32_t>(PROD_REGS_);
  static constexpr uint32_t kStagesPerMathWg = static_cast<uint32_t>(STAGES_PER_WG_);
  static constexpr uint32_t kOutputTilesPerCta = static_cast<uint32_t>(TILES_PER_CTA_);
  static constexpr uint32_t kSubsPerStage = static_cast<uint32_t>(SUBS_PER_STAGE_);
  static constexpr bool kPrepack = PREPACK_;
  static constexpr bool kPdl = PDL_;
  static constexpr bool kEpiOverlap = EPI_OVERLAP_;
  static constexpr bool kSplitKCta = SPLIT_K_CTA_;
  static constexpr bool kSingleProducerWarp = SINGLE_PRODUCER_WARP_;
  static constexpr bool kPreloadActivation = PRELOAD_ACT_;
  static constexpr bool kScaleInSmem = SCALE_IN_SMEM_;

  static constexpr uint32_t kWgsPerOutputTile = kNumMathWgs / kOutputTilesPerCta;
  static constexpr uint32_t kStages = kNumMathWgs * kStagesPerMathWg;
  static constexpr uint32_t kNumThreads = (kNumMathWgs + 1) * 128;

  static constexpr uint32_t kSubtileWeightBytes = kOutputRows * kKBlock;
  static constexpr uint32_t kWeightSmemBytes =
      kStages * kSubsPerStage * kSubtileWeightBytes + kSmemPaddingBytes;
  static constexpr uint32_t kBarrierBytes =
      align_up_u32(2 * kStages * 8 + (kPreloadActivation ? 8u : 0u), 16u);
  static constexpr uint32_t kPartialBytes =
      kWgsPerOutputTile > 1
          ? align_up_u32(kNumMathWgs * kWgmmaRows * kMTile * sizeof(float), 16u)
          : 0u;
  static constexpr uint32_t kSplitKFlagBytes =
      kSplitKCta ? align_up_u32(kOutputTilesPerCta * 4u, 16u) : 0u;
  static constexpr uint32_t kFixedSmemBytes =
      kWeightSmemBytes + kBarrierBytes + kPartialBytes + kSplitKFlagBytes;
  static_assert(kFixedSmemBytes <= kMaxDynamicSmemBytes,
                "static shared-memory footprint already exceeds 227 KiB");

  static __host__ __device__ constexpr uint32_t act_rows(int m) {
    return m < static_cast<int>(kMTile) ? static_cast<uint32_t>(m) : kMTile;
  }
  static __host__ __device__ constexpr uint32_t activation_smem_bytes(int m, int k) {
    return kPreloadActivation
               ? act_rows(m) * static_cast<uint32_t>(k)
               : kStages * kSubsPerStage * act_rows(m) * kKBlock;
  }
  static __host__ __device__ constexpr uint32_t scale_smem_bytes(int k) {
    return kScaleInSmem ? align_up_u32(kOutputTilesPerCta * kMTile * 2 *
                                           (static_cast<uint32_t>(k) / kKBlock) *
                                           sizeof(float), 16u)
                        : 0u;
  }
  static __host__ __device__ constexpr uint32_t dynamic_smem_bytes(int m, int k) {
    return kFixedSmemBytes + align_up_u32(activation_smem_bytes(m, k), 16u) +
           scale_smem_bytes(k);
  }
  static __host__ __device__ constexpr uint32_t num_output_tiles(int n) {
    return ceil_div(static_cast<uint32_t>(n), kOutputRows);
  }
  static __host__ __device__ constexpr uint32_t num_ctas_x(int n) {
    return ceil_div(num_output_tiles(n), kOutputTilesPerCta);
  }
  static const char* validate(int m, int n, int k) {
    if (m < 1 || static_cast<uint32_t>(m) > kMTile) return "M out of [1, M_TILE]";
    if (n < 1) return "N must be positive";
    if (k % kKBlock != 0) return "K must be a multiple of 128";
    const uint32_t nkb = static_cast<uint32_t>(k) / kKBlock;
    if (nkb % kSubsPerStage != 0) return "K/128 must be a multiple of SUBS_PER_STAGE";
    const uint32_t chunks = nkb / kSubsPerStage;
    if (chunks % kWgsPerOutputTile != 0)
      return "K/(128*SUBS) must be a multiple of WGsPerOutputTile";
    if (kSplitKCta) {
      if (chunks % 2 != 0) return "SPLIT_K_CTA needs an even number of k chunks";
      if ((chunks / 2) % kWgsPerOutputTile != 0)
        return "SPLIT_K_CTA: (K/2)/(128*SUBS) must be a multiple of WGsPerOutputTile";
      if ((chunks / 2) < kStagesPerMathWg)
        return "SPLIT_K_CTA: not enough k chunks to fill the pipeline";
    }
    if (dynamic_smem_bytes(m, k) > kMaxDynamicSmemBytes)
      return "dynamic shared memory exceeds 227 KiB";
    return nullptr;
  }
};

// ---------------------------------------------------- Problem / runtime -----
struct Problem {
  int M = 0;
  int N = 0;
  int K = 0;
  const __nv_fp8_e4m3* activation = nullptr;
  const __nv_fp8_e4m3* weight = nullptr;
  const float* weight_scales = nullptr;
  const float* activation_scales = nullptr;
  __nv_bfloat16* output = nullptr;
  float* splitk_ws = nullptr;
  int* splitk_sem = nullptr;
};

struct GemmRuntime {
  int M = 0;
  int N = 0;
  int K = 0;
  float* splitk_ws = nullptr;
  int* splitk_sem = nullptr;
};

struct LaunchPlan {
  dim3 grid{0, 0, 0};
  dim3 block{0, 0, 0};
  uint32_t smem_bytes = 0;
  int num_output_tiles = 0;
  int padded_n = 0;
  const char* error = nullptr;
};

// ------------------------------------------------------- tensor map 走私 ----
namespace detail {

struct StubMapPayload {
  const void* base;
  long long rows;
  long long cols;
  int bm;
  int tiles;
  int subs;
  int prepack;
  int m_tile;
  int k_blocks;
};
static_assert(sizeof(StubMapPayload) <= sizeof(CUtensorMap),
              "stub payload must fit inside CUtensorMap");

inline CUtensorMap pack_map(const StubMapPayload& p) {   // host only
  CUtensorMap m{};
  std::memcpy(&m, &p, sizeof(p));
  return m;
}
// device 侧不能用 memcpy：CUtensorMap 是 alignas(64) 的 128B 不透明块
__host__ __device__ inline StubMapPayload unpack_map(const CUtensorMap& m) {
  return *reinterpret_cast<const StubMapPayload*>(&m);
}

// packed 布局（= kernels/weight_prepack.cuh）：
//   ((tile * num_k_blocks + k_block) * BM + row_in_tile) * 128 + (k % 128)
__device__ __forceinline__ float stub_weight_at(const StubMapPayload& w, int n,
                                                int k) {
  const __nv_fp8_e4m3* base = static_cast<const __nv_fp8_e4m3*>(w.base);
  size_t idx;
  if (w.prepack) {
    const int kb = k >> 7;
    const int kin = k & 127;
    const int tile = n / w.bm;
    const int row = n - tile * w.bm;
    idx = (static_cast<size_t>(tile) * w.k_blocks + kb) * w.bm + row;
    idx = idx * 128 + kin;
  } else {
    idx = static_cast<size_t>(n) * w.cols + k;
  }
  return static_cast<float>(base[idx]);
}

constexpr int kStubThreads = 256;
inline int stub_grid(int M, int N) {
  const long long total = static_cast<long long>(M) * ((N + 7) / 8);
  return static_cast<int>((total + kStubThreads - 1) / kStubThreads);
}

}  // namespace detail

template <typename Cfg>
inline CUtensorMap make_weight_tma(const Problem& p) {
  const int bm = static_cast<int>(Cfg::kOutputRows);
  const int padded = static_cast<int>(Cfg::num_output_tiles(p.N)) * bm;
  detail::StubMapPayload pay{p.weight,
                             padded,
                             p.K,
                             bm,
                             static_cast<int>(Cfg::kOutputTilesPerCta),
                             static_cast<int>(Cfg::kSubsPerStage),
                             Cfg::kPrepack ? 1 : 0,
                             static_cast<int>(Cfg::kMTile),
                             p.K / 128};
  return detail::pack_map(pay);
}

template <typename Cfg>
inline CUtensorMap make_activation_tma(const Problem& p) {
  detail::StubMapPayload pay{p.activation,
                             p.M,
                             p.K,
                             static_cast<int>(Cfg::kOutputRows),
                             static_cast<int>(Cfg::kOutputTilesPerCta),
                             static_cast<int>(Cfg::kSubsPerStage),
                             0,
                             static_cast<int>(Cfg::kMTile),
                             p.K / 128};
  return detail::pack_map(pay);
}

// ------------------------------------------------------------ device kernel -
template <typename Cfg>
__global__ void fp8_decode_gemm(__nv_bfloat16* __restrict__ output,
                                const float* __restrict__ weight_scales,
                                const float* __restrict__ activation_scales,
                                const __grid_constant__ CUtensorMap w_tma,
                                const __grid_constant__ CUtensorMap a_tma,
                                GemmRuntime rt) {
  extern __shared__ float stub_smem[];
  (void)stub_smem;
  if (Cfg::kPdl) asm volatile("griddepcontrol.launch_dependents;");
  const detail::StubMapPayload w = detail::unpack_map(w_tma);
  const detail::StubMapPayload a = detail::unpack_map(a_tma);
  const int M = rt.M, N = rt.N, K = rt.K;
  const int kb_count = K / 128;
  const int n_groups = (N + 7) / 8;
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= static_cast<long long>(M) * n_groups) return;
  const int m = static_cast<int>(tid / n_groups);
  const int n0 = static_cast<int>(tid - static_cast<long long>(m) * n_groups) * 8;
  const __nv_fp8_e4m3* abase = static_cast<const __nv_fp8_e4m3*>(a.base);

  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.0f;

  for (int kb = 0; kb < kb_count; ++kb) {
    float local[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) local[j] = 0.0f;
    const int k_base = kb * 128;
    for (int kk = 0; kk < 128; ++kk) {
      const int k = k_base + kk;
      const float av =
          static_cast<float>(abase[static_cast<size_t>(m) * K + k]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int n = n0 + j;
        if (n < N) local[j] += av * detail::stub_weight_at(w, n, k);
      }
    }
    const float as = activation_scales[static_cast<size_t>(m) * kb_count + kb];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float ws =
            weight_scales[static_cast<size_t>(n / 128) * kb_count + kb];
        acc[j] = fmaf(as * ws, local[j], acc[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int n = n0 + j;
    if (n < N)
      output[static_cast<size_t>(m) * N + n] = __float2bfloat16_rn(acc[j]);
  }
  if (Cfg::kEpiOverlap && Cfg::kPdl)
    asm volatile("griddepcontrol.launch_dependents;");
}

// ------------------------------------------------------------- host side ----
template <typename Cfg>
inline LaunchPlan make_launch_plan(const Problem& p) {
  LaunchPlan plan;
  plan.error = Cfg::validate(p.M, p.N, p.K);
  plan.num_output_tiles = static_cast<int>(Cfg::num_output_tiles(p.N));
  plan.padded_n = plan.num_output_tiles * static_cast<int>(Cfg::kOutputRows);
  // stub 的 grid 是 naive GEMM 的 grid（真 kernel 是 num_ctas_x x (SPLITK?2:1)）
  plan.grid = dim3(static_cast<unsigned>(detail::stub_grid(p.M, p.N)), 1u, 1u);
  plan.block = dim3(static_cast<unsigned>(detail::kStubThreads), 1u, 1u);
  plan.smem_bytes = Cfg::dynamic_smem_bytes(p.M, p.K);
  return plan;
}

template <typename Cfg>
inline cudaError_t ensure_smem_attribute(uint32_t smem_bytes) {
  static std::atomic<int> cached{-1};
  if (cached.load(std::memory_order_relaxed) == static_cast<int>(smem_bytes))
    return cudaSuccess;
  const cudaError_t err = cudaFuncSetAttribute(
      fp8_decode_gemm<Cfg>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (err == cudaSuccess)
    cached.store(static_cast<int>(smem_bytes), std::memory_order_relaxed);
  return err;
}

template <typename Cfg>
inline cudaError_t launch_with_maps(const Problem& p, const CUtensorMap& w_tma,
                                    const CUtensorMap& a_tma, cudaStream_t stream) {
  const LaunchPlan plan = make_launch_plan<Cfg>(p);
  if (plan.error != nullptr) return cudaErrorInvalidValue;
  if (Cfg::kSplitKCta && (p.splitk_ws == nullptr || p.splitk_sem == nullptr))
    return cudaErrorInvalidValue;
  const cudaError_t attr_err = ensure_smem_attribute<Cfg>(plan.smem_bytes);
  if (attr_err != cudaSuccess) return attr_err;
  GemmRuntime rt;
  rt.M = p.M; rt.N = p.N; rt.K = p.K;
  rt.splitk_ws = p.splitk_ws; rt.splitk_sem = p.splitk_sem;
  cudaLaunchConfig_t config{};
  cudaLaunchAttribute attributes[1]{};
  config.gridDim = plan.grid;
  config.blockDim = plan.block;
  config.dynamicSmemBytes = plan.smem_bytes;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 0;
  if constexpr (Cfg::kPdl) {
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&config, fp8_decode_gemm<Cfg>, p.output,
                            p.weight_scales, p.activation_scales, w_tma, a_tma, rt);
}

template <typename Cfg>
inline cudaError_t launch(const Problem& p, cudaStream_t stream = nullptr) {
  if (Cfg::validate(p.M, p.N, p.K) != nullptr) return cudaErrorInvalidValue;
  const CUtensorMap w_tma = make_weight_tma<Cfg>(p);
  const CUtensorMap a_tma = make_activation_tma<Cfg>(p);
  return launch_with_maps<Cfg>(p, w_tma, a_tma, stream);
}

// ------------------------------------------------------------- prepack ------
namespace prepack {

inline constexpr uint32_t kBytesPerUint4 = 16;
inline constexpr uint32_t kUint4PerKBlock = kKBlock / kBytesPerUint4;

inline int padded_n(int n, int output_rows_per_cta) {
  return ((n + output_rows_per_cta - 1) / output_rows_per_cta) * output_rows_per_cta;
}
inline size_t packed_weight_bytes(int n, int k, int output_rows_per_cta) {
  return static_cast<size_t>(padded_n(n, output_rows_per_cta)) *
         static_cast<size_t>(k);
}

template <int OutputRowsPerCta, int OutputTilesPerCta = 1>
__global__ void prepack_weight_u4(const uint4* __restrict__ source,
                                  uint4* __restrict__ destination, int n, int k,
                                  int num_output_tiles, int num_k_blocks) {
  const uint32_t vectors_per_row = static_cast<uint32_t>(k) / kUint4PerKBlock;
  const uint32_t total_vectors =
      static_cast<uint32_t>(num_output_tiles) *
      static_cast<uint32_t>(OutputRowsPerCta) * vectors_per_row;
  const uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total_vectors) return;
  const uint32_t row_global = linear / vectors_per_row;
  const uint32_t k_vector = linear - row_global * vectors_per_row;
  const uint32_t k_block = k_vector / kUint4PerKBlock;
  const uint32_t k_vector_in_block = k_vector - k_block * kUint4PerKBlock;
  const uint32_t output_tile = row_global / OutputRowsPerCta;
  const uint32_t row_in_tile = row_global - output_tile * OutputRowsPerCta;
  const uint32_t packed_vector =
      ((output_tile * static_cast<uint32_t>(num_k_blocks) + k_block) *
           static_cast<uint32_t>(OutputRowsPerCta) +
       row_in_tile) *
          kUint4PerKBlock +
      k_vector_in_block;
  uint4 value = {0u, 0u, 0u, 0u};
  if (row_global < static_cast<uint32_t>(n))
    value = source[static_cast<size_t>(row_global) * vectors_per_row + k_vector];
  destination[packed_vector] = value;
}

template <int OutputRowsPerCta, int OutputTilesPerCta = 1>
inline cudaError_t launch_prepack_weight(const __nv_fp8_e4m3* source,
                                         __nv_fp8_e4m3* destination, int n, int k,
                                         cudaStream_t stream = nullptr) {
  if (k % kKBlock != 0) return cudaErrorInvalidValue;
  const int num_output_tiles = (n + OutputRowsPerCta - 1) / OutputRowsPerCta;
  const int num_k_blocks = k / kKBlock;
  const size_t total_vectors =
      static_cast<size_t>(num_output_tiles) * OutputRowsPerCta * (k / kBytesPerUint4);
  constexpr uint32_t kThreads = 256;
  const uint32_t blocks =
      static_cast<uint32_t>((total_vectors + kThreads - 1) / kThreads);
  prepack_weight_u4<OutputRowsPerCta, OutputTilesPerCta>
      <<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const uint4*>(source),
          reinterpret_cast<uint4*>(destination), n, k, num_output_tiles,
          num_k_blocks);
  return cudaGetLastError();
}

}  // namespace prepack
}  // namespace decode_gemm
