// SPDX-License-Identifier: MIT
//
// ============================================================================
//  decode_gemm — FP8 (e4m3) decode GEMM for H20/SM90, one clean kernel
// ============================================================================
//
//  out[M,N] = act[M,K] @ W[N,K]^T,  fp8 e4m3 inputs, per-128(K) scales, bf16 out
//
//  This is the "clean rewrite" of the reference kernel
//  DSA-learn/mk/h20/cutedsl/shared/swapab_m1_kernel_v2.cuh (the 945-line oracle
//  that reached 85.3% of the 4.0 TB/s HBM3 spec line).  Same numerical path,
//  same producer/consumer pipeline, same barrier handshakes and stage rotation
//  formula — but:
//
//    * N, K and M are runtime values (tensor maps are built on the host per
//      launch) instead of template parameters, so one instantiation serves all
//      model shapes;
//    * M_TILE (the compile-time upper bound of runtime M) spans 8..128 instead
//      of the oracle's conservative LogicalM<=4;
//    * every technique of the article's step ladder S0..S6 is a compile-time
//      switch on one single kernel template, so the steps are directly
//      comparable and none of them can silently rot.
//
// ----------------------------------------------------------------------------
//  Why swap A/B (the foundation, step S0)
// ----------------------------------------------------------------------------
//  `wgmma.mma_async.m64n8k32` *requires* 64 rows on the A side.  In decode, M
//  is 1..128 while N is thousands, so putting the activation on the A side
//  wastes up to 63/64 of the tensor-core throughput.  We therefore swap:
//
//      A operand = weight tile      (64 wgmma rows == 64 output rows n)
//      B operand = activation tile  (8 wgmma cols == 8 activation rows m)
//
//  A warp group covers rows {warp*16 + lane/4, +8} of the output tile, and
//  columns {(lane&3)*2, +1} of the M range per n8 wgmma.  M_TILE > 8 simply
//  means M_TILES = M_TILE/8 n8 wgmma groups per k_block.
//
//  Decode at M<=128 is a pure bandwidth problem: arithmetic intensity is
//  2*M FLOP/byte, i.e. one to two orders of magnitude below the H20 ridge
//  point, so the only thing that matters is streaming N*K weight bytes from HBM
//  as fast as possible.  Everything below (op size, CTA count, pipeline depth,
//  PDL, split-K) is about exactly that.
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <mutex>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "sm90_compat.cuh"

namespace decode_gemm {

using Barrier = cutlass::arch::ClusterTransactionBarrier;
using fp8_swapab_wgmma::sm90_compat::ld_shared;
using fp8_swapab_wgmma::sm90_compat::make_smem_desc;
using fp8_swapab_wgmma::sm90_compat::st_shared;
using fp8_swapab_wgmma::sm90_compat::warpgroup_arrive;
using fp8_swapab_wgmma::sm90_compat::warpgroup_commit_batch;
using fp8_swapab_wgmma::sm90_compat::warpgroup_fence_operand;
using fp8_swapab_wgmma::sm90_compat::warpgroup_wait;

// ---------------------------------------------------------------------------
// Hardware / layout constants
// ---------------------------------------------------------------------------
inline constexpr uint32_t kKBlock = 128;        // one scale group == 4 x wgmma k32
inline constexpr uint32_t kWgmmaRows = 64;      // wgmma m64 (== A rows == output rows)
inline constexpr uint32_t kWgmmaCols = 8;       // wgmma n8  (== B rows == activation rows)
inline constexpr uint32_t kK32PerBlock = kKBlock / 32;
inline constexpr uint32_t kRegisterFileSize = 65536;
inline constexpr uint32_t kMaxDynamicSmemBytes = 227 * 1024;
// GMMA "stride byte offset": distance between two 8-row groups of a K-major
// SWIZZLE_128B operand == 8 rows * 128 bytes.
inline constexpr uint32_t kWgmmaCoreStrideBytes = 8 * kKBlock;
// A tile of <= 64 output rows spans at most two 128-row scale groups.
inline constexpr uint32_t kScaleSlotsPerOutputTile = 2;
// v2's trick: 4 KiB of dead smem between the weight stages and the activation
// region shifts every stage start address, which spreads the TMA writes over
// more smem banks.
inline constexpr uint32_t kSmemPaddingBytes = 4096;

template <class T>
__host__ __device__ constexpr T ceil_div(T a, T b) { return (a + b - 1) / b; }

__host__ __device__ constexpr uint32_t align_up_u32(uint32_t v, uint32_t a) {
  return ceil_div(v, a) * a;
}

// ---------------------------------------------------------------------------
// KernelConfig — every technique is one template parameter
// ---------------------------------------------------------------------------
//  M_TILE              compile-time upper bound of runtime M (8/16/32/64/128).
//                      The B operand is always n8; M_TILES = M_TILE/8 n8 wgmma
//                      groups are issued per k_block.
//  OUTPUT_ROWS_PER_CTA BM: output rows (weight rows) handled by one output tile.
//                      <= 64 (one wgmma m64 A operand).
//  NUM_MATH_WG         math warp groups (1/2/4).  Plus one producer warp group,
//                      so the CTA has (NUM_MATH_WG+1)*128 threads.
//  MATH_REGS           register budget handed to the math warp groups by
//                      `warpgroup_reg_alloc` after the producer deallocates.
//  STAGES_PER_WG       pipeline depth per math warp group; total stages
//                      kStages = NUM_MATH_WG * STAGES_PER_WG.
//  TILES_PER_CTA       output tiles per CTA.  NUM_MATH_WG / TILES_PER_CTA warp
//                      groups cooperate on one tile == intra-CTA split-K.
//  SUBS_PER_STAGE      (S2 "blockK") k_blocks per TMA op / per pipeline stage.
//                      One op moves SUBS*BM*128 weight bytes instead of BM*128:
//                      bigger ops are worth ~10 bandwidth points on cold data.
//  PREPACK             (S3) tile-major packed weight vs raw [N,K].
//  PDL                 (S1) issue `griddepcontrol.launch_dependents` from the
//                      producer once the last TMA of the CTA is in flight.
//  EPI_OVERLAP         (S5) second trigger from the math side, right before the
//                      epilogue stores, so the next kernel's prologue overlaps
//                      our stores.
//  SPLIT_K_CTA         (S6) CTA-level K split: gridDim.y = S, each of the S
//                      slices of CTAs takes one contiguous 1/S of K and the S
//                      fp32 partials are reduced through global memory with a
//                      self-resetting semaphore.  Kept as a bool for interface
//                      compatibility with CONTRACT 4 (true == S=2); the width S
//                      itself is SPLITK_FACTOR below.
//  SPLITK_FACTOR       (S6) the split width S: 0 = legacy (follow SPLIT_K_CTA:
//                      true -> 2, false -> 1), 1 = off, 2/4/8 = gridDim.y = S.
//                      S >= 2 implies split-K on regardless of SPLIT_K_CTA, so
//                      old instantiations keep their exact meaning.
//  PROD_REGS           producer warp-group register budget (deallocated).
//  SINGLE_PRODUCER_WARP let one producer warp issue all TMAs (v2: measured, not
//                      a win at BM48/SUBS2 -> default off).
//  PRELOAD_ACT         load the whole activation once into smem instead of
//                      staging it (v2: measured +0.18us, i.e. no win; only
//                      allowed for M_TILE=8 + PREPACK per CONTRACT).
//  SCALE_IN_SMEM       pre-multiply w_scale*a_scale into smem in the prologue
//                      (v2's path) or read both scales with __ldg straight from
//                      L1 inside the main loop.  The smem variant needs
//                      TILES*M_TILE*2*num_k_blocks*4 bytes, which is 57 KiB at
//                      M_TILE=128/K=7168 — so it defaults off for M_TILE>=32.
// ---------------------------------------------------------------------------
template <int M_TILE_,
          int OUTPUT_ROWS_PER_CTA_,
          int NUM_MATH_WG_,
          int MATH_REGS_,
          int STAGES_PER_WG_,
          int TILES_PER_CTA_,
          int SUBS_PER_STAGE_,
          bool PREPACK_,
          bool PDL_,
          bool EPI_OVERLAP_,
          bool SPLIT_K_CTA_,
          int PROD_REGS_ = 32,
          bool SINGLE_PRODUCER_WARP_ = false,
          bool PRELOAD_ACT_ = false,
          bool SCALE_IN_SMEM_ = (M_TILE_ <= 16),
          // Appended *last* (with a default) so that every existing
          // positional instantiation of the CONTRACT 4 template keeps working.
          int SPLITK_FACTOR_ = 0>
struct KernelConfig {
  static_assert(M_TILE_ >= 8 && M_TILE_ <= 128 && (M_TILE_ & (M_TILE_ - 1)) == 0,
                "M_TILE must be one of 8/16/32/64/128");
  static_assert(OUTPUT_ROWS_PER_CTA_ > 0 && OUTPUT_ROWS_PER_CTA_ <= kWgmmaRows &&
                    OUTPUT_ROWS_PER_CTA_ % 8 == 0,
                "OUTPUT_ROWS_PER_CTA (BM) must be a multiple of 8 in (0, 64]");
  static_assert(NUM_MATH_WG_ == 1 || NUM_MATH_WG_ == 2 || NUM_MATH_WG_ == 4,
                "NUM_MATH_WG must be 1, 2 or 4");
  static_assert(TILES_PER_CTA_ > 0 && NUM_MATH_WG_ % TILES_PER_CTA_ == 0,
                "NUM_MATH_WG must be divisible by TILES_PER_CTA");
  static_assert(STAGES_PER_WG_ > 0, "STAGES_PER_WG must be positive");
  static_assert(SUBS_PER_STAGE_ > 0 && SUBS_PER_STAGE_ <= 64,
                "SUBS_PER_STAGE must be in [1, 64]");
  static_assert(MATH_REGS_ >= 24 && MATH_REGS_ <= 256 && MATH_REGS_ % 8 == 0,
                "MATH_REGS must be a multiple of 8 in the PTX range [24, 256]");
  static_assert(PROD_REGS_ >= 24 && PROD_REGS_ <= 256 && PROD_REGS_ % 8 == 0,
                "PROD_REGS must be a multiple of 8 in the PTX range [24, 256]");
  static_assert((NUM_MATH_WG_ * MATH_REGS_ + PROD_REGS_) * 128 <= kRegisterFileSize,
                "register budget exceeded: (WG*MATH_REGS + PROD_REGS)*128 <= 65536");
  // EPI_OVERLAP 与 PDL 独立：host 侧 ProgrammaticStreamSerialization 属性由
  // harness 的计时协议决定，kernel 只决定两个触发点发不发（--pdl-placement 消融需要
  // producer-only / both / store-only 三种组合）。
  static_assert(!PRELOAD_ACT_ || (M_TILE_ == 8 && PREPACK_),
                "PRELOAD_ACT is only allowed for M_TILE=8 + PREPACK (CONTRACT 4)");
  static_assert((NUM_MATH_WG_ / TILES_PER_CTA_) * 128 % 32 == 0,
                "named barrier participant counts must be warp multiples");
  // Named barrier ids: 0 = scale prologue, 1..TILES = intra-CTA split-K reduce,
  // TILES+1..2*TILES = CTA-split-K handshake.  16 ids exist in hardware.
  static_assert(2 * TILES_PER_CTA_ + 1 <= 16, "too many named barrier ids");
  static_assert(SPLITK_FACTOR_ == 0 || SPLITK_FACTOR_ == 1 || SPLITK_FACTOR_ == 2 ||
                    SPLITK_FACTOR_ == 4 || SPLITK_FACTOR_ == 8,
                "SPLITK_FACTOR must be 0 (legacy: follow SPLIT_K_CTA), 1 (off), 2, 4 or 8");
  static_assert(SPLITK_FACTOR_ != 1 || !SPLIT_K_CTA_,
                "SPLITK_FACTOR=1 means split-K off and contradicts SPLIT_K_CTA=true");

  static constexpr uint32_t kMTile = static_cast<uint32_t>(M_TILE_);
  static constexpr uint32_t kMTiles = kMTile / kWgmmaCols;      // n8 wgmma groups
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
  // Effective split width S (gridDim.y).  SPLITK_FACTOR wins when it is >= 2;
  // 1 forces split-K off; 0 falls back to the legacy bool (true == 2).
  static constexpr uint32_t kSplitKFactor =
      (SPLITK_FACTOR_ >= 2) ? static_cast<uint32_t>(SPLITK_FACTOR_)
                            : ((SPLITK_FACTOR_ == 1 || !SPLIT_K_CTA_) ? 1u : 2u);
  static constexpr bool kSplitKCta = (kSplitKFactor > 1u);
  static_assert(!kSplitKCta || kSplitKFactor == 2u || kSplitKFactor == 4u ||
                    kSplitKFactor == 8u,
                "split-K width must be 2, 4 or 8");
  static constexpr bool kSingleProducerWarp = SINGLE_PRODUCER_WARP_;
  static constexpr bool kPreloadActivation = PRELOAD_ACT_;
  static constexpr bool kScaleInSmem = SCALE_IN_SMEM_;

  // ---- thread / pipeline topology
  static constexpr uint32_t kWgsPerOutputTile = kNumMathWgs / kOutputTilesPerCta;
  static constexpr uint32_t kStages = kNumMathWgs * kStagesPerMathWg;
  static constexpr uint32_t kNumMathThreads = kNumMathWgs * 128;
  static constexpr uint32_t kNumThreads = (kNumMathWgs + 1) * 128;
  static constexpr uint32_t kNumMathThreadsPerOutputTile = kWgsPerOutputTile * 128;
  static constexpr uint32_t kNumProducerWarps = 4;

  // ---- shared-memory geometry (everything that does not depend on N/K/M)
  static constexpr uint32_t kSubtileWeightBytes = kOutputRows * kKBlock;
  static constexpr uint32_t kWeightBytesPerStage = kSubsPerStage * kSubtileWeightBytes;
  static constexpr uint32_t kWeightSmemBytes = kStages * kWeightBytesPerStage + kSmemPaddingBytes;
  // Worst case (runtime M == M_TILE) activation / transaction sizes.  The real
  // values are derived from runtime M below; these bound them and are what
  // max_dynamic_smem_bytes() uses.
  static constexpr uint32_t kMaxActBytesPerSubtile = kMTile * kKBlock;
  static constexpr uint32_t kMaxActBytesPerStage = kSubsPerStage * kMaxActBytesPerSubtile;
  static constexpr uint32_t kMaxTmaTxBytesPerStage =
      kSubsPerStage * kSubtileWeightBytes +
      (kPreloadActivation ? 0u : kSubsPerStage * kMaxActBytesPerSubtile);
  static constexpr uint32_t kPipelineBarrierBytes = 2 * kStages * sizeof(Barrier);
  static constexpr uint32_t kBarrierBytes =
      align_up_u32(kPipelineBarrierBytes + (kPreloadActivation ? sizeof(Barrier) : 0u), 16u);
  static constexpr uint32_t kPartialStride = kWgmmaRows * kMTile;
  static constexpr uint32_t kPartialBytes =
      kWgsPerOutputTile > 1 ? align_up_u32(kNumMathWgs * kPartialStride * sizeof(float), 16u) : 0u;
  static constexpr uint32_t kSplitKFlagBytes = kSplitKCta ? align_up_u32(kOutputTilesPerCta * 4u, 16u) : 0u;
  static constexpr uint32_t kFixedSmemBytes =
      kWeightSmemBytes + kBarrierBytes + kPartialBytes + kSplitKFlagBytes;
  static_assert(kFixedSmemBytes <= kMaxDynamicSmemBytes,
                "static shared-memory footprint already exceeds 227 KiB");

  // ---- runtime-shaped geometry
  static __host__ __device__ constexpr uint32_t act_rows(int m) {
    return m < static_cast<int>(kMTile) ? static_cast<uint32_t>(m) : kMTile;
  }
  static __host__ __device__ constexpr uint32_t act_bytes_per_subtile(int m) {
    return act_rows(m) * kKBlock;
  }
  static __host__ __device__ constexpr uint32_t act_bytes_per_stage(int m) {
    return kSubsPerStage * act_bytes_per_subtile(m);
  }
  // Bytes the producer announces to the `full` barrier per stage.  TMA counts
  // out-of-range (zero filled) rows as written bytes, so this is the box size.
  static __host__ __device__ constexpr uint32_t tma_tx_bytes_per_stage(int m) {
    return kSubsPerStage * kSubtileWeightBytes +
           (kPreloadActivation ? 0u : kSubsPerStage * act_bytes_per_subtile(m));
  }
  static __host__ __device__ constexpr uint32_t activation_smem_bytes(int m, int k) {
    return kPreloadActivation ? act_rows(m) * static_cast<uint32_t>(k)
                              : kStages * act_bytes_per_stage(m);
  }
  static __host__ __device__ constexpr uint32_t scale_smem_bytes(int k) {
    return kScaleInSmem
               ? align_up_u32(kOutputTilesPerCta * kMTile * kScaleSlotsPerOutputTile *
                                  (static_cast<uint32_t>(k) / kKBlock) * sizeof(float),
                              16u)
               : 0u;
  }
  static __host__ __device__ constexpr uint32_t dynamic_smem_bytes(int m, int k) {
    return kFixedSmemBytes + align_up_u32(activation_smem_bytes(m, k), 16u) + scale_smem_bytes(k);
  }
  // Upper bound over all legal runtime M (== dynamic_smem_bytes(M_TILE, k)).
  static __host__ __device__ constexpr uint32_t max_dynamic_smem_bytes(int k) {
    return kFixedSmemBytes + align_up_u32(kStages * kMaxActBytesPerStage, 16u) +
           scale_smem_bytes(k);
  }
  static __host__ __device__ constexpr uint32_t num_output_tiles(int n) {
    return ceil_div(static_cast<uint32_t>(n), kOutputRows);
  }
  static __host__ __device__ constexpr uint32_t num_ctas_x(int n) {
    return ceil_div(num_output_tiles(n), kOutputTilesPerCta);
  }
  // Total CTAs of one launch == gridDim.x * gridDim.y (gridDim.y == kSplitKFactor).
  static __host__ __device__ constexpr uint32_t num_ctas(int n) {
    return num_ctas_x(n) * kSplitKFactor;
  }
  // k chunks one CTA of one split has to stream (== its pipeline length).
  static __host__ __device__ constexpr uint32_t chunks_per_split(int k) {
    return static_cast<uint32_t>(k) / (kKBlock * kSubsPerStage * kSplitKFactor);
  }

  // ---- the divisibility rule, compile-time mirror of validate() ------------
  // Split-K needs K to be cuttable into S *equal* pipeline-friendly pieces:
  //     K % (128 * SUBS_PER_STAGE * WGsPerOutputTile * S) == 0
  // K is a runtime value here, so validate() enforces it per launch; callers
  // that know K at compile time (smoke_test, bench) can pin it with
  //     static_assert(Cfg::template k_ok_for_splitk<K>());
  template <uint32_t K>
  static constexpr bool k_ok_for_splitk() {
    return (K % (kKBlock * kSubsPerStage * kWgsPerOutputTile * kSplitKFactor)) == 0;
  }

  // Returns nullptr when the shape is supported, else a human readable reason.
  static const char* validate(int m, int n, int k) {
    if (m < 1 || static_cast<uint32_t>(m) > kMTile) return "M out of [1, M_TILE]";
    if (n < 1) return "N must be positive";
    if (k % kKBlock != 0) return "K must be a multiple of 128";
    const uint32_t num_k_blocks = static_cast<uint32_t>(k) / kKBlock;
    if (num_k_blocks % kSubsPerStage != 0) return "K/128 must be a multiple of SUBS_PER_STAGE";
    const uint32_t num_k_chunks = num_k_blocks / kSubsPerStage;
    if (num_k_chunks % kWgsPerOutputTile != 0)
      return "K/(128*SUBS) must be a multiple of WGsPerOutputTile";
    if (kSplitKCta) {
      // K % (128*SUBS*WGsPerTile*S) == 0, spelled out as the two divisibility
      // steps the pipeline actually performs.
      if (num_k_chunks % kSplitKFactor != 0)
        return "SPLIT_K: K/(128*SUBS) must be a multiple of SPLITK_FACTOR";
      if ((num_k_chunks / kSplitKFactor) % kWgsPerOutputTile != 0)
        return "SPLIT_K: (K/S)/(128*SUBS) must be a multiple of WGsPerOutputTile";
      // One *group* per warp group is enough to be correct (unused stages are
      // simply never waited on).  Requiring STAGES_PER_WG chunks instead would
      // forbid exactly the interesting corner -- S=8 on a short K, where the
      // per-CTA pipeline is shallower than the stage count -- and that corner
      // is one of the three accounts S has to pay (kernels/README.md §15).
      if ((num_k_chunks / kSplitKFactor) < kWgsPerOutputTile)
        return "SPLIT_K: not even one k group per warp group";
    }
    if (dynamic_smem_bytes(m, k) > kMaxDynamicSmemBytes)
      return "dynamic shared memory exceeds 227 KiB";
    return nullptr;
  }
};

// ---------------------------------------------------------------------------
// Runtime problem description (host side) / kernel argument (device side)
// ---------------------------------------------------------------------------
struct Problem {
  int M = 0;
  int N = 0;
  int K = 0;
  const __nv_fp8_e4m3* activation = nullptr;    // [M, K], K contiguous
  const __nv_fp8_e4m3* weight = nullptr;        // PREPACK ? packed : [N, K]
  const float* weight_scales = nullptr;         // [ceil(N/128), K/128], row major
  const float* activation_scales = nullptr;     // [M, K/128], row major
  __nv_bfloat16* output = nullptr;              // [M, N], N contiguous
  float* splitk_ws = nullptr;                   // SPLIT_K: [S][M][N] fp32, S = kSplitKFactor
  int* splitk_sem = nullptr;                    // SPLIT_K: >= num_output_tiles int32, zeroed once
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
  const char* error = nullptr;   // nullptr == usable
};

// ---------------------------------------------------------------------------
// Tensor maps
// ---------------------------------------------------------------------------
// Both weight layouts use a rank-4 map with box {128, BM, SUBS, 1}, so the TMA
// instruction itself is identical; only the meaning of the coordinates and the
// global strides differ.  In both cases one op lands SUBS*BM*128 contiguous
// bytes in smem in the [SUBS][BM][128] order the wgmma descriptors expect.
//
//   PREPACK : dims {128, BM, num_k_blocks, num_output_tiles}
//             strides {128, BM*128, num_k_blocks*BM*128}
//             coords (0, 0, chunk*SUBS, output_tile)
//             -> the whole (tile, SUBS k_blocks) chunk is contiguous in HBM.
//
//   raw     : dims {128, N, num_k_blocks, 1}
//             strides {K, 128, N*K}          <-- note: non monotonic on purpose
//             coords (0, output_tile*BM, chunk*SUBS, 0)
//             -> dim0/dim2 walk K (128 bytes per k_block, 128 bytes apart),
//                dim1 walks N.  This is what makes "one op = SUBS k_blocks"
//                possible without prepacking: CONTRACT 4.4 spells this as
//                "box {128*SUBS, BM, 1, 1}", but cuTensorMapEncodeTiled rejects
//                an inner box wider than the 128 B swizzle span
//                (CUDA_ERROR_INVALID_VALUE, verified with _dev/probe_tma.cu),
//                and a 256-byte row could not be described by a K-major SW128
//                wgmma descriptor anyway.  Splitting K into {128, num_k_blocks}
//                gives the identical single big op, byte-for-byte verified.
//
// The activation is a rank-3 map {128, M, num_k_blocks} with strides {K, 128},
// box {128, act_rows, SUBS}: rows >= M are out of range and get zero filled by
// TMA (still counted in the transaction bytes), which is exactly what the
// runtime M predicate on the math side expects.

template <typename Cfg>
inline CUtensorMap make_weight_tma(const Problem& p) {
  CUtensorMap tensor_map{};
  constexpr uint32_t kRank = 4;
  const uint32_t num_k_blocks = static_cast<uint32_t>(p.K) / kKBlock;
  const uint32_t num_output_tiles = Cfg::num_output_tiles(p.N);
  uint64_t global_dims[kRank];
  uint64_t global_strides[kRank - 1];
  if constexpr (Cfg::kPrepack) {
    global_dims[0] = kKBlock;
    global_dims[1] = Cfg::kOutputRows;
    global_dims[2] = num_k_blocks;
    global_dims[3] = num_output_tiles;
    global_strides[0] = kKBlock;                                     // row stride
    global_strides[1] = static_cast<uint64_t>(Cfg::kOutputRows) * kKBlock;
    global_strides[2] = static_cast<uint64_t>(num_k_blocks) * Cfg::kOutputRows * kKBlock;
  } else {
    global_dims[0] = kKBlock;
    global_dims[1] = static_cast<uint64_t>(p.N);
    global_dims[2] = num_k_blocks;
    global_dims[3] = 1;
    global_strides[0] = static_cast<uint64_t>(p.K);
    global_strides[1] = kKBlock;
    global_strides[2] = static_cast<uint64_t>(p.N) * static_cast<uint64_t>(p.K);
  }
  uint32_t box_dims[kRank] = {kKBlock, Cfg::kOutputRows, Cfg::kSubsPerStage, 1};
  uint32_t element_strides[kRank] = {1, 1, 1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
      const_cast<__nv_fp8_e4m3*>(p.weight), global_dims, global_strides, box_dims,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  fp8_swapab_wgmma::sm90_compat::check_driver_result(result, "cuTensorMapEncodeTiled(weight)");
  return tensor_map;
}

template <typename Cfg>
inline CUtensorMap make_activation_tma(const Problem& p) {
  CUtensorMap tensor_map{};
  constexpr uint32_t kRank = 3;
  const uint32_t num_k_blocks = static_cast<uint32_t>(p.K) / kKBlock;
  const uint32_t act_rows = Cfg::act_rows(p.M);
  uint64_t global_dims[kRank] = {kKBlock, static_cast<uint64_t>(p.M), num_k_blocks};
  uint64_t global_strides[kRank - 1] = {static_cast<uint64_t>(p.K), kKBlock};
  // PRELOAD_ACT pulls every k_block in one shot, otherwise SUBS per stage.
  const uint32_t box_k_blocks = Cfg::kPreloadActivation ? num_k_blocks : Cfg::kSubsPerStage;
  uint32_t box_dims[kRank] = {kKBlock, act_rows, box_k_blocks};
  uint32_t element_strides[kRank] = {1, 1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
      const_cast<__nv_fp8_e4m3*>(p.activation), global_dims, global_strides, box_dims,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  fp8_swapab_wgmma::sm90_compat::check_driver_result(result, "cuTensorMapEncodeTiled(activation)");
  return tensor_map;
}

// ---------------------------------------------------------------------------
// Small device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tma_load_4d(const CUtensorMap* map, uint64_t* barrier,
                                            void* smem, int32_t c1, int32_t c2, int32_t c3) {
  constexpr uint64_t kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  cute::SM90_TMA_LOAD_4D::copy(map, barrier, kCacheHint, smem, 0, c1, c2, c3);
}

__device__ __forceinline__ void tma_load_3d(const CUtensorMap* map, uint64_t* barrier,
                                            void* smem, int32_t c1, int32_t c2) {
  constexpr uint64_t kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  cute::SM90_TMA_LOAD_3D::copy(map, barrier, kCacheHint, smem, 0, c1, c2);
}

// PDL (programmatic dependent launch).  "This CTA is done producing work the
// next kernel could depend on" — the driver may then start the dependent grid's
// prologue while we are still draining.  Cost: ~nothing.  Worth: -3.2us/launch
// on a 16us kernel (v2 measurement).
__device__ __forceinline__ void pdl_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;");
}

// wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3, both operands in smem.
struct WgmmaM64N8K32 {
  __device__ static void wgmma(const uint64_t& desc_a, const uint64_t& desc_b, float* accum,
                               bool scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3}, %4, %5, p, 1, 1;\n"
        "}\n"
        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
        : "l"(desc_a), "l"(desc_b), "r"(static_cast<int32_t>(scale_d)));
  }
};

// ---------------------------------------------------------------------------
// Epilogue element iteration
// ---------------------------------------------------------------------------
// One math thread owns, per n8 wgmma group `mt`, four accumulators:
//   value[mt*4 + 0] = (row0, m = mt*8 + (lane&3)*2)
//   value[mt*4 + 1] = (row0, m + 1)
//   value[mt*4 + 2] = (row8, m)
//   value[mt*4 + 3] = (row8, m + 1)
// with row0 = warp*16 + lane/4 and row8 = row0 + 8.  The three epilogue flavours
// (direct bf16 store / intra-CTA split-K reduce / CTA split-K reduce) all walk
// exactly this set, so the walk lives here once and the bodies stay one-liners.
template <int MTiles>
struct EpilogueElements {
  float* value;                       // partial[MTiles][4], flattened
  uint32_t n_stride;                  // runtime N
  uint32_t row_base;                  // first output row (n) of this tile
  uint32_t row0;
  uint32_t row8;
  uint32_t m_base;                    // (lane & 3) * 2
  bool valid_row0;
  bool valid_row8;
  uint32_t valid_m0_mask;             // bit mt set -> m = mt*8 + m_base < M
  uint32_t valid_m1_mask;             // bit mt set -> m + 1 < M
};

template <int MTiles, typename Body>
__device__ __forceinline__ void for_each_output_element(const EpilogueElements<MTiles>& e,
                                                        Body body) {
#pragma unroll
  for (int mt = 0; mt < MTiles; ++mt) {
    const bool has_m0 = ((e.valid_m0_mask >> mt) & 1u) != 0u;
    const bool has_m1 = ((e.valid_m1_mask >> mt) & 1u) != 0u;
    if (!has_m0 && !has_m1) continue;
    const uint32_t m = static_cast<uint32_t>(mt) * kWgmmaCols + e.m_base;
    if (has_m0) {
      if (e.valid_row0) body(mt, 0, m, e.row0, e.row_base + e.row0, e.value[mt * 4 + 0]);
      if (e.valid_row8) body(mt, 2, m, e.row8, e.row_base + e.row8, e.value[mt * 4 + 2]);
    }
    if (has_m1) {
      if (e.valid_row0) body(mt, 1, m + 1, e.row0, e.row_base + e.row0, e.value[mt * 4 + 1]);
      if (e.valid_row8) body(mt, 3, m + 1, e.row8, e.row_base + e.row8, e.value[mt * 4 + 3]);
    }
  }
}

// ===========================================================================
//  The kernel
// ===========================================================================
//  Thread layout (identical to the oracle): warp groups [0, NUM_MATH_WG) do the
//  math, the last warp group is the TMA producer.  Right after the barrier init
//  the producer deallocates registers down to PROD_REGS and the math warp groups
//  allocate up to MATH_REGS (`warpgroup_reg_dealloc/alloc`) — that is what makes
//  a 168-register accumulator budget affordable next to a 4-warp producer.
//
//  Pipeline: kStages = NUM_MATH_WG * STAGES_PER_WG slots.  Stage rotation
//    stage      = (group % STAGES_PER_WG) * NUM_MATH_WG + math_wg
//    generation = group / STAGES_PER_WG
//  i.e. the stages of one warp group are strided by NUM_MATH_WG, so consecutive
//  groups of the same warp group land on different smem regions and different
//  mbarriers, and the four producer warps share the issue work by `stage % 4`.
//
//  Handshake per stage (v2 semantics, kept bit for bit):
//    producer: empty->wait((generation+1)&1)  ->  TMA copies  ->  full.arrive_and_expect_tx(bytes)
//    math    : full->wait(generation&1)       ->  wgmma + fold ->  empty->arrive() (lane 0)
// ===========================================================================
template <typename Cfg>
__global__ __launch_bounds__(Cfg::kNumThreads, 1)
void fp8_decode_gemm(__nv_bfloat16* __restrict__ output,
                     const float* __restrict__ weight_scales,
                     const float* __restrict__ activation_scales,
                     const __grid_constant__ CUtensorMap w_tma,
                     const __grid_constant__ CUtensorMap a_tma,
                     GemmRuntime rt) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t warp_idx = __shfl_sync(0xffffffffu, threadIdx.x / 32u, 0);
  const uint32_t lane_idx = threadIdx.x & 31u;

  // ---------------- runtime shape (N/K/M are *not* template parameters) ------
  const uint32_t m_rows = static_cast<uint32_t>(rt.M);
  const uint32_t n_cols = static_cast<uint32_t>(rt.N);
  const uint32_t num_k_blocks = static_cast<uint32_t>(rt.K) / kKBlock;
  const uint32_t num_scale_rows = ceil_div(n_cols, kKBlock);
  const uint32_t num_output_tiles = ceil_div(n_cols, Cfg::kOutputRows);
  const uint32_t num_k_chunks = num_k_blocks / Cfg::kSubsPerStage;
  // S6 SPLIT_K: gridDim.y == S (Cfg::kSplitKFactor), CTA y takes one contiguous
  // 1/S slice of the k chunks.  Every slice is a complete pipeline of its own,
  // so the stage rotation and the barrier generations simply restart at group 0
  // -- nothing downstream has to know that K was cut.  S == 1 folds both
  // expressions to the unsplit values at compile time (zero overhead).
  const uint32_t chunks_per_split =
      Cfg::kSplitKCta ? num_k_chunks / Cfg::kSplitKFactor : num_k_chunks;
  const uint32_t chunk_base = Cfg::kSplitKCta ? blockIdx.y * chunks_per_split : 0u;
  const uint32_t groups_per_math_wg = chunks_per_split / Cfg::kWgsPerOutputTile;
  const uint32_t act_rows = Cfg::act_rows(rt.M);
  const uint32_t act_bytes_per_subtile = act_rows * kKBlock;
  const uint32_t act_bytes_per_stage = Cfg::kSubsPerStage * act_bytes_per_subtile;
  const uint32_t tx_bytes_per_stage = Cfg::tma_tx_bytes_per_stage(rt.M);

  if (threadIdx.x == Cfg::kNumMathThreads) {
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&w_tma));
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&a_tma));
  }

  // ---------------- shared memory partition ---------------------------------
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* weight_base = reinterpret_cast<__nv_fp8_e4m3*>(smem);
  auto* activation_base =
      reinterpret_cast<__nv_fp8_e4m3*>(smem + Cfg::kWeightSmemBytes);
  const uint32_t activation_smem_bytes =
      align_up_u32(Cfg::kPreloadActivation ? act_rows * static_cast<uint32_t>(rt.K)
                                           : Cfg::kStages * act_bytes_per_stage,
                   16u);
  auto* combined_scales =
      reinterpret_cast<float*>(smem + Cfg::kWeightSmemBytes + activation_smem_bytes);
  auto* barriers = reinterpret_cast<Barrier*>(
      reinterpret_cast<uint8_t*>(combined_scales) + Cfg::scale_smem_bytes(rt.K));
  Barrier* activation_barrier = barriers + 2 * Cfg::kStages;
  auto* partials = reinterpret_cast<float*>(
      reinterpret_cast<uint8_t*>(barriers) + Cfg::kBarrierBytes);
  auto* splitk_flags = reinterpret_cast<float*>(
      reinterpret_cast<uint8_t*>(partials) + Cfg::kPartialBytes);

  // ---------------- barrier init (producer warp group) -----------------------
  if (threadIdx.x >= Cfg::kNumMathThreads && lane_idx == 0) {
    const uint32_t producer_warp = (threadIdx.x - Cfg::kNumMathThreads) / 32u;
#pragma unroll
    for (uint32_t stage = producer_warp; stage < Cfg::kStages; stage += Cfg::kNumProducerWarps) {
      (barriers + stage)->init(1);                       // full: 1 arrival (the TMA)
      (barriers + Cfg::kStages + stage)->init(4);        // empty: 4 math warps
    }
    if constexpr (Cfg::kPreloadActivation) {
      if (producer_warp == 0) activation_barrier->init(1);
    }
    cutlass::arch::fence_view_async_shared();
  }
  __syncthreads();

  // PRELOAD_ACT: one TMA for the whole activation, then it never moves again.
  if constexpr (Cfg::kPreloadActivation) {
    if (threadIdx.x == Cfg::kNumMathThreads) {
      activation_barrier->arrive_and_expect_tx(act_rows * static_cast<uint32_t>(rt.K));
      tma_load_3d(&a_tma, reinterpret_cast<uint64_t*>(activation_barrier), activation_base, 0, 0);
    }
    __syncthreads();
  }

  const uint32_t cta_output_tile_base = blockIdx.x * Cfg::kOutputTilesPerCta;

  // =========================================================================
  //  PRODUCER
  // =========================================================================
  if (threadIdx.x >= Cfg::kNumMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<Cfg::kProducerRegs>();
    const uint32_t producer_warp = (threadIdx.x - Cfg::kNumMathThreads) / 32u;
    if constexpr (Cfg::kSingleProducerWarp) {
      if (producer_warp != 0) return;
    }
    if (lane_idx == 0) {
#pragma unroll
      for (uint32_t local_tile = 0; local_tile < Cfg::kOutputTilesPerCta; ++local_tile) {
        const uint32_t output_tile = cta_output_tile_base + local_tile;
        if (output_tile >= num_output_tiles) continue;   // padding tile: no work
#pragma unroll 1
        for (uint32_t local_chunk = 0; local_chunk < chunks_per_split; ++local_chunk) {
          const uint32_t chunk = chunk_base + local_chunk;
          const uint32_t split_k_index = local_chunk % Cfg::kWgsPerOutputTile;
          const uint32_t math_wg = local_tile * Cfg::kWgsPerOutputTile + split_k_index;
          const uint32_t group = local_chunk / Cfg::kWgsPerOutputTile;
          const uint32_t stage =
              (group % Cfg::kStagesPerMathWg) * Cfg::kNumMathWgs + math_wg;
          if constexpr (!Cfg::kSingleProducerWarp) {
            if (stage % Cfg::kNumProducerWarps != producer_warp) continue;
          }
          const uint32_t generation = group / Cfg::kStagesPerMathWg;
          Barrier* empty = barriers + Cfg::kStages + stage;
          Barrier* full = barriers + stage;
          empty->wait((generation + 1u) & 1u);

          auto* weight_stage = weight_base + stage * Cfg::kWeightBytesPerStage;
          if constexpr (Cfg::kPrepack) {
            tma_load_4d(&w_tma, reinterpret_cast<uint64_t*>(full), weight_stage, 0,
                        static_cast<int32_t>(chunk * Cfg::kSubsPerStage),
                        static_cast<int32_t>(output_tile));
          } else {
            tma_load_4d(&w_tma, reinterpret_cast<uint64_t*>(full), weight_stage,
                        static_cast<int32_t>(output_tile * Cfg::kOutputRows),
                        static_cast<int32_t>(chunk * Cfg::kSubsPerStage), 0);
          }
          if constexpr (!Cfg::kPreloadActivation) {
            tma_load_3d(&a_tma, reinterpret_cast<uint64_t*>(full),
                        activation_base + stage * act_bytes_per_stage, 0,
                        static_cast<int32_t>(chunk * Cfg::kSubsPerStage));
          }
          // expect_tx *after* the copies: the mbarrier tx-count is signed, so
          // announcing late is fine and keeps the critical path shorter.
          full->arrive_and_expect_tx(tx_bytes_per_stage);
        }
      }
    }
    // S1 PDL: every TMA this CTA will ever issue is now in flight.  The next
    // kernel in the stream may start its prologue (descriptor prefetch, barrier
    // init, scale merge) while we are still streaming weights.
    if constexpr (Cfg::kPdl) {
      if (threadIdx.x == Cfg::kNumMathThreads) pdl_launch_dependents();
    }
    return;
  }

  // =========================================================================
  //  MATH
  // =========================================================================
  cutlass::arch::warpgroup_reg_alloc<Cfg::kMathRegisters>();

  const uint32_t math_wg = warp_idx / 4u;
  const uint32_t local_warp = warp_idx & 3u;
  const uint32_t local_output_tile = math_wg / Cfg::kWgsPerOutputTile;
  const uint32_t split_k_index = math_wg - local_output_tile * Cfg::kWgsPerOutputTile;
  const uint32_t output_tile = cta_output_tile_base + local_output_tile;
  const bool output_tile_active = output_tile < num_output_tiles;
  // Padded tiles still run the pipeline (the producer skips them, so they must
  // not wait); every index used for a *load* is clamped to a legal tile/group,
  // every index used for a *store* is predicated by output_tile_active.
  const uint32_t clamped_output_tile =
      output_tile_active ? output_tile : num_output_tiles - 1u;
  const uint32_t output_row_base = output_tile * Cfg::kOutputRows;
  const uint32_t clamped_output_row_base = clamped_output_tile * Cfg::kOutputRows;

  // row0/row8: the two output rows (n) this thread's accumulators belong to.
  const uint32_t row0 = local_warp * 16u + lane_idx / 4u;
  const uint32_t row8 = row0 + 8u;
  const bool valid_row0 =
      output_tile_active && row0 < Cfg::kOutputRows && (output_row_base + row0) < n_cols;
  const bool valid_row8 =
      output_tile_active && row8 < Cfg::kOutputRows && (output_row_base + row8) < n_cols;
  // m columns of the n8 B operand this lane owns.
  const uint32_t logical_m_base = (lane_idx & 3u) * 2u;

  // Scale bookkeeping: an output tile of <= 64 rows touches at most two 128-row
  // scale groups; slot = group - group0 in {0,1}.  Clamped so that padded tiles
  // never read out of bounds even though their results are thrown away.
  const uint32_t scale_group0 = clamped_output_row_base / kKBlock;
  const uint32_t slot0 = ((clamped_output_row_base + row0) / kKBlock) - scale_group0;
  const uint32_t slot8 = ((clamped_output_row_base + row8) / kKBlock) - scale_group0;
  const uint32_t w_group0 = min(scale_group0 + slot0, num_scale_rows - 1u);
  const uint32_t w_group8 = min(scale_group0 + slot8, num_scale_rows - 1u);

  const uint32_t scales_per_logical_row = kScaleSlotsPerOutputTile * num_k_blocks;
  const uint32_t scales_per_tile = Cfg::kMTile * scales_per_logical_row;
  const uint32_t scale_tile_offset = local_output_tile * scales_per_tile;
  const float* w_scale_row0 = weight_scales + static_cast<size_t>(w_group0) * num_k_blocks;
  const float* w_scale_row8 = weight_scales + static_cast<size_t>(w_group8) * num_k_blocks;

  // ---------------- prologue: merge w_scale * a_scale into smem --------------
  // "Why": the wgmma accumulator is raw fp8xfp8 products; the dequantisation
  // factor is (w_scale[n/128][k/128] * a_scale[m][k/128]) and only changes
  // every 128 k elements.  Multiplying the two once per (tile, m, slot, k_block)
  // in the prologue turns the inner loop into one shared-memory read + one fmaf
  // per k_block.  At M_TILE>=32 the table no longer fits the smem budget, so the
  // SCALE_IN_SMEM=false path reads both scales through L1 (__ldg) instead.
  if constexpr (Cfg::kScaleInSmem) {
    // Split-K 只填自己那 1/S 段 k_block。老代码把整段 K 的表都合并一遍，于是
    // prologue 成本**不随 S 缩小**、而 CTA 数 ∝ S -> 全网格的 prologue 总量 ∝ S
    // （S=8/N=2048 时 512 个 CTA 各合并 32 个 k_block，其中 31/32 是用不到的）。
    // 主循环只会读 [kb_lo, kb_lo + chunks_per_split*SUBS)，所以缩小填充范围是
    // 等价的；S=1 时 kb_lo=0、kb_cnt=num_k_blocks，下标退化成原来的 i，逐字相同。
    const uint32_t kb_lo = chunk_base * Cfg::kSubsPerStage;
    const uint32_t kb_cnt =
        Cfg::kSplitKCta ? chunks_per_split * Cfg::kSubsPerStage : num_k_blocks;
    const uint32_t row_split = kScaleSlotsPerOutputTile * kb_cnt;   // 每 m
    const uint32_t tile_split = Cfg::kMTile * row_split;            // 每 output tile
    const uint32_t scales_per_cta = Cfg::kOutputTilesPerCta * tile_split;
    for (uint32_t i = threadIdx.x; i < scales_per_cta; i += Cfg::kNumMathThreads) {
      const uint32_t local_tile = i / tile_split;
      const uint32_t in_tile = i - local_tile * tile_split;
      const uint32_t m = in_tile / row_split;
      const uint32_t rest = in_tile - m * row_split;
      const uint32_t slot = rest / kb_cnt;
      const uint32_t k_block = kb_lo + (rest - slot * kb_cnt);
      const uint32_t tile = min(cta_output_tile_base + local_tile, num_output_tiles - 1u);
      const uint32_t group = min(
          (tile * Cfg::kOutputRows + slot * (Cfg::kOutputRows - 1u)) / kKBlock,
          num_scale_rows - 1u);
      const uint32_t m_clamped = min(m, m_rows - 1u);   // runtime M <= M_TILE
      const float combined =
          __ldg(weight_scales + static_cast<size_t>(group) * num_k_blocks + k_block) *
          __ldg(activation_scales + static_cast<size_t>(m_clamped) * num_k_blocks + k_block);
      // smem 里仍按**绝对** k_block 摆（表的大小/布局不变，只是稀疏填充）。
      // S=1 时 kb_cnt==num_k_blocks、tile_split==scales_per_tile，绝对下标恒等于
      // 枚举下标 i —— 直接写 i，S0..S5 的生成代码与改动前逐字相同。
      const uint32_t smem_index =
          Cfg::kSplitKCta ? (local_tile * scales_per_tile +
                             m * scales_per_logical_row + slot * num_k_blocks +
                             k_block)
                          : i;
      st_shared(combined_scales + smem_index, combined);
    }
    cutlass::arch::NamedBarrier(Cfg::kNumMathThreads, 0).sync();
  }
  if constexpr (Cfg::kPreloadActivation) {
    activation_barrier->wait(0);
  }

  // Which (mt, m) pairs are real for this runtime M.
  uint32_t valid_m0_mask = 0u;
  uint32_t valid_m1_mask = 0u;
#pragma unroll
  for (int mt = 0; mt < static_cast<int>(Cfg::kMTiles); ++mt) {
    const uint32_t m0 = static_cast<uint32_t>(mt) * kWgmmaCols + logical_m_base;
    if (m0 < m_rows) valid_m0_mask |= (1u << mt);
    if (m0 + 1u < m_rows) valid_m1_mask |= (1u << mt);
  }

  float partial[Cfg::kMTiles][4];
#pragma unroll
  for (int mt = 0; mt < static_cast<int>(Cfg::kMTiles); ++mt) {
#pragma unroll
    for (int i = 0; i < 4; ++i) partial[mt][i] = 0.0f;
  }

  // ---------------- main loop ------------------------------------------------
  if (output_tile_active) {
#pragma unroll 1
    for (uint32_t group = 0; group < groups_per_math_wg; ++group) {
      const uint32_t chunk = chunk_base + group * Cfg::kWgsPerOutputTile + split_k_index;
      const uint32_t stage =
          (group % Cfg::kStagesPerMathWg) * Cfg::kNumMathWgs + math_wg;
      const uint32_t phase = (group / Cfg::kStagesPerMathWg) & 1u;
      Barrier* full = barriers + stage;
      Barrier* empty = barriers + Cfg::kStages + stage;
      auto* weight_stage = weight_base + stage * Cfg::kWeightBytesPerStage;
      full->wait(phase);

      // One k_block = 4 x wgmma k32, one accumulator set per (sub, mt).
      // raw[][] is *not* pre-zeroed: the first k32 of every wgmma group passes
      // scale_d=false, which makes wgmma overwrite the accumulator instead of
      // adding to it.  Register cost = SUBS * M_TILES * 4, which is why
      // M_TILE=128 has to run with SUBS=1 (128 accumulator registers).
      float raw[Cfg::kSubsPerStage][Cfg::kMTiles][4];
#pragma unroll
      for (uint32_t sub = 0; sub < Cfg::kSubsPerStage; ++sub) {
        auto* act_sub =
            Cfg::kPreloadActivation
                ? activation_base + (chunk * Cfg::kSubsPerStage + sub) * act_bytes_per_subtile
                : activation_base + stage * act_bytes_per_stage +
                      sub * act_bytes_per_subtile;
#pragma unroll
        for (int mt = 0; mt < static_cast<int>(Cfg::kMTiles); ++mt)
#pragma unroll
          for (int i = 0; i < 4; ++i) warpgroup_fence_operand(raw[sub][mt][i]);
        warpgroup_arrive();
#pragma unroll
        for (uint32_t k32 = 0; k32 < kK32PerBlock; ++k32) {
          const uint64_t a_desc = make_smem_desc(
              weight_stage + sub * Cfg::kSubtileWeightBytes + k32 * 32u, 1, 0,
              kWgmmaCoreStrideBytes);
#pragma unroll
          for (int mt = 0; mt < static_cast<int>(Cfg::kMTiles); ++mt) {
            const uint64_t b_desc =
                make_smem_desc(act_sub + static_cast<uint32_t>(mt) * kWgmmaCols * kKBlock +
                                   k32 * 32u,
                               1, 0, kWgmmaCoreStrideBytes);
            WgmmaM64N8K32::wgmma(a_desc, b_desc, raw[sub][mt], k32 != 0u);
          }
        }
        warpgroup_commit_batch();
#pragma unroll
        for (int mt = 0; mt < static_cast<int>(Cfg::kMTiles); ++mt)
#pragma unroll
          for (int i = 0; i < 4; ++i) warpgroup_fence_operand(raw[sub][mt][i]);
      }

      // One commit group per sub; wait for all of them before touching raw.
#pragma unroll
      for (uint32_t sub = 0; sub < Cfg::kSubsPerStage; ++sub)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int mt = 0; mt < static_cast<int>(Cfg::kMTiles); ++mt)
            warpgroup_fence_operand(raw[sub][mt][i]);
      warpgroup_wait<0>();
#pragma unroll
      for (uint32_t sub = 0; sub < Cfg::kSubsPerStage; ++sub)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int mt = 0; mt < static_cast<int>(Cfg::kMTiles); ++mt)
            warpgroup_fence_operand(raw[sub][mt][i]);

      // ---------------- dequantise + accumulate (v2 numerical path) ----------
#pragma unroll
      for (uint32_t sub = 0; sub < Cfg::kSubsPerStage; ++sub) {
        const uint32_t k_block = chunk * Cfg::kSubsPerStage + sub;
        float w0 = 0.0f;
        float w8 = 0.0f;
        if constexpr (!Cfg::kScaleInSmem) {
          w0 = __ldg(w_scale_row0 + k_block);
          w8 = __ldg(w_scale_row8 + k_block);
        }
#pragma unroll
        for (int mt = 0; mt < static_cast<int>(Cfg::kMTiles); ++mt) {
          const uint32_t m0 = static_cast<uint32_t>(mt) * kWgmmaCols + logical_m_base;
          if (((valid_m0_mask >> mt) & 1u) != 0u) {
            float c0;
            float c8;
            if constexpr (Cfg::kScaleInSmem) {
              const float* s = combined_scales + scale_tile_offset +
                               m0 * scales_per_logical_row + k_block;
              c0 = ld_shared(s + slot0 * num_k_blocks);
              c8 = ld_shared(s + slot8 * num_k_blocks);
            } else {
              const float a = __ldg(activation_scales +
                                    static_cast<size_t>(m0) * num_k_blocks + k_block);
              c0 = w0 * a;
              c8 = w8 * a;
            }
            partial[mt][0] = fmaf(c0, raw[sub][mt][0], partial[mt][0]);
            partial[mt][2] = fmaf(c8, raw[sub][mt][2], partial[mt][2]);
          }
          if (((valid_m1_mask >> mt) & 1u) != 0u) {
            const uint32_t m1 = m0 + 1u;
            float c0;
            float c8;
            if constexpr (Cfg::kScaleInSmem) {
              const float* s = combined_scales + scale_tile_offset +
                               m1 * scales_per_logical_row + k_block;
              c0 = ld_shared(s + slot0 * num_k_blocks);
              c8 = ld_shared(s + slot8 * num_k_blocks);
            } else {
              const float a = __ldg(activation_scales +
                                    static_cast<size_t>(m1) * num_k_blocks + k_block);
              c0 = w0 * a;
              c8 = w8 * a;
            }
            partial[mt][1] = fmaf(c0, raw[sub][mt][1], partial[mt][1]);
            partial[mt][3] = fmaf(c8, raw[sub][mt][3], partial[mt][3]);
          }
        }
      }
      if (lane_idx == 0) empty->arrive();
    }
  }

  // =========================================================================
  //  EPILOGUE
  // =========================================================================
  // S5 EPI_OVERLAP — "why a *second* trigger": the producer-side trigger of S1
  // already lets the next kernel build its tensor maps / init its barriers while
  // we stream.  But the tail of *this* kernel (the bf16 stores, and for
  // SPLIT_K_CTA also the fp32 workspace round trip) still runs alone.  Emitting
  // launch_dependents a second time, right before the stores, hands the SMs over
  // to the dependent grid's prologue while our epilogue drains.
  if constexpr (Cfg::kEpiOverlap) pdl_launch_dependents();

  EpilogueElements<static_cast<int>(Cfg::kMTiles)> elements;
  elements.value = &partial[0][0];
  elements.n_stride = n_cols;
  elements.row_base = output_row_base;
  elements.row0 = row0;
  elements.row8 = row8;
  elements.m_base = logical_m_base;
  elements.valid_row0 = valid_row0;
  elements.valid_row8 = valid_row8;
  elements.valid_m0_mask = valid_m0_mask;
  elements.valid_m1_mask = valid_m1_mask;

  // ---- Phase A: intra-CTA split-K reduce ------------------------------------
  // kWgsPerOutputTile warp groups each accumulated a different slice of K for
  // the *same* output tile.  They cannot talk through registers, so the slices
  // go through a small smem buffer (64 rows * M_TILE floats per warp group) and
  // one named barrier; warp group 0 of the tile folds them together.
  if constexpr (Cfg::kWgsPerOutputTile > 1) {
    if (output_tile_active) {
      float* mine = partials + math_wg * Cfg::kPartialStride;
      for_each_output_element(
          elements, [&](int, int, uint32_t m, uint32_t row, uint32_t, float v) {
            st_shared(mine + m * kWgmmaRows + row, v);
          });
    }
    cutlass::arch::NamedBarrier(Cfg::kNumMathThreadsPerOutputTile, 1u + local_output_tile)
        .sync();
    if (split_k_index == 0u && output_tile_active) {
      float* owner_base =
          partials + local_output_tile * Cfg::kWgsPerOutputTile * Cfg::kPartialStride;
      for_each_output_element(
          elements, [&](int mt, int slot, uint32_t m, uint32_t row, uint32_t, float) {
            float total = ld_shared(owner_base + m * kWgmmaRows + row);
#pragma unroll
            for (uint32_t owner = 1u; owner < Cfg::kWgsPerOutputTile; ++owner) {
              total += ld_shared(owner_base + owner * Cfg::kPartialStride + m * kWgmmaRows + row);
            }
            elements.value[mt * 4 + slot] = total;
          });
    }
  }

  // Only the warp group that owns the reduced result may touch global memory.
  const bool reduce_owner = (Cfg::kWgsPerOutputTile == 1u) || (split_k_index == 0u);

  if constexpr (!Cfg::kSplitKCta) {
    // ---- Phase B (plain): one bf16 store per element ------------------------
    if (reduce_owner) {
      for_each_output_element(
          elements, [&](int, int, uint32_t m, uint32_t, uint32_t n, float v) {
            output[static_cast<size_t>(m) * n_cols + n] = __float2bfloat16_rn(v);
          });
    }
  } else if (output_tile_active) {
    // ---- Phase B (S6 SPLIT_K): S CTAs, one output tile ----------------------
    // "Why": the number of concurrent DRAM streams is N/BM (times gridDim.y),
    // and cold-data bandwidth is directly proportional to it (v2 measured 36%
    // at 78 CTAs, 60% at 154, 83% at 308 for 10 KiB ops).  Small N (e.g. 2048
    // -> 32 tiles at BM=64) simply cannot fill 78 SMs along N, so we cut K into
    // S contiguous slices across gridDim.y instead and reduce the S fp32
    // partials in gmem.  S is Cfg::kSplitKFactor (2/4/8); S=2 is what the
    // original SPLIT_K_CTA bool meant, and the code below reduces to exactly
    // the old two-slab handshake for that case.
    //
    // Handshake: write own partial -> __threadfence (release) -> tile barrier ->
    // one atomicAdd on a per-tile semaphore -> the CTA that sees "S-1" is last,
    // fences (acquire), adds the other S-1 partials and stores bf16, then resets
    // the semaphore to 0 so the next launch needs no host-side clearing.
    //
    // Cost model (kernels/README.md §15): the workspace traffic grows as
    // S*M*N*4 B written + (S-1)*M*N*4 B read back, i.e. linearly in S *and* in
    // M -- which is why split-K is an M=1..8 technique and dies at large M.
    const size_t split_stride = static_cast<size_t>(rt.M) * n_cols;
    float* ws_self = rt.splitk_ws + blockIdx.y * split_stride;
    if (reduce_owner) {
      for_each_output_element(
          elements, [&](int, int, uint32_t m, uint32_t, uint32_t n, float v) {
            ws_self[static_cast<size_t>(m) * n_cols + n] = v;
          });
    }
    __threadfence();
    const uint32_t handshake_barrier = Cfg::kOutputTilesPerCta + 1u + local_output_tile;
    cutlass::arch::NamedBarrier(Cfg::kNumMathThreadsPerOutputTile, handshake_barrier).sync();

    const bool tile_leader =
        (math_wg == local_output_tile * Cfg::kWgsPerOutputTile) && local_warp == 0u &&
        lane_idx == 0u;
    if (tile_leader) {
      // Ticket: exactly one CTA per output tile observes previous == S-1.
      const int previous = atomicAdd(rt.splitk_sem + output_tile, 1);
      st_shared(splitk_flags + local_output_tile,
                previous == static_cast<int>(Cfg::kSplitKFactor - 1u) ? 1.0f : 0.0f);
    }
    cutlass::arch::NamedBarrier(Cfg::kNumMathThreadsPerOutputTile, handshake_barrier).sync();

    if (ld_shared(splitk_flags + local_output_tile) != 0.0f) {
      __threadfence();
      if (reduce_owner) {
        for_each_output_element(
            elements, [&](int, int, uint32_t m, uint32_t, uint32_t n, float v) {
              const size_t index = static_cast<size_t>(m) * n_cols + n;
              float total = v;                     // own slice, still in a register
#pragma unroll
              for (uint32_t s = 1u; s < Cfg::kSplitKFactor; ++s) {
                // rotate the peer order so that the S-1 reducers of a wave do
                // not all hammer slab 0 first (s == S-1 -> peer == self, skipped
                // by construction: s only runs to S-1).
                const uint32_t peer = (blockIdx.y + s) % Cfg::kSplitKFactor;
                total += rt.splitk_ws[peer * split_stride + index];
              }
              output[index] = __float2bfloat16_rn(total);
            });
      }
      if (tile_leader) atomicExch(rt.splitk_sem + output_tile, 0);   // self-resetting
    }
  }
#else
  if (blockIdx.x == 0 && threadIdx.x == 0) asm volatile("trap;");
#endif
}

// ===========================================================================
//  Host side
// ===========================================================================
template <typename Cfg>
inline LaunchPlan make_launch_plan(const Problem& p) {
  LaunchPlan plan;
  plan.error = Cfg::validate(p.M, p.N, p.K);
  plan.num_output_tiles = static_cast<int>(Cfg::num_output_tiles(p.N));
  plan.padded_n = plan.num_output_tiles * static_cast<int>(Cfg::kOutputRows);
  plan.grid = dim3(Cfg::num_ctas_x(p.N), Cfg::kSplitKFactor, 1u);
  plan.block = dim3(Cfg::kNumThreads, 1u, 1u);
  plan.smem_bytes = Cfg::dynamic_smem_bytes(p.M, p.K);
  return plan;
}

// cudaFuncSetAttribute(MaxDynamicSharedMemorySize) is sticky per kernel, but the
// smem footprint depends on runtime M/K, so it is refreshed whenever it changes.
template <typename Cfg>
inline cudaError_t ensure_smem_attribute(uint32_t smem_bytes) {
  static std::atomic<int> cached{-1};
  if (cached.load(std::memory_order_relaxed) == static_cast<int>(smem_bytes)) return cudaSuccess;
  const cudaError_t err = cudaFuncSetAttribute(
      fp8_decode_gemm<Cfg>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (err == cudaSuccess) cached.store(static_cast<int>(smem_bytes), std::memory_order_relaxed);
  return err;
}

// Launch with pre-built tensor maps (use this inside timing loops: encoding a
// tensor map is a host-side driver call and must stay out of the timed region).
template <typename Cfg>
inline cudaError_t launch_with_maps(const Problem& p, const CUtensorMap& w_tma,
                                    const CUtensorMap& a_tma, cudaStream_t stream) {
  const LaunchPlan plan = make_launch_plan<Cfg>(p);
  if (plan.error != nullptr) return cudaErrorInvalidValue;
  if (Cfg::kSplitKCta && (p.splitk_ws == nullptr || p.splitk_sem == nullptr)) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t attr_err = ensure_smem_attribute<Cfg>(plan.smem_bytes);
  if (attr_err != cudaSuccess) return attr_err;

  GemmRuntime rt;
  rt.M = p.M;
  rt.N = p.N;
  rt.K = p.K;
  rt.splitk_ws = p.splitk_ws;
  rt.splitk_sem = p.splitk_sem;

  cudaLaunchConfig_t config{};
  cudaLaunchAttribute attributes[1]{};
  config.gridDim = plan.grid;
  config.blockDim = plan.block;
  config.dynamicSmemBytes = plan.smem_bytes;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 0;
  if constexpr (Cfg::kPdl) {
    // S1 PDL: allow this kernel to start before its predecessor has drained.
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&config, fp8_decode_gemm<Cfg>, p.output, p.weight_scales,
                            p.activation_scales, w_tma, a_tma, rt);
}

// Convenience: encode both tensor maps, then launch.
template <typename Cfg>
inline cudaError_t launch(const Problem& p, cudaStream_t stream = nullptr) {
  if (Cfg::validate(p.M, p.N, p.K) != nullptr) return cudaErrorInvalidValue;
  if (Cfg::kSplitKCta && (p.splitk_ws == nullptr || p.splitk_sem == nullptr)) {
    return cudaErrorInvalidValue;
  }
  const CUtensorMap w_tma = make_weight_tma<Cfg>(p);
  const CUtensorMap a_tma = make_activation_tma<Cfg>(p);
  return launch_with_maps<Cfg>(p, w_tma, a_tma, stream);
}

}  // namespace decode_gemm
