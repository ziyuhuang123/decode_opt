// SPDX-License-Identifier: MIT
//
// decode_gemm weight layouts + the raw -> tile-major prepack kernel.
//
// Two layouts are supported by the GEMM kernel (compile-time switch PREPACK):
//
//   raw     weight[N, K]  fp8 e4m3, K contiguous.
//           This is what a checkpoint gives you: zero preprocessing, but the
//           K-contiguous segment a CTA needs is chopped into BM rows that are
//           K bytes apart, so one TMA op can only cover `SUBS` 128-byte pieces
//           per row.
//
//   packed  packed[(tile * num_k_blocks + k_block) * BM * 128
//                  + row_in_tile * 128 + (k % 128)]
//           i.e. for every (output tile, k_block) pair, BM rows x 128 bytes
//           live in one contiguous chunk.  Everything a CTA reads for one
//           pipeline stage is then one contiguous SUBS*BM*128 byte block, so a
//           single TMA op moves a whole stage.  Rows beyond N (padding of the
//           last tile up to a multiple of BM) are zero filled.
//
// The layout only depends on BM (= OutputRowsPerCta); TILES_PER_CTA is part of
// the template signature for symmetry with the GEMM config but does not change
// the byte layout.
//
// Generalized from the reference `swapab_m1_weight_prepack_v2.cuh`
// (DSA-learn/mk/h20/cutedsl/shared): N and K are runtime values here instead of
// template parameters, so one instantiation serves every model shape.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace decode_gemm {
namespace prepack {

inline constexpr uint32_t kKBlock = 128;              // scale group / wgmma k32*4
inline constexpr uint32_t kBytesPerUint4 = 16;
inline constexpr uint32_t kUint4PerKBlock = kKBlock / kBytesPerUint4;   // 8

// Number of output rows stored in the packed buffer (N rounded up to BM).
inline int padded_n(int n, int output_rows_per_cta) {
  return ((n + output_rows_per_cta - 1) / output_rows_per_cta) * output_rows_per_cta;
}

// Byte size of the packed weight buffer.
inline size_t packed_weight_bytes(int n, int k, int output_rows_per_cta) {
  return static_cast<size_t>(padded_n(n, output_rows_per_cta)) * static_cast<size_t>(k);
}

// One uint4 (16 bytes) per thread: raw[n][k] -> packed[tile][k_block][row][k%128].
template <int OutputRowsPerCta, int OutputTilesPerCta = 1>
__global__ void prepack_weight_u4(const uint4* __restrict__ source,
                                  uint4* __restrict__ destination,
                                  int n,
                                  int k,
                                  int num_output_tiles,
                                  int num_k_blocks) {
  static_assert(OutputRowsPerCta > 0 && OutputRowsPerCta % 8 == 0,
                "OutputRowsPerCta must be a positive multiple of 8");
  static_assert(OutputTilesPerCta > 0, "OutputTilesPerCta must be positive");

  // One uint4 (16 B) per thread: a weight row of K fp8 bytes is K/16 vectors,
  // each k_block holds kUint4PerKBlock = 8 of them.
  const uint32_t vectors_per_row = static_cast<uint32_t>(k) / kBytesPerUint4;
  const uint32_t total_vectors =
      static_cast<uint32_t>(num_output_tiles) * static_cast<uint32_t>(OutputRowsPerCta) *
      vectors_per_row;   // == padded_n * K / 16

  const uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total_vectors) return;

  const uint32_t row_global = linear / vectors_per_row;          // n index (may be padded)
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

  // Padding rows (row_global >= n) must be zero: the GEMM predicates them out,
  // but zero keeps the smem/wgmma input well defined (no NaN garbage).
  uint4 value = {0u, 0u, 0u, 0u};
  if (row_global < static_cast<uint32_t>(n)) {
    value = source[static_cast<size_t>(row_global) * vectors_per_row + k_vector];
  }
  destination[packed_vector] = value;
}

// Host launcher.  `source` is the raw [n, k] fp8 buffer, `destination` must have
// packed_weight_bytes(n, k, OutputRowsPerCta) bytes allocated.
template <int OutputRowsPerCta, int OutputTilesPerCta = 1>
inline cudaError_t launch_prepack_weight(const __nv_fp8_e4m3* source,
                                         __nv_fp8_e4m3* destination,
                                         int n,
                                         int k,
                                         cudaStream_t stream = nullptr) {
  if (k % kKBlock != 0) return cudaErrorInvalidValue;
  const int num_output_tiles = (n + OutputRowsPerCta - 1) / OutputRowsPerCta;
  const int num_k_blocks = k / kKBlock;
  const size_t total_vectors =
      static_cast<size_t>(num_output_tiles) * OutputRowsPerCta * (k / kBytesPerUint4);
  constexpr uint32_t kThreads = 256;
  const uint32_t blocks =
      static_cast<uint32_t>((total_vectors + kThreads - 1) / kThreads);
  prepack_weight_u4<OutputRowsPerCta, OutputTilesPerCta><<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(source),
      reinterpret_cast<uint4*>(destination),
      n, k, num_output_tiles, num_k_blocks);
  return cudaGetLastError();
}

}  // namespace prepack
}  // namespace decode_gemm
