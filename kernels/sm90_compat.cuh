// SPDX-License-Identifier: MIT
//
// SM90 low-level wrappers (shared-memory ld/st, wgmma descriptor, wgmma
// fence/commit/wait, 2D TMA copy, driver error check).
//
// Verbatim copy of
//   DSA-learn/mk/h20/cutedsl/shared/fp8_swapab_wgmma/common/sm90_compat.cuh
// (the reference implementation this project is derived from).  Kept in the
// decode_gemm tree so that kernels/ is self-contained; namespace unchanged.

#pragma once

#include <cuda.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cute/arch/copy_sm90_tma.hpp>

namespace fp8_swapab_wgmma::sm90_compat {

inline void check_driver_result(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;

  const char* error_name = "unknown CUDA driver error";
  const char* error_description = "no description";
  (void)cuGetErrorName(result, &error_name);
  (void)cuGetErrorString(result, &error_description);
  throw std::runtime_error(
      std::string(operation) + " failed: " + error_name + " (" +
      error_description + ")");
}

__device__ __forceinline__ float ld_shared(
    const float* __restrict__ pointer) {
  float value;
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
  asm volatile("ld.shared.f32 %0, [%1];"
               : "=f"(value)
               : "r"(shared_address)
               : "memory");
  return value;
}

__device__ __forceinline__ void st_shared(
    float* pointer,
    float value) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
  asm volatile("st.shared.f32 [%0], %1;"
               :
               : "r"(shared_address), "f"(value)
               : "memory");
}

__device__ __forceinline__ void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void warpgroup_fence_operand(float& value) {
  asm volatile("" : "+f"(value) : : "memory");
}

template <int PendingGroups>
__device__ __forceinline__ void warpgroup_wait() {
  static_assert(
      PendingGroups >= 0 && PendingGroups <= 7,
      "WGMMA wait group count must be in [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;"
               :
               : "n"(PendingGroups)
               : "memory");
}

union GmmaDescriptor {
  __host__ __device__ constexpr GmmaDescriptor() noexcept : bits(0) {}

  uint64_t bits;
  struct {
    uint16_t start_address : 14, : 2;
    uint16_t leading_byte_offset : 14, : 2;
    uint16_t stride_byte_offset : 14, : 2;
    uint8_t : 1, base_offset : 3, : 4;
    uint8_t : 6, layout_type : 2;
  } fields;

  __host__ __device__ constexpr operator uint64_t() const noexcept {
    return bits;
  }
};

static_assert(sizeof(GmmaDescriptor) == sizeof(uint64_t));

template <class Pointer>
__device__ __forceinline__ GmmaDescriptor make_smem_desc(
    Pointer pointer,
    int layout_type,
    int leading_byte_offset = 0,
    int stride_byte_offset = 1024) {
  GmmaDescriptor descriptor;
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
  descriptor.fields.start_address = shared_address >> 4;
  descriptor.fields.layout_type = layout_type;
  descriptor.fields.leading_byte_offset = leading_byte_offset >> 4;
  descriptor.fields.stride_byte_offset = stride_byte_offset >> 4;
  descriptor.fields.base_offset = 0;
  return descriptor;
}

__device__ __forceinline__ void tma_copy_2d(
    const void* descriptor,
    uint64_t* barrier,
    void* shared_destination,
    int32_t coordinate_0,
    int32_t coordinate_1) {
  constexpr uint64_t kCacheHint =
      static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  cute::SM90_TMA_LOAD_2D::copy(
      descriptor,
      barrier,
      kCacheHint,
      shared_destination,
      coordinate_0,
      coordinate_1);
}

}  // namespace fp8_swapab_wgmma::sm90_compat
