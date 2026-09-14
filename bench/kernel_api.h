// SPDX-License-Identifier: MIT
//
// bench/kernel_api.h —— kernel 头的唯一入口。
//
//   make            -> kernels/decode_gemm.cuh (+ kernels/weight_prepack.cuh)   [真 kernel]
//   make stub       -> bench/stub_kernel.h                                      [流程自测]
//
// 切换开关：-DBENCH_USE_STUB（只有 Makefile 的 stub 目标会加）。
#pragma once

#if defined(BENCH_USE_STUB)

#include "stub_kernel.h"

#else  // ---------------- 真 kernel（CONTRACT §4 / §5）----------------

#include "decode_gemm.cuh"
#if defined(__has_include)
#if __has_include("weight_prepack.cuh")
#include "weight_prepack.cuh"
#endif
#endif

#endif
