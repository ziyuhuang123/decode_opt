// SPDX-License-Identifier: MIT
//
// bench/bench_common.cuh — 主程序（bench_decode.cu）与 per-M_TILE 实例化 TU
// （bench_m8.cu ... bench_m128.cu）之间的唯一耦合面：一个纯函数指针的
// VariantOps 注册表。主程序 *不* include kernel 头，因此 5 个 TU + main 可以
// 完全并行编译（CONTRACT §6「编译拆分」）。

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "configs.cuh"

namespace bench {

// ---------------------------------------------------------------------------
// 运行期缓冲视图（kernel 无关）。per-M_TILE TU 负责把它翻译成
// decode_gemm::Problem（CONTRACT §4）。
// ---------------------------------------------------------------------------
struct DeviceBuffers {
  int M = 0;                       // runtime M（<= M_TILE）
  int N = 0;                       // logical N（128 的倍数）
  int K = 0;
  int padded_n = 0;                // 该 variant 的 ceil(N/BM)*BM
  const __nv_fp8_e4m3* activation = nullptr;   // [act_rows_alloc, K]，前 M 行有效
  const __nv_fp8_e4m3* weight = nullptr;       // PREPACK ? packed : raw [N,K]
  const float* weight_scales = nullptr;        // [scale_rows_alloc, K/128]
  const float* activation_scales = nullptr;    // [act_rows_alloc, K/128]
  __nv_bfloat16* output = nullptr;             // [M,N] N-contiguous（含 pad slack）
  float* splitk_ws = nullptr;                  // SPLIT_K_CTA: [2][M,N]
  int* splitk_sem = nullptr;                   // SPLIT_K_CTA: [M*N] int32
};

// 一次 launch 的全部输入。tensor map 由 harness 在 timed region 外预建
// （CONTRACT §4 make_weight_tma / make_activation_tma），每个冷权重 set 一份。
struct LaunchRequest {
  DeviceBuffers buf;
  const CUtensorMap* w_map = nullptr;
  const CUtensorMap* a_map = nullptr;
  cudaStream_t stream = nullptr;
  bool pdl_attr = false;   // harness 侧 ProgrammaticStreamSerialization 开关
  bool use_api = false;    // true -> 走 decode_gemm::launch<Cfg>（attr 由 kernel 决定）
};

// ---------------------------------------------------------------------------
// VariantOps：一个已实例化的 (M_TILE, step, config) 的全部能力
// ---------------------------------------------------------------------------
struct VariantOps {
  ConfigSpec spec{};
  int step = -1;             // 0..6 = canonical
  int tune_index = -1;       // >=0 = tune 网格下标；-1 = canonical
  int role = 0;              // 0 = canonical 主配置；1 = M_TILE=128 的 wg1reg232 备选
  int shape_n = 0;           // 0 = 形状无关；>0 = 仅该 (shape_n,shape_k) 可用（tune override）
  int shape_k = 0;
  int m_tile = 8;
  std::string step_name;
  std::string config_str;
  const char* mechanism = "api";   // "kernel_ex"（harness 控制 attr）/ "api"

  // 几何 & 能力（全部 host 侧、无 CUDA 调用）
  int (*fn_padded_n)(int N) = nullptr;
  size_t (*fn_set_bytes)(int N, int K) = nullptr;   // 单个冷权重 set 的字节数
  int (*fn_smem_bytes)(int M, int K) = nullptr;     // Cfg::dynamic_smem_bytes(M,K)
  int (*fn_num_threads)() = nullptr;
  int (*fn_num_ctas)(int N) = nullptr;              // 含 gridDim.y（SPLIT_K_CTA）
  const char* (*fn_supported)(int M, int N, int K) = nullptr;  // nullptr = OK（Cfg::validate）

  // 操作
  cudaError_t (*fn_prepare)(const DeviceBuffers&) = nullptr;  // smem attr 等一次性设置
  cudaError_t (*fn_prepack)(const __nv_fp8_e4m3* src, __nv_fp8_e4m3* dst, int N,
                            int K, cudaStream_t stream) = nullptr;  // nullptr = raw
  cudaError_t (*fn_make_maps)(const DeviceBuffers&, CUtensorMap* w,
                              CUtensorMap* a) = nullptr;
  cudaError_t (*fn_launch)(const LaunchRequest&) = nullptr;   // 主路径
  cudaError_t (*fn_launch_api)(const DeviceBuffers&,
                               cudaStream_t) = nullptr;       // decode_gemm::launch<Cfg>
};

std::vector<VariantOps>& registry();
void register_variant(const VariantOps& v);

// per-M_TILE TU 的注册入口（main 显式调用，避免静态初始化顺序问题）
void register_m8();
void register_m16();
void register_m32();
void register_m64();
void register_m128();
inline void register_all() {
  register_m8();
  register_m16();
  register_m32();
  register_m64();
  register_m128();
}

// M 阶梯 -> compile-time M_TILE（CONTRACT §1）
inline int m_tile_for(int m) {
  if (m <= 8) return 8;
  if (m <= 16) return 16;
  if (m <= 32) return 32;
  if (m <= 64) return 64;
  return 128;
}

// canonical：按 (m_tile, step, N, M) 选 BM + 选 split-K on/off
// （S6 的 SPLIT_K_CTA 是条件启用：ctas_no_split < 78 且 M >= 8，见 configs.cuh）
const VariantOps* find_canonical(int m_tile, int step, int n, int m);
// M_TILE=128 有两条 canonical（kernels/README.md §8）：WG=2/REG=168（20B spill）
// 和 WG=1/REG=232/BM=64（0 spill，config 串带 wg1reg232=1）。两条都要跑。
std::vector<const VariantOps*> find_canonical_all(int m_tile, int step, int n,
                                                 int m);
// S6 的 split-K on/off 一对（--splitk-ablation 用）：role 0，同一 BM
std::vector<const VariantOps*> find_splitk_pair(int m_tile, int step, int n);

// ---- S 路 split-K 扫描（--splitk-factor-sweep，splitk-S agent 2026-09-13）----
// 一个 (基准几何, S) 的命中结果；v==nullptr 表示这个组合没实例化（调用方记
// skipped，不算失败）。基准几何有两条：
//   base0 = find_canonical 挑中的几何（该形状该 M 的现役 config，可能是 tune override）
//   base1 = canonical_for(6, m_tile, canonical_bm(...))，即形状无关的 canonical S6 几何
struct FactorPick {
  const VariantOps* v = nullptr;
  int factor = 1;      // S
  int base = 0;        // 0 / 1，见上
  std::string geom;    // "BM56/WG4/T1/REG96/STG3/SUBS2/SISM1"
};
std::vector<FactorPick> find_splitk_factor_sweep(int m_tile, int step, int n,
                                                 int k, int m,
                                                 const std::vector<int>& factors);

std::vector<const VariantOps*> find_tune(int m_tile, int step);
bool tune_step_instantiated(int m_tile, int step);

}  // namespace bench
