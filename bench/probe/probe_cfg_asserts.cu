// SPDX-License-Identifier: MIT
//
// bench/probe/probe_cfg_asserts.cu —— 编译期一致性闸门。
//
// 把 bench/configs.cuh 判定为 feasible() 的每一个 config（canonical 七步 x 5 个候选
// BM，以及整个 --tune 网格）都拿去实例化 decode_gemm::KernelConfig<...>，
// 让 kernel 头里的 static_assert（寄存器预算 / kFixedSmemBytes / 结构约束）亲自把关。
// 只实例化 config struct，不实例化 __global__ kernel，所以几秒钟就能编完。
//
//   make probe-cfg          # 真 kernel
//   make probe-cfg STUB=1   # stub
//
// 编不过 = harness 的 feasible() 比 kernel 松 -> 必须收紧 configs.cuh，
// 否则 bench_m*.cu 会在正式构建时炸 static_assert。

#include <cstddef>
#include <type_traits>

#include "configs.cuh"
#include "kernel_api.h"

namespace probe {

using namespace bench;

// SKF = SPLITK_FACTOR（S 路 split-K 宽度，kernel 模板第 16 个参数）
template <int MT, int BM, int WG, int REGS, int STG, int TILES, int SUBS,
          bool PP, bool PDL, bool EPI, bool SK, int PR, bool SPW, bool PLA,
          bool SISM, int SKF = 0>
using CfgOf = decode_gemm::KernelConfig<MT, BM, WG, REGS, STG, TILES, SUBS, PP,
                                        PDL, EPI, SK, PR, SPW, PLA, SISM, SKF>;

// 只在 OK==true 时才碰 C，从而触发 KernelConfig 的隐式实例化
template <class C, bool OK>
struct Probe {
  static constexpr long long v = 0;
};
template <class C>
struct Probe<C, true> {
  static constexpr long long v = static_cast<long long>(C::kFixedSmemBytes) +
                                 static_cast<long long>(C::kNumThreads) +
                                 static_cast<long long>(C::kStages);
};

template <class S>
struct CfgFromSpec;

template <int MT, int STEP, int IDX>
struct CheckTune {
  static constexpr ConfigSpec s = tune_slot_at(STEP, MT, IDX);
  using C = CfgOf<s.m_tile, s.bm, s.wg, s.math_regs, s.stg, s.tiles, s.subs,
                  s.prepack, s.pdl, s.epi_overlap, s.splitk_cta, s.prod_regs,
                  s.single_producer_warp, s.preload_act, s.scale_in_smem,
                  s.splitk_factor>;
  static constexpr long long v = Probe<C, feasible(s)>::v;
};

template <int MT, int STEP, int BMI>
struct CheckCanon {
  static constexpr ConfigSpec s = canonical_for(STEP, MT, kCandidateBm[BMI]);
  using C = CfgOf<s.m_tile, s.bm, s.wg, s.math_regs, s.stg, s.tiles, s.subs,
                  s.prepack, s.pdl, s.epi_overlap, s.splitk_cta, s.prod_regs,
                  s.single_producer_warp, s.preload_act, s.scale_in_smem,
                  s.splitk_factor>;
  static constexpr long long v = Probe<C, feasible(s)>::v;
};

template <int MT, int STEP, size_t... I>
constexpr long long sum_tune(std::index_sequence<I...>) {
  return (CheckTune<MT, STEP, static_cast<int>(I)>::v + ...);
}
template <int MT, int STEP, size_t... I>
constexpr long long sum_canon(std::index_sequence<I...>) {
  return (CheckCanon<MT, STEP, static_cast<int>(I)>::v + ...);
}
// S 路 split-K（step6 才注册）：与 variant_impl 的 register_canonical_step 对齐
template <int MT, int BM, int SKF>
struct CheckCanonSk {
  static constexpr ConfigSpec s =
      with_splitk_factor(canonical_for(6, MT, BM), SKF);
  using C = CfgOf<s.m_tile, s.bm, s.wg, s.math_regs, s.stg, s.tiles, s.subs,
                  s.prepack, s.pdl, s.epi_overlap, s.splitk_cta, s.prod_regs,
                  s.single_producer_warp, s.preload_act, s.scale_in_smem,
                  s.splitk_factor>;
  static constexpr long long v = Probe<C, feasible(s)>::v;
};
template <int MT, int SKF, size_t... I>
constexpr long long sum_canon_sk(std::index_sequence<I...>) {
  return (CheckCanonSk<MT, kCandidateBm[I], SKF>::v + ...);
}
template <int MT, size_t... S>
constexpr long long sum_steps(std::index_sequence<S...>) {
  return (sum_tune<MT, static_cast<int>(S)>(
              std::make_index_sequence<tune_grid_size_for(
                  static_cast<int>(S))>{}) +
          ...);
}
template <int MT, size_t... S>
constexpr long long sum_canon_steps(std::index_sequence<S...>) {
  return (sum_canon<MT, static_cast<int>(S)>(
              std::make_index_sequence<kCandidateBmCount>{}) +
          ...);
}

// 全部 constexpr -> 编译期就把 static_assert 跑完；sink 防止被优化掉
template <int MT>
constexpr long long sum_sk_ladder() {
  return sum_canon_sk<MT, 4>(std::make_index_sequence<kCandidateBmCount>{}) +
         sum_canon_sk<MT, 8>(std::make_index_sequence<kCandidateBmCount>{});
}

constexpr long long kSink =
    (sum_sk_ladder<8>() + sum_sk_ladder<16>() + sum_sk_ladder<32>() +
     sum_sk_ladder<64>() + sum_sk_ladder<128>() +
     sum_canon_steps<8>(std::make_index_sequence<kNumSteps>{}) +
     sum_canon_steps<16>(std::make_index_sequence<kNumSteps>{}) +
     sum_canon_steps<32>(std::make_index_sequence<kNumSteps>{}) +
     sum_canon_steps<64>(std::make_index_sequence<kNumSteps>{}) +
     sum_canon_steps<128>(std::make_index_sequence<kNumSteps>{}) +
     sum_steps<8>(std::make_index_sequence<kNumSteps>{}) +
     sum_steps<16>(std::make_index_sequence<kNumSteps>{}) +
     sum_steps<32>(std::make_index_sequence<kNumSteps>{}) +
     sum_steps<64>(std::make_index_sequence<kNumSteps>{}) +
     sum_steps<128>(std::make_index_sequence<kNumSteps>{}));

// 顺便验证 harness 直接 launch 需要的 kernel 侧符号都存在（CONTRACT §4）
template <class C>
struct DirectLaunchSymbols {
  using Cfg = C;
  static cudaError_t check(const decode_gemm::Problem& p, cudaStream_t s,
                           const CUtensorMap& w, const CUtensorMap& a) {
    const decode_gemm::LaunchPlan plan = decode_gemm::make_launch_plan<Cfg>(p);
    if (plan.error != nullptr) return cudaErrorInvalidValue;
    const cudaError_t e = decode_gemm::ensure_smem_attribute<Cfg>(plan.smem_bytes);
    if (e != cudaSuccess) return e;
    decode_gemm::GemmRuntime rt{p.M, p.N, p.K, p.splitk_ws, p.splitk_sem};
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = plan.grid;
    cfg.blockDim = plan.block;
    cfg.dynamicSmemBytes = plan.smem_bytes;
    cfg.stream = s;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, decode_gemm::fp8_decode_gemm<Cfg>, p.output,
                              p.weight_scales, p.activation_scales, w, a, rt);
  }
};
using ProbeCfg = CfgOf<8, 48, 2, 80, 3, 1, 2, true, true, true, false, 32, false,
                       false, true>;
// 只声明、不调用：取地址即可确认符号可用（不产生 kernel codegen）
inline auto* kDirectSym = &DirectLaunchSymbols<ProbeCfg>::check;
inline auto* kPrepackSym =
    &decode_gemm::prepack::launch_prepack_weight<48, 1>;

}  // namespace probe

int main() {
  std::printf("probe_cfg_asserts OK: sink=%lld direct_launch=%p prepack=%p\n",
              static_cast<long long>(probe::kSink),
              reinterpret_cast<const void*>(probe::kDirectSym),
              reinterpret_cast<const void*>(probe::kPrepackSym));
  return 0;
}
