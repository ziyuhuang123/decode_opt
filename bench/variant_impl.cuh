// SPDX-License-Identifier: MIT
//
// bench/variant_impl.cuh —— 把一个 decode_gemm::KernelConfig<...> 包成
// bench::VariantOps（纯函数指针），供主程序调用。
// 只有 bench_m8.cu ... bench_m128.cu 这些「实例化 TU」include 本文件；主程序
// bench_decode.cu 不 include kernel 头，所以模板爆炸只发生在 5 个可并行的 TU 里
// （CONTRACT §6「编译拆分」）。
//
// 与 kernel 的耦合面（全部是 CONTRACT §4/§5 点名、且已在
// bench/probe/probe_cfg_asserts.cu 里编译验证过的符号）：
//   decode_gemm::KernelConfig<15 params>      Cfg::validate / num_ctas_x /
//                                             dynamic_smem_bytes / kNumThreads /
//                                             kFixedSmemBytes / kSplitKCta
//   decode_gemm::Problem / GemmRuntime / LaunchPlan
//   decode_gemm::make_launch_plan / ensure_smem_attribute / launch_with_maps /
//                 launch / make_weight_tma / make_activation_tma /
//                 fp8_decode_gemm
//   decode_gemm::prepack::padded_n / packed_weight_bytes /
//                 launch_prepack_weight<BM, TILES>
//
// 计时口径的关键点：三个协议都走 harness 自己的 cudaLaunchKernelEx，
// ProgrammaticStreamSerialization 只在 PDL 协议打开。这样 B2B 与 PDL 只差一个
// attr，可比；否则 kernel 的 launch_with_maps 在 kPdl 时恒开 PSS，B2B 会被污染。
// 若必须退回 kernel launcher（-DBENCH_NO_DIRECT_LAUNCH），mechanism 会记成
// "api" 并在 summary JSON 里标红：那时 B2B 与 PDL 等价，不能用来讲 PDL 收益。

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <type_traits>
#include <utility>

#include "bench_common.cuh"
#include "configs.cuh"
#include "kernel_api.h"

#if !defined(BENCH_NO_DIRECT_LAUNCH)
#define BENCH_HAVE_DIRECT_LAUNCH 1
#endif

namespace bench {

template <typename Cfg>
inline decode_gemm::Problem make_problem(const DeviceBuffers& b) {
  decode_gemm::Problem p;
  p.M = b.M;
  p.N = b.N;
  p.K = b.K;
  p.activation = b.activation;
  p.weight = b.weight;
  p.weight_scales = b.weight_scales;
  p.activation_scales = b.activation_scales;
  p.output = b.output;
  p.splitk_ws = b.splitk_ws;
  p.splitk_sem = b.splitk_sem;
  return p;
}

// SKF = SPLITK_FACTOR（S 路 split-K 的宽度，kernel 模板第 16 个参数）：
// 0 = 沿用 SPLITK 布尔（true -> S=2），1 = 关，2/4/8 = gridDim.y = S。
// 放在最后并带默认值，所有旧的实例化点一字不改也能编。
template <int MT, int BM, int WG, int REGS, int STG, int TILES, int SUBS,
          bool PREPACK, bool PDL, bool EPI, bool SPLITK, int PR, bool SPW,
          bool PLA, bool SISM, int SKF = 0>
struct VariantImpl {
  using Cfg = decode_gemm::KernelConfig<MT, BM, WG, REGS, STG, TILES, SUBS,
                                        PREPACK, PDL, EPI, SPLITK, PR, SPW, PLA,
                                        SISM, SKF>;

  // 与 kernel 侧 Cfg::kSplitKFactor 同一条归一化规则（configs.cuh sk_factor_of）
  static constexpr int kSkFactor =
      (SKF >= 2) ? SKF : ((SKF == 1 || !SPLITK) ? 1 : 2);

  static constexpr ConfigSpec spec() {
    ConfigSpec s = make_spec(MT, BM, WG, REGS, STG, TILES, SUBS, PREPACK, PDL,
                             EPI, SPLITK, PR, SPW, PLA);
    s.scale_in_smem = SISM;
    // 归一化：registry / CSV / pick() 看到的永远是**有效**值，
    // 于是「legacy 布尔 S=2」与「显式 SK=2」在 config 串上完全一致，不会重复计数。
    s.splitk_factor = kSkFactor;
    s.splitk_cta = (kSkFactor >= 2);
    return s;
  }

  // ---- 几何 / 能力：一律问 kernel，不在 harness 侧另算一套 ----
  static int padded_n_fn(int N) {
    return decode_gemm::prepack::padded_n(N, BM);
  }
  static size_t set_bytes_fn(int N, int K) {
    return PREPACK ? decode_gemm::prepack::packed_weight_bytes(N, K, BM)
                   : static_cast<size_t>(N) * static_cast<size_t>(K);
  }
  static int smem_fn(int M, int K) {
    return static_cast<int>(Cfg::dynamic_smem_bytes(M, K));
  }
  static int threads_fn() { return static_cast<int>(Cfg::kNumThreads); }
  static int ctas_fn(int N) {
    // gridDim.x * gridDim.y，gridDim.y == SPLITK_FACTOR（一律问 kernel）
    return static_cast<int>(Cfg::num_ctas(N));
  }
  static const char* supported_fn(int M, int N, int K) {
    return Cfg::validate(M, N, K);
  }

  static cudaError_t prepare_fn(const DeviceBuffers& b) {
    const decode_gemm::Problem p = make_problem<Cfg>(b);
    const decode_gemm::LaunchPlan plan = decode_gemm::make_launch_plan<Cfg>(p);
    if (plan.error != nullptr) return cudaErrorInvalidValue;
    return decode_gemm::ensure_smem_attribute<Cfg>(plan.smem_bytes);
  }

  static cudaError_t prepack_fn(const __nv_fp8_e4m3* src, __nv_fp8_e4m3* dst,
                                int N, int K, cudaStream_t stream) {
    return decode_gemm::prepack::launch_prepack_weight<BM, TILES>(src, dst, N, K,
                                                                  stream);
  }

  static cudaError_t make_maps_fn(const DeviceBuffers& b, CUtensorMap* w,
                                  CUtensorMap* a) {
    const decode_gemm::Problem p = make_problem<Cfg>(b);
    *w = decode_gemm::make_weight_tma<Cfg>(p);
    *a = decode_gemm::make_activation_tma<Cfg>(p);
    return cudaGetLastError();
  }

  // kernel 自带 launcher（tensor map 由它自己 encode；kPdl 时它自己开 PSS）
  static cudaError_t launch_api_fn(const DeviceBuffers& b, cudaStream_t s) {
    return decode_gemm::launch<Cfg>(make_problem<Cfg>(b), s);
  }

#if defined(BENCH_HAVE_DIRECT_LAUNCH)
  // harness 自己 launch：attr 由调用方（协议）决定；tensor map 用预建好的
  static cudaError_t launch_direct_fn(const LaunchRequest& rq) {
    const decode_gemm::Problem p = make_problem<Cfg>(rq.buf);
    const decode_gemm::LaunchPlan plan = decode_gemm::make_launch_plan<Cfg>(p);
    if (plan.error != nullptr) return cudaErrorInvalidValue;
    if (Cfg::kSplitKCta && (p.splitk_ws == nullptr || p.splitk_sem == nullptr))
      return cudaErrorInvalidValue;
    if (rq.w_map == nullptr || rq.a_map == nullptr) return cudaErrorInvalidValue;
    decode_gemm::GemmRuntime rt;
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
    config.stream = rq.stream;
    config.attrs = attributes;
    config.numAttrs = 0;
    if (rq.pdl_attr) {
      attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attributes[0].val.programmaticStreamSerializationAllowed = 1;
      config.numAttrs = 1;
    }
    return cudaLaunchKernelEx(&config, decode_gemm::fp8_decode_gemm<Cfg>,
                              p.output, p.weight_scales, p.activation_scales,
                              *rq.w_map, *rq.a_map, rt);
  }
#endif

  static cudaError_t launch_fn(const LaunchRequest& rq) {
#if defined(BENCH_HAVE_DIRECT_LAUNCH)
    if (!rq.use_api) return launch_direct_fn(rq);
    return decode_gemm::launch_with_maps<Cfg>(make_problem<Cfg>(rq.buf),
                                              *rq.w_map, *rq.a_map, rq.stream);
#else
    (void)rq.pdl_attr;   // 无法控制 attr：PDL 语义交给 kernel launcher
    return decode_gemm::launch_with_maps<Cfg>(make_problem<Cfg>(rq.buf),
                                              *rq.w_map, *rq.a_map, rq.stream);
#endif
  }

  static VariantOps ops(int step, int tune_index, int role = 0) {
    VariantOps o{};
    o.spec = spec();
    o.step = step;
    o.tune_index = tune_index;
    o.role = role;
    o.m_tile = MT;
    o.step_name = (step >= 0 && step < kNumSteps) ? kStepNames[step] : "tune";
    o.config_str = config_to_string(spec());
#if defined(BENCH_HAVE_DIRECT_LAUNCH)
    o.mechanism = "kernel_ex";
#else
    o.mechanism = "api";
#endif
    o.fn_padded_n = &padded_n_fn;
    o.fn_set_bytes = &set_bytes_fn;
    o.fn_smem_bytes = &smem_fn;
    o.fn_num_threads = &threads_fn;
    o.fn_num_ctas = &ctas_fn;
    o.fn_supported = &supported_fn;
    o.fn_prepare = &prepare_fn;
    o.fn_prepack = PREPACK ? &prepack_fn : nullptr;
    o.fn_make_maps = &make_maps_fn;
    o.fn_launch = &launch_fn;
    o.fn_launch_api = &launch_api_fn;
    return o;
  }
};

#if defined(BENCH_TUNE_GRID_S4)
inline constexpr int kTuneGridSizeEff = kTuneS4GridSize;   // 156
#else
inline constexpr int kTuneGridSizeEff = kTuneGridSize;     // 390
#endif

// ------------------------------------------------------------- 注册展开 -----
// C++17 不能拿 struct 当模板实参，所以把 constexpr ConfigSpec 的字段拆开喂给
// VariantImpl（字段本身都是常量表达式）。
// SKO: 0 = 用 canonical_for 给出的 splitk；1 = 反过来（S6 注册 on/off 消融对）
// SKF: >=2 时显式指定 S 路宽度（覆盖 SKO）；0 = 走 SKO 的老逻辑。
// S6 的 canonical 基线 splitk_cta==true（即 S=2），SKO=1 给 S=1，所以
// {SKO=0, SKO=1, SKF=4, SKF=8} 四条正好铺开 S ∈ {1,2,4,8} 且不重复。
template <int MT, int STEP, int BM, int SKO, int SKF = 0>
inline void register_canonical_variant() {
  constexpr ConfigSpec base = canonical_for(STEP, MT, BM);
  constexpr ConfigSpec s =
      (SKF >= 2) ? with_splitk_factor(base, SKF)
                 : ((SKO == 1) ? with_splitk(base, !base.splitk_cta) : base);
  if constexpr (feasible(s)) {
    static VariantOps o = VariantImpl<
        s.m_tile, s.bm, s.wg, s.math_regs, s.stg, s.tiles, s.subs, s.prepack,
        s.pdl, s.epi_overlap, s.splitk_cta, s.prod_regs, s.single_producer_warp,
        s.preload_act, s.scale_in_smem, s.splitk_factor>::ops(STEP, -1, /*role=*/0);
    register_variant(o);
  }
}

template <int MT, int STEP, int IDX>
inline void register_tune_variant() {
  // 分块展开的最后一块会超出网格大小（kTuneChunks*kTuneChunk >= kTuneGridSizeEff），
  // 越界下标不注册，否则同一个 config 会被登记两次 -> tune CSV 出现重复行
  if constexpr (IDX < tune_grid_size_for(STEP)) {
    // step6 且 TUNE_SKF=1 时 IDX 里含 S 维（configs.cuh tune_slot_at）
    constexpr ConfigSpec s = tune_slot_at(STEP, MT, IDX);
    if constexpr (feasible(s)) {
      static VariantOps o = VariantImpl<
          s.m_tile, s.bm, s.wg, s.math_regs, s.stg, s.tiles, s.subs, s.prepack,
          s.pdl, s.epi_overlap, s.splitk_cta, s.prod_regs,
          s.single_producer_warp, s.preload_act,
          s.scale_in_smem, s.splitk_factor>::ops(STEP, IDX);
      register_variant(o);
    }
  }
}

// M_TILE=128 的第二条 canonical（WG=1/REG=232/BM=64/TILES=1/SUBS=1，0 spill）
// M_TILE=128 的第二条 canonical（kernels/README.md §8：WG=1/REG=232/BM=64，0 spill）
template <int MT, int STEP, int SKO>
inline void register_alt_variant() {
  constexpr ConfigSpec base = canonical_alt(STEP, MT);
  constexpr ConfigSpec s =
      (SKO == 1) ? with_splitk(base, !base.splitk_cta) : base;
  if constexpr (feasible(s)) {
    static VariantOps o = VariantImpl<
        s.m_tile, s.bm, s.wg, s.math_regs, s.stg, s.tiles, s.subs, s.prepack,
        s.pdl, s.epi_overlap, s.splitk_cta, s.prod_regs, s.single_producer_warp,
        s.preload_act, s.scale_in_smem, s.splitk_factor>::ops(STEP, -1, /*role=*/1);
    register_variant(o);
  }
}

// ---- tune override 变体注册（orchestrator 2026-09-13）----
// kShapeOverrides 里每行是一个 (N,K,M_TILE) 的 tune-winner config；这里把它实例化
// 成带 shape tag 的 variant，runtime pick 时优先匹配形状。
// SKF >= 2：给这个 tune-winner 几何再挂一条 S 路 split-K 的变体。
// 闸门（控制实例化数）：只有 sk_factor_expand(o.n, o.bm) 且 K 整除性成立的
// (形状, S) 才展开；其余形状的 S6 仍然只有 {worth, !worth} 两条。
template <int MT, int IDX, int STEP, int SKO, int SKF = 0>
inline void register_override_variant() {
  constexpr ShapeOverride o = kShapeOverrides[IDX];
  if constexpr (o.bm > 0 && STEP >= 4 && o.m_tile == MT) {
    constexpr bool epi = (STEP >= 5);
    constexpr bool worth = splitk_worth_it(o.n, o.bm);
    constexpr bool sk = (STEP == 5) ? ((SKO == 0) ? worth : !worth) : false;
    constexpr bool expand =
        (STEP == 5) && (SKF >= 2) && sk_factor_expand(o.n, o.bm) &&
        sk_factor_ok_k(o.k, o.subs, o.wg / o.tiles, SKF);
    constexpr ConfigSpec base = make_spec(
        o.m_tile, o.bm, o.wg, o.regs, o.stg, o.tiles, o.subs,
        /*prepack=*/true, /*pdl=*/true, epi, sk, kDefaultProducerRegs,
        /*spw=*/false, /*pla=*/false);
    constexpr ConfigSpec s2 = [=]() {
      ConfigSpec t = expand ? with_splitk_factor(base, SKF) : base;
      t.scale_in_smem = (o.m_tile <= 16);
      if (!expand && SKF >= 2) t.m_tile = -1;    // 闸门不通过 -> feasible()==false
      return t;
    }();
    if constexpr (feasible(s2)) {
      static VariantOps ov = VariantImpl<
          s2.m_tile, s2.bm, s2.wg, s2.math_regs, s2.stg, s2.tiles, s2.subs,
          s2.prepack, s2.pdl, s2.epi_overlap, s2.splitk_cta, s2.prod_regs,
          s2.single_producer_warp, s2.preload_act, s2.scale_in_smem,
          s2.splitk_factor>::ops(STEP, -1, /*role=*/0);
      ov.shape_n = o.n;
      ov.shape_k = o.k;
      register_variant(ov);
    }
  }
}

// ---- PDL 触发点放置消融（role=2，仅 M_TILE=8）----
// S4 几何下三个触发点变体：producer-only(PDL=1,EPI=0) / both(PDL=1,EPI=1) /
// store-only(PDL=0,EPI=1)。per-CTA 信号语义下「先发的先生效」，故预期 producer≈both≈store；
// 该消融用数据钉死这件事（--pdl-placement）。
template <int MT, int BM, int PDLF, int EPIF>
inline void register_epi_ablation_variant() {
  constexpr ConfigSpec s = []() {
    ConfigSpec t = canonical_for(4, MT, BM);
    t.pdl = (PDLF == 1);
    t.epi_overlap = (EPIF == 1);
    return t;
  }();
  if constexpr (feasible(s)) {
    static VariantOps o = VariantImpl<
        s.m_tile, s.bm, s.wg, s.math_regs, s.stg, s.tiles, s.subs, s.prepack,
        s.pdl, s.epi_overlap, s.splitk_cta, s.prod_regs, s.single_producer_warp,
        s.preload_act, s.scale_in_smem, s.splitk_factor>::ops(4, -1, /*role=*/2);
    register_variant(o);
  }
}

template <int MT, int BM>
inline void register_epi_triple() {
  register_epi_ablation_variant<MT, BM, 1, 0>();   // producer-only
  register_epi_ablation_variant<MT, BM, 1, 1>();   // both
  register_epi_ablation_variant<MT, BM, 0, 1>();   // store-only
}

template <int MT, int STEP, size_t... II>
inline void register_override_step(std::index_sequence<II...>) {
  (register_override_variant<MT, static_cast<int>(II), STEP, 0>(), ...);
  if constexpr (STEP == 5) {
    (register_override_variant<MT, static_cast<int>(II), STEP, 1>(), ...);
    // S 维：{2,4,8}；S=2 已由 SKO 的两条之一覆盖，这里只补 4/8
    (register_override_variant<MT, static_cast<int>(II), STEP, 0, 4>(), ...);
    (register_override_variant<MT, static_cast<int>(II), STEP, 0, 8>(), ...);
  }
}

template <int MT, size_t... SI>
inline void register_override_all(std::index_sequence<SI...>) {
  (register_override_step<MT, static_cast<int>(SI)>(
       std::make_index_sequence<kShapeOverrideCount>{}),
   ...);
}

template <int MT, int STEP, size_t... BI>
inline void register_canonical_step(std::index_sequence<BI...>) {
  (register_canonical_variant<MT, STEP, kCandidateBm[BI], 0>(), ...);
  if constexpr (STEP == 6) {
    // split-K on/off 消融对（S=2 / S=1）
    (register_canonical_variant<MT, STEP, kCandidateBm[BI], 1>(), ...);
    // S 路：canonical 的形状无关，闸门在运行时由 Cfg::validate 兜底，
    // 这里对 5 个候选 BM 一律铺开 S=4/8（每个 M_TILE 只多 10 个实例化）
    (register_canonical_variant<MT, STEP, kCandidateBm[BI], 0, 4>(), ...);
    (register_canonical_variant<MT, STEP, kCandidateBm[BI], 0, 8>(), ...);
  }
  register_alt_variant<MT, STEP, 0>();
  if constexpr (STEP == 6) register_alt_variant<MT, STEP, 1>();
}

// tune 网格分块展开（一次 fold 540 项会拖慢编译）
inline constexpr int kTuneChunk = 60;
template <int MT, int STEP, int BASE, size_t... II>
inline void register_tune_chunk(std::index_sequence<II...>) {
  (register_tune_variant<MT, STEP, BASE + static_cast<int>(II)>(), ...);
}
// step6 且开了 TUNE_SKF 时索引空间 x kTuneSkSlots，所以 chunk 数是 per-step 的
template <int STEP>
inline constexpr int tune_chunks_for() {
  return (tune_grid_size_for(STEP) + kTuneChunk - 1) / kTuneChunk;
}
template <int MT, int STEP, size_t... CI>
inline void register_tune_step_chunks(std::index_sequence<CI...>) {
  (register_tune_chunk<MT, STEP, static_cast<int>(CI) * kTuneChunk>(
       std::make_index_sequence<kTuneChunk>{}),
   ...);
}
template <int MT, int STEP>
inline void register_tune_step() {
  register_tune_step_chunks<MT, STEP>(
      std::make_index_sequence<tune_chunks_for<STEP>()>{});
}

// 哪些 step 烘进 tune 网格：Makefile 的 TUNE_STEPS 决定（默认 all = 0x7F）。
// 全量是 7 step x 540 网格项 x 5 个 M_TILE，编译很久；快速迭代用 TUNE=0。
#if !defined(BENCH_TUNE_STEPS_MASK)
#define BENCH_TUNE_STEPS_MASK 0x7F
#endif
#if !defined(BENCH_TUNE)
#define BENCH_TUNE 1
#endif

template <int MT, size_t... SI>
inline void register_canonical_all(std::index_sequence<SI...>) {
  (register_canonical_step<MT, static_cast<int>(SI)>(
       std::make_index_sequence<kCandidateBmCount>{}),
   ...);
}

template <int MT, size_t... SI>
inline void register_tune_all(std::index_sequence<SI...>) {
#if BENCH_TUNE
#if defined(BENCH_TUNE_GRID_S4)
  // s4 网格的布尔特性就是 S4 的（PREPACK/PDL/SUBS=2），所以只挂在 step 4 上
  (void)sizeof...(SI);
  register_tune_step<MT, 4>();
#else
  (
      [&] {
        if constexpr (((BENCH_TUNE_STEPS_MASK >> static_cast<int>(SI)) & 1) != 0)
          register_tune_step<MT, static_cast<int>(SI)>();
      }(),
      ...);
#endif
#else
  (void)sizeof...(SI);
#endif
}

template <int MT>
inline void register_m_tile() {
  static bool done = false;
  if (done) return;
  done = true;
  if constexpr (MT == 8) register_epi_triple<8, 56>();
  register_canonical_all<MT>(std::make_index_sequence<kNumSteps>{});
  register_override_all<MT>(std::make_index_sequence<kNumSteps>{});
  register_tune_all<MT>(std::make_index_sequence<kNumSteps>{});
}

}  // namespace bench
