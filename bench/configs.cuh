// SPDX-License-Identifier: MIT
//
// bench/configs.cuh — step -> canonical KernelConfig 映射表 + --tune 网格枚举。
//
// 纯 constexpr：per-M_TILE TU（bench_m8.cu ... bench_m128.cu）用它决定实例化哪些
// decode_gemm::KernelConfig<...>。编译期闸门 feasible() 镜像 kernels/decode_gemm.cuh
// 里 KernelConfig 的 static_assert（只含与 N/K/M 无关的那部分：kFixedSmemBytes、
// 寄存器预算、结构约束），所以不合法的 config 根本不会被实例化，不会把 kernel 的
// static_assert 炸出来；与形状相关的约束（SUBS/WGsPerTile 整除、act+scale smem）
// 交给运行时的 Cfg::validate(M,N,K)。
//
// 口径来源：
//   CONTRACT §3   M_TILE=8 的 canonical 七步（继承 v2 调优史）
//   CONTRACT §4   KernelConfig 模板参数顺序（含第 15 个 SCALE_IN_SMEM，默认 M_TILE<=16）
//   CONTRACT §6   --tune 网格 BM/WG/STG/SUBS/TILES/MATH_REGS
//   orchestrator 2026-09-13 补充：canonical BM 按形状选（整除 N、pad 最小、tile 数
//                 贴近 128 CTA 甜区者优先），tune 网格加入 BM=40/56
//
// !!! M_TILE >= 16 的 canonical 表是占位 !!!
// 等 kernel agent 的 kernels/README.md 定稿后必须同步（见 bench/README.md「待同步」）。
// 占位规则（CONTRACT §3 末尾）：TILES_PER_CTA = NUM_MATH_WG；M_TILE=128 用
// REGS=168 / WG=2 / STG=2 / SUBS=1；预算超了就自动降级 SUBS->STG->WG 并把
// downgraded 置 true（会出现在 CSV 的 config 列 DOWNGRADED=1 与 summary JSON）。

#pragma once

#include <cstddef>
#include <string>

namespace bench {

// ---------------------------------------------------------------- 常量 -------
inline constexpr int kNumSteps = 6;                        // S0..S5（epi_overlap 不再是独立步：见 README「PDL 与触发点」）
inline constexpr int kKBlock = 128;                        // scale group = wgmma k32*4
inline constexpr int kWgmmaRows = 64;                      // A 侧 m64
inline constexpr size_t kMaxDynamicSmemBytes = 227 * 1024;  // SM90 上限
inline constexpr size_t kRegisterFileSize = 65536;          // 每 CTA 寄存器
inline constexpr int kDefaultProducerRegs = 32;
inline constexpr int kTargetCtaCount = 128;                 // v2 实测甜区(ANALYSIS §4.3)
inline constexpr int kSmemPaddingBytes = 4096;              // = kernels kSmemPaddingBytes

inline constexpr const char* kStepNames[kNumSteps] = {
    "baseline", "pdl", "blockk", "prepack", "tile_stage", "splitk"};

// canonical BM 候选（orchestrator 补充：整除 N 优先，pad 最小者优先）
inline constexpr int kCandidateBmCount = 5;
inline constexpr int kCandidateBm[kCandidateBmCount] = {32, 40, 48, 56, 64};

struct ConfigSpec {
  int m_tile = 8;
  int bm = 64;              // OUTPUT_ROWS_PER_CTA
  int wg = 4;               // NUM_MATH_WG
  int math_regs = 112;      // MATH_REGS
  int stg = 2;              // STAGES_PER_WG
  int tiles = 1;            // TILES_PER_CTA
  int subs = 1;             // SUBS_PER_STAGE
  bool prepack = false;
  bool pdl = false;
  bool epi_overlap = false;
  bool splitk_cta = false;
  // S 路 split-K 的宽度 S（kernel 模板第 16 个参数 SPLITK_FACTOR）：
  //   0 = 未指定（沿用 splitk_cta 布尔：true -> 2，false -> 1）
  //   1 = 关；2/4/8 = gridDim.y = S
  // 有效值一律用 sk_factor_of() 读，不要直接看这个字段。
  int splitk_factor = 0;
  int prod_regs = kDefaultProducerRegs;
  bool single_producer_warp = false;
  bool preload_act = false;              // 仅 M_TILE=8 且 PREPACK（CONTRACT §4）
  bool scale_in_smem = true;             // kernels 默认 = (M_TILE <= 16)
  bool downgraded = false;               // canonical_for() 降级过 SUBS/STG/WG
  bool wg1reg232 = false;                // M_TILE=128 的第二条 canonical（0 spill）
};

// 显式构造（不依赖聚合初始化顺序）
constexpr ConfigSpec make_spec(int mt, int bm, int wg, int regs, int stg,
                              int tiles, int subs, bool prepack, bool pdl,
                              bool epi, bool splitk, int pr, bool spw, bool pla,
                              bool downgraded = false, int sk_factor = 0) {
  ConfigSpec s{};
  s.splitk_factor = sk_factor;
  s.m_tile = mt; s.bm = bm; s.wg = wg; s.math_regs = regs; s.stg = stg;
  s.tiles = tiles; s.subs = subs; s.prepack = prepack; s.pdl = pdl;
  s.epi_overlap = epi; s.splitk_cta = splitk; s.prod_regs = pr;
  s.single_producer_warp = spw; s.preload_act = pla;
  s.scale_in_smem = (mt <= 16);          // 与 kernels/decode_gemm.cuh 的默认一致
  s.downgraded = downgraded;
  return s;
}

// ------------------------------------------------------------ 派生量 --------
constexpr int wgs_per_tile(const ConfigSpec& s) { return s.wg / s.tiles; }
// 有效 split-K 宽度 S：与 kernels/decode_gemm.cuh 的 Cfg::kSplitKFactor 逐字对齐
// （SPLITK_FACTOR>=2 优先；==1 强制关；==0 回落到旧布尔，true 即 2）。
constexpr int sk_factor_of(const ConfigSpec& s) {
  if (s.splitk_factor >= 2) return s.splitk_factor;
  if (s.splitk_factor == 1) return 1;
  return s.splitk_cta ? 2 : 1;
}
constexpr bool sk_active(const ConfigSpec& s) { return sk_factor_of(s) >= 2; }
constexpr bool sk_factor_legal(int f) {
  return f == 0 || f == 1 || f == 2 || f == 4 || f == 8;
}
constexpr int num_stages(const ConfigSpec& s) { return s.wg * s.stg; }
constexpr int num_threads(const ConfigSpec& s) { return (s.wg + 1) * 128; }
constexpr int act_rows_of(const ConfigSpec& s) { return s.m_tile; }
constexpr size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// host 侧几何（正式值以 Cfg::num_ctas_x / prepack::padded_n 为准）
constexpr int num_output_tiles(int n, int bm) { return (n + bm - 1) / bm; }
constexpr int padded_n(int n, int bm) { return num_output_tiles(n, bm) * bm; }
constexpr int num_ctas_x(int n, int bm, int tiles) {
  return (num_output_tiles(n, bm) + tiles - 1) / tiles;
}

// 编译期 smem 闸门：镜像 kernels/decode_gemm.cuh 的 kFixedSmemBytes
//   weight(kStages*SUBS*BM*128 + 4096) + barriers + partials + splitk flags
// act/scale smem 依赖 runtime M,K，由 Cfg::validate() 在运行时把关。
// 保守 K：feasible() 无 runtime K，用模型里最大的 K=16384 做最坏估计。
inline constexpr int kFeasibleMaxK = 16384;

constexpr size_t fixed_smem_bytes(const ConfigSpec& s) {
  const size_t stages = static_cast<size_t>(num_stages(s));
  const size_t weight =
      stages * static_cast<size_t>(s.subs) * s.bm * kKBlock + kSmemPaddingBytes;
  // activation（最坏 M=M_TILE）与 combined scales（SIS=true 时）必须计入，
  // 否则与 kernel 侧 Cfg::dynamic_smem_bytes(M,K) 不一致（orchestrator 2026-09-13 修）。
  const size_t act = s.preload_act
      ? static_cast<size_t>(s.m_tile) * kFeasibleMaxK
      : stages * static_cast<size_t>(s.subs) * s.m_tile * kKBlock;
  const size_t scales = s.scale_in_smem
      ? static_cast<size_t>(s.tiles) * s.m_tile * 2 *
            (kFeasibleMaxK / kKBlock) * sizeof(float)
      : 0;
  const size_t barriers = align_up(
      2 * stages * 8 + (s.preload_act ? 8 : 0), 16);
  const size_t partial =
      (wgs_per_tile(s) > 1)
          ? align_up(static_cast<size_t>(s.wg) * kWgmmaRows * s.m_tile *
                         sizeof(float), 16)
          : 0;
  const size_t splitk =
      sk_active(s) ? align_up(static_cast<size_t>(s.tiles) * 4, 16) : 0;
  return weight + act + scales + barriers + partial + splitk;
}

// 只用于报告的回退估算（正常路径直接问 Cfg::dynamic_smem_bytes(M,K)）
constexpr size_t dyn_smem_estimate(const ConfigSpec& s, int m, int k) {
  const size_t stages = static_cast<size_t>(num_stages(s));
  const int rows = m < s.m_tile ? m : s.m_tile;
  const size_t act = s.preload_act
                         ? static_cast<size_t>(rows) * k
                         : stages * static_cast<size_t>(s.subs) * rows * kKBlock;
  const size_t scale =
      s.scale_in_smem
          ? align_up(static_cast<size_t>(s.tiles) * s.m_tile * 2 *
                         (static_cast<size_t>(k) / kKBlock) * sizeof(float), 16)
          : 0;
  return fixed_smem_bytes(s) + align_up(act, 16) + scale;
}

constexpr bool reg_budget_ok(const ConfigSpec& s) {
  return (static_cast<size_t>(s.wg) * s.math_regs + s.prod_regs) * 128 <=
         kRegisterFileSize;
}
// __launch_bounds__((WG+1)*128, 1) 决定 ptxas 的寄存器硬上限（kernel agent 实测，
// kernels/README.md §8）：WG=4 -> 最多 96，WG=2 -> 168，WG=1 -> 232(上限 256)。
// 请求超过上限只会拿到上限值，所以「WG=4 + REG=112/168」是 REG=96 的重复配置。
constexpr int reg_cap(int wg) { return wg >= 4 ? 96 : (wg == 2 ? 168 : 232); }
constexpr bool reg_cap_ok(const ConfigSpec& s) {
  return s.math_regs <= reg_cap(s.wg);
}
// 累加器寄存器需求 ≈ M_TILE（CONTRACT §1：M_TILE=128 -> 128 accum regs/thread）。
// kernel 侧没有这条 static_assert，实例化了会因为 setmaxnreg 装不下累加器而 spill
// 甚至编译失败，所以 harness 侧加这道编译期保护。
constexpr bool accum_regs_ok(const ConfigSpec& s) {
  return s.math_regs >= s.m_tile + 40;
}
constexpr bool m_tile_ok(int m_tile) {
  return m_tile == 8 || m_tile == 16 || m_tile == 32 || m_tile == 64 ||
         m_tile == 128;
}

// 编译期可行性闸门（只有 feasible()==true 的 spec 才会被实例化）
constexpr bool feasible(const ConfigSpec& s) {
  if (!m_tile_ok(s.m_tile)) return false;
  if (s.bm <= 0 || s.bm > kWgmmaRows || (s.bm % 8) != 0) return false;
  if (s.wg != 1 && s.wg != 2 && s.wg != 4) return false;
  if (s.tiles < 1 || s.tiles > s.wg || (s.wg % s.tiles) != 0) return false;
  if (2 * s.tiles + 1 > 16) return false;         // named barrier id 上限
  if (s.stg < 1) return false;
  if (s.subs < 1 || s.subs > 64) return false;
  if (s.math_regs < 24 || s.math_regs > 256 || (s.math_regs % 8) != 0) return false;
  if (s.prod_regs < 24 || s.prod_regs > 256 || (s.prod_regs % 8) != 0) return false;
  if (!sk_factor_legal(s.splitk_factor)) return false;   // kernels static_assert
  if (s.splitk_factor == 1 && s.splitk_cta) return false;  // 自相矛盾
  // epi/pdl 独立（kernels 侧 static_assert 已放宽，见 --pdl-placement 消融）
  if (s.preload_act && !(s.m_tile == 8 && s.prepack)) return false;
  if (!reg_budget_ok(s)) return false;
  if (!accum_regs_ok(s)) return false;
  if (fixed_smem_bytes(s) > kMaxDynamicSmemBytes) return false;
  // CONTRACT §3：M_TILE>=16 时 TILES_PER_CTA = NUM_MATH_WG（无 intra-CTA splitK
  // reduce，避免 partial smem 爆炸）
  if (s.m_tile >= 16 && wgs_per_tile(s) != 1) return false;
  return true;
}

// ------------------------------------------------------- canonical BM 选择 --
// 规则（orchestrator 补充）：BM ∈ {32,40,48,56,64}；pad==0 优先；其次 output tile 数
// 最贴近 128（v2 甜区：N=6144->BM48=128 tiles，N=7168->BM56=128 tiles，
// N=2048->BM32=64 tiles 优于 BM64=32 tiles）；再次 BM 大者优先。
constexpr int bm_score(int n, int bm) {
  const int tiles = num_output_tiles(n, bm);
  const int pad = tiles * bm - n;
  const int dist = tiles > kTargetCtaCount ? tiles - kTargetCtaCount
                                           : kTargetCtaCount - tiles;
  return pad * 1000000 + dist * 1000 - bm;   // pad 权重压过 tile 距离
}
constexpr int choose_bm(int n) {
  int best = kCandidateBm[0];
  int best_score = bm_score(n, best);
  for (int i = 1; i < kCandidateBmCount; ++i) {
    const int sc = bm_score(n, kCandidateBm[i]);
    if (sc < best_score) { best_score = sc; best = kCandidateBm[i]; }
  }
  return best;
}
// 按 score 排好序的 BM 列表（canonical 不可行时的回退顺序，也用于报告）
constexpr int bm_rank_list(int n, int* out) {
  int list[kCandidateBmCount] = {};
  for (int i = 0; i < kCandidateBmCount; ++i) list[i] = kCandidateBm[i];
  for (int i = 0; i < kCandidateBmCount; ++i)
    for (int j = i + 1; j < kCandidateBmCount; ++j)
      if (bm_score(n, list[j]) < bm_score(n, list[i])) {
        const int t = list[i]; list[i] = list[j]; list[j] = t;
      }
  for (int i = 0; i < kCandidateBmCount; ++i) out[i] = list[i];
  return kCandidateBmCount;
}

// --------------------------------------------------------- canonical 七步 ---
// 已与 kernels/README.md §6（kernel agent 2026-09-13 定稿）逐行对齐：
//
//  §6.2 M_TILE=8：
//    S0/S1  BM=64, WG=4, REG=112, STG=2, TILES=1, SUBS=1   (CONTRACT §3 baseline)
//    S2/S3  同上，只把 SUBS 提到 2
//    S4-S6  BM=<每形状选值>, WG=2, REG=80, STG=3, TILES=1, SUBS=2
//    开关：PREPACK>=S3, PDL>=S1, SPLITK=S5（EPI 不进阶梯，仅作 --pdl-placement 消融）
//    注意 REG=112 + WG=4 会被 ptxas 夹到 96（README §8），但 canonical 仍按
//    CONTRACT §3 的字面值写 112，配置串里如实反映。
//
//  §6.3 M_TILE>=16：拓扑统一 WG=2, TILES=2, STG=2（TILES=WG -> 无 intra-CTA
//    splitK reduce），BM 用每形状选值；REG: 16->112, 32/64/128->168；
//    SUBS: 16/32/64 -> 2，**128 -> 1**（累加器 (SUBS+1)*M_TILES*4 超 168 上限）；
//    SCALE_IN_SMEM: 16 -> true，>=32 -> false（scale 表在 K=16384 时要 128KB）。
//    开关叠加方式与 M_TILE=8 相同。
//
//  §8 M_TILE=128 的第二条 canonical：BM=64, WG=1, REG=232, TILES=1, SUBS=1
//    -> 0 spill（canonical WG=2/REG=168 有 20B spill stores / 52B loads）。
//    两条都跑，config 串里用 wg1reg232=1 区分（orchestrator 2026-09-13）。
constexpr ConfigSpec canonical_switches(ConfigSpec s, int step) {
  s.pdl = step >= 1;             // S1 pdl（触发点=producer；epilogue 重叠是它的副产品）
  s.prepack = step >= 3;         // S3 prepack
  s.epi_overlap = false;         // 不再进主阶梯：第二次触发在 per-CTA 语义下被 S1 的早触发覆盖，
                                 // 增量实测≈0；只作为 --pdl-placement 消融变体存在
  s.splitk_cta = step >= 5;      // S5 splitk
  return s;
}

constexpr ConfigSpec canonical_base(int step, int m_tile, int bm) {
  ConfigSpec s = make_spec(m_tile, bm, 2, 168, 2, 2, 1, false, false, false,
                           false, kDefaultProducerRegs, false, false);
  s.scale_in_smem = (m_tile <= 16);
  if (m_tile == 8) {
    if (step <= 3) {
      s.bm = 64;                       // §6.2：S0-S3 一律 BM=64（零 pad，v2 baseline）
      s.wg = 4; s.math_regs = 112; s.stg = 2; s.tiles = 1; s.subs = 1;
      if (step >= 2) s.subs = 2;       // S2 blockk
    } else {
      s.wg = 2; s.math_regs = 80; s.stg = 3; s.tiles = 1; s.subs = 2;
    }
    return canonical_switches(s, step);
  }
  // ---- M_TILE >= 16（README §6.3）----
  s.wg = 2; s.tiles = 2; s.stg = 2;
  s.math_regs = (m_tile == 16) ? 112 : 168;
  s.subs = (m_tile >= 128) ? 1 : 2;
  return canonical_switches(s, step);
}

// M_TILE=128 的第二条 canonical（README §8：WG=1/REG=232/BM=64/TILES=1/SUBS=1，0 spill）
constexpr ConfigSpec canonical_alt(int step, int m_tile) {
  if (m_tile != 128) return make_spec(-1, 0, 1, 24, 1, 1, 1, false, false, false,
                                      false, 32, false, false);  // feasible()==false
  ConfigSpec s = make_spec(128, 64, 1, 232, 2, 1, 1, false, false, false, false,
                           kDefaultProducerRegs, false, false);
  s.scale_in_smem = false;
  s = canonical_switches(s, step);
  s.wg1reg232 = true;
  return s;
}

// 自动降级：SUBS -> STG -> WG/TILES，直到 feasible（CONTRACT §3 / README §6.3
// 「跑不通时降级 STG 3->2，再 SUBS 2->1」）。降级发生 -> downgraded=true。
constexpr ConfigSpec canonical_for(int step, int m_tile, int bm) {
  ConfigSpec s = canonical_base(step, m_tile, bm);
  if (feasible(s)) return s;
  for (int guard = 0; guard < 16; ++guard) {
    if (s.stg > 2) { s.stg = 2; s.downgraded = true; }
    else if (s.subs > 1) { s.subs = 1; s.downgraded = true; }
    else if (s.stg > 1) { s.stg = 1; s.downgraded = true; }
    else if (s.wg > 1) {
      s.wg /= 2;
      if (s.tiles > s.wg) s.tiles = s.wg;
      s.downgraded = true;
    } else { break; }
    if (feasible(s)) return s;
  }
  return s;   // 仍不可行：调用方用 feasible() 判断后跳过并告警
}

// ---------------------------------------------------------------------------
// 每形状 canonical override（--tune 跑完把 winner 填进来，然后重编）
//   orchestrator 2026-09-13：tune-best 作为该形状 S4/S5/S6 的 base config
//   （S5 = base+EPI，S6 = base+EPI+SPLITK(条件)）。
//   n/k 匹配 models.json 的形状；m_tile=0 表示对该形状所有 M_TILE 生效。
struct ShapeOverride {
  int n, k, m_tile;
  int bm, wg, regs, stg, tiles, subs;
};
inline constexpr ShapeOverride kShapeOverrides[] = {
    // ---- tune winners @ binding M（m8 用 M=8 行；m128 不 override，v1 canonical 更优）----
    {7168, 16384, 8, 56, 4, 96, 3, 1, 2},  // 3410 GB/s @M=8  M_TILE=8;BM=56;WG=4;REG=96;STG=3;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {7168, 16384, 16, 64, 1, 80, 4, 1, 2},  // 2369 GB/s @M=16  M_TILE=16;BM=64;WG=1;REG=80;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {7168, 16384, 32, 64, 1, 112, 4, 1, 2},  // 1728 GB/s @M=32  M_TILE=32;BM=64;WG=1;REG=112;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {7168, 16384, 64, 64, 1, 112, 2, 1, 2},  // 634 GB/s @M=64  M_TILE=64;BM=64;WG=1;REG=112;STG=2;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {6144, 16384, 8, 56, 2, 80, 2, 1, 2},  // 3186 GB/s @M=8  M_TILE=8;BM=56;WG=2;REG=80;STG=2;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {6144, 16384, 16, 64, 1, 80, 4, 1, 2},  // 2046 GB/s @M=16  M_TILE=16;BM=64;WG=1;REG=80;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {6144, 16384, 32, 64, 1, 112, 4, 1, 2},  // 1503 GB/s @M=32  M_TILE=32;BM=64;WG=1;REG=112;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {6144, 16384, 64, 48, 2, 168, 3, 2, 2},  // 544 GB/s @M=64  M_TILE=64;BM=48;WG=2;REG=168;STG=3;TILES=2;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {7168, 12288, 8, 56, 1, 112, 4, 1, 2},  // 3342 GB/s @M=8  M_TILE=8;BM=56;WG=1;REG=112;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {7168, 12288, 16, 56, 1, 80, 4, 1, 2},  // 2336 GB/s @M=16  M_TILE=16;BM=56;WG=1;REG=80;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {7168, 12288, 32, 64, 1, 112, 4, 1, 2},  // 1706 GB/s @M=32  M_TILE=32;BM=64;WG=1;REG=112;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {7168, 12288, 64, 64, 2, 168, 3, 2, 2},  // 639 GB/s @M=64  M_TILE=64;BM=64;WG=2;REG=168;STG=3;TILES=2;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {6144, 8192, 8, 56, 1, 112, 4, 1, 2},  // 3172 GB/s @M=8  M_TILE=8;BM=56;WG=1;REG=112;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {6144, 8192, 16, 64, 1, 80, 4, 1, 2},  // 2038 GB/s @M=16  M_TILE=16;BM=64;WG=1;REG=80;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {6144, 8192, 32, 64, 1, 112, 4, 1, 2},  // 1538 GB/s @M=32  M_TILE=32;BM=64;WG=1;REG=112;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {6144, 8192, 64, 64, 2, 168, 3, 2, 2},  // 579 GB/s @M=64  M_TILE=64;BM=64;WG=2;REG=168;STG=3;TILES=2;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {2048, 4096, 8, 56, 4, 96, 3, 1, 2},  // 2216 GB/s @M=8  M_TILE=8;BM=56;WG=4;REG=96;STG=3;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {2048, 4096, 16, 64, 1, 232, 4, 1, 2},  // 1267 GB/s @M=16  M_TILE=16;BM=64;WG=1;REG=232;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=1
    {2048, 4096, 32, 56, 1, 80, 4, 1, 2},  // 1003 GB/s @M=32  M_TILE=32;BM=56;WG=1;REG=80;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
    {2048, 4096, 64, 64, 1, 232, 4, 1, 2},  // 398 GB/s @M=64  M_TILE=64;BM=64;WG=1;REG=232;STG=4;TILES=1;SUBS=2;PREPACK=1;PDL=1;EPI=0;SPLITK=0;PR=32;SPW=0;PLA=0;SISM=0
};
inline constexpr int kShapeOverrideCount =
    static_cast<int>(sizeof(kShapeOverrides) / sizeof(kShapeOverrides[0]));

constexpr bool has_override(int n, int k, int m_tile) {
  for (int i = 0; i < kShapeOverrideCount; ++i) {
    const ShapeOverride& o = kShapeOverrides[i];
    if (o.n == n && o.k == k && (o.m_tile == 0 || o.m_tile == m_tile) && o.bm > 0)
      return true;
  }
  return false;
}
constexpr ShapeOverride get_override(int n, int k, int m_tile) {
  for (int i = 0; i < kShapeOverrideCount; ++i) {
    const ShapeOverride& o = kShapeOverrides[i];
    if (o.n == n && o.k == k && (o.m_tile == 0 || o.m_tile == m_tile) && o.bm > 0)
      return o;
  }
  return ShapeOverride{0, 0, 0, 0, 0, 0, 0, 0, 0};
}

// S0-S3(M_TILE=8) 固定 BM=64；其余按形状选（整除 N + CTA 数贴近 128）
constexpr int canonical_bm_for_step(int step, int m_tile, int n) {
  if (m_tile == 8 && step <= 3) return 64;
  return choose_bm(n);
}

// 某 (step,m_tile,N) 的 canonical BM：优先上面的规则，不可行按 rank 回退
constexpr int canonical_bm(int step, int m_tile, int n) {
  int rank[kCandidateBmCount] = {};
  bm_rank_list(n, rank);
  const int want = canonical_bm_for_step(step, m_tile, n);
  if (feasible(canonical_for(step, m_tile, want))) return want;
  for (int i = 0; i < kCandidateBmCount; ++i)
    if (feasible(canonical_for(step, m_tile, rank[i]))) return rank[i];
  return rank[0];
}

// ---------------------------------------------------------------------------
// S6 的 SPLIT_K_CTA 是**按形状 + 按 M 条件启用**的（orchestrator 2026-09-13 数据审查）：
//   开的前提 = 「不开 splitK 的 CTA 数 ×2 <= SM 数(78)」**且 M >= 8**。
//   判据来自实测（article agent 数据审查 2026-09-13）：qwen36 64 CTA 时 M<=8 开
//   splitK 反降 15-22%（翻倍后 128 CTA = 1.64 wave，量化损失 > 收益），32 CTA 时
//   M>=16 升 35-49%（翻倍后 64 CTA 仍单 wave）；kimi/glm/dspk/minimax 128 CTA
//   全关（M=64 实测 580 vs 634）。即：翻倍后仍落在一个 wave 内才开。
// 关掉时 S6 的 config 与 S5 相同，文章里如实写「该形状/该 M 下 split-K 不触发」。
inline constexpr int kSmCount = 78;                 // H20 的 SM 数
inline constexpr int kSplitKMinM = 8;  // 配合 worth_it 区间判据               // M < 8 不开 split-K
constexpr bool splitk_worth_it(int n, int bm) {
  const int ctas = num_output_tiles(n, bm);
  // 数据定标（2026-09-13 ablation + v4）：只有「翻倍后跨进 >=1 个 wave」才正收益：
  // 64->128 CTA +35~49%（qwen 旧 BM32, M>=16）；32->64 与 37->74 均负（仍 <1 wave
  // 却多付 gmem 两段 reduce）。即 ctas ∈ [ceil(78/2), 78)。
  return ctas < kSmCount && ctas * 2 >= kSmCount;
}
constexpr bool splitk_for_shape(int step, int m_tile, int n) {
  if (step < 6) return false;
  return splitk_worth_it(n, canonical_bm_for_step(6, m_tile, n));
}
// shape-aware 版：override 存在时用 override 的 BM 判 splitK（qwen m16 BM=32 -> 64 CTA
// -> 翻倍 128 仍值得开；BM=64 -> 32 CTA -> 翻倍 64 < 78 不开）。
constexpr bool splitk_for_shape_k(int step, int m_tile, int n, int k) {
  if (step < 6) return false;
  const int bm = (has_override(n, k, m_tile) && m_tile >= 8)
                     ? get_override(n, k, m_tile).bm
                     : canonical_bm_for_step(6, m_tile, n);
  return splitk_worth_it(n, bm);
}
inline bool splitk_enabled_at_runtime(int step, int m_tile, int n, int m) {
  return step >= 5 && m >= kSplitKMinM && splitk_for_shape(step, m_tile, n);
}
inline bool splitk_enabled_at_runtime_k(int step, int m_tile, int n, int k, int m) {
  return step >= 5 && m >= kSplitKMinM && splitk_for_shape_k(step, m_tile, n, k);
}
// runtime 选 BM：step>=4 且该形状有 override -> override.bm
inline int canonical_bm_shape(int step, int m_tile, int n, int k) {
  if (step >= 4 && has_override(n, k, m_tile)) {
    const int obm = get_override(n, k, m_tile).bm;
    if (obm > 0) return obm;
  }
  return canonical_bm(step, m_tile, n);
}
// 归一化：on/off 之后 splitk_factor 一定是显式值（1 或 >=2），这样两条路径
// 产生的 config 串与 KernelConfig 模板参数都一致，registry 里不会重复。
constexpr ConfigSpec with_splitk(const ConfigSpec& in, bool on) {
  ConfigSpec s = in;
  s.splitk_cta = on;
  s.splitk_factor = on ? (in.splitk_factor >= 2 ? in.splitk_factor : 2) : 1;
  return s;
}
constexpr ConfigSpec with_splitk_factor(const ConfigSpec& in, int factor) {
  ConfigSpec s = in;
  s.splitk_factor = factor;
  s.splitk_cta = factor >= 2;
  return s;
}

// ---- S 维展开的闸门（控制实例化数）----------------------------------------
// 只有「值得开 split-K」或「小 N 形状」才把 S∈{2,4,8} 这一维展开：
//   splitk_worth_it(n,bm)  ->  CTA 数落在 [39,78)，翻倍刚好进一个 wave
//   n <= 4096              ->  qwen36 这类小 N，S 是唯一的并行度来源
// 其余形状（kimi/glm/dpsk/minimax，128 CTA 起步）只留 S=1/2，不白烧编译时间。
constexpr bool sk_factor_expand(int n, int bm) {
  return n > 0 && (splitk_worth_it(n, bm) || n <= 4096);
}
// K 已知时把整除性也在编译期筛掉（运行时 Cfg::validate 仍是最终裁判）：
//   K % (128 * SUBS * WGsPerTile * S) == 0
constexpr bool sk_factor_ok_k(int k, int subs, int wpt, int factor) {
  if (k <= 0) return true;
  return (k % (128 * subs * wpt * factor)) == 0;
}


// ------------------------------------------------------------ tune 网格 -----
// CONTRACT §6：BM∈{32,48,64}, WG∈{1,2,4}, STG∈{2,3,4}, SUBS∈{1,2},
//              TILES∈{1,2}, MATH_REGS∈{80,112,168}
// orchestrator 补充：BM 再加 40/56
inline constexpr int kTuneBm[] = {32, 40, 48, 56, 64};
inline constexpr int kTuneStg[] = {2, 3, 4};
inline constexpr int kTuneSubs[] = {1, 2};
// (WG, TILES, REG) 的合法组合：REG 不超过 ptxas 的 __launch_bounds__ 上限
// （WG=4 -> 96，WG=2 -> 168，WG=1 -> 232），所以 WG=4 只留 REG=96，
// 删掉与它等价的 WG=4/REG=112 与 WG=4/REG=168（kernel agent 2026-09-13）。
inline constexpr int kTuneWtr[][3] = {
    {1, 1, 80}, {1, 1, 112}, {1, 1, 168}, {1, 1, 232},
    {2, 1, 80}, {2, 1, 112}, {2, 1, 168},
    {2, 2, 80}, {2, 2, 112}, {2, 2, 168},
    {4, 1, 96}, {4, 2, 96}, {4, 4, 96},
};
inline constexpr int kTuneBmN = 5;
inline constexpr int kTuneStgN = 3;
inline constexpr int kTuneSubsN = 2;
inline constexpr int kTuneWtrN = 13;
inline constexpr int kTuneGridSize =
    kTuneBmN * kTuneWtrN * kTuneStgN * kTuneSubsN;   // 5*13*3*2 = 390

struct TuneGeom { int bm, wg, stg, subs, tiles, regs; };

// ---------------------------------------------------------------------------
// s4 网格（orchestrator 2026-09-13 修订指令）：只扫 S4+ 的旋钮，SUBS=2、PREPACK=1
// 固定，BM∈{32,48,56,64} x WG∈{1,2,4} x STG∈{2,3,4} x TILES∈{1,2} x REG∈{80,112,168}
// （WG=4 时 ptxas 只给 96 regs，所以 WG=4 只留一个 REG=96 条目，避免重复配置）。
// `make tune-bin TUNE_GRID=s4` 把它烘进 bench_decode_tune。
inline constexpr int kTuneS4Bm[] = {32, 48, 56, 64};
inline constexpr int kTuneS4Stg[] = {2, 3, 4};
inline constexpr int kTuneS4BmN = 4;
inline constexpr int kTuneS4StgN = 3;
inline constexpr int kTuneS4GridSize = kTuneS4BmN * kTuneWtrN * kTuneS4StgN;  // 156

constexpr TuneGeom tune_s4_geom_at(int i) {
  int x = i;
  const int stg = kTuneS4Stg[x % kTuneS4StgN]; x /= kTuneS4StgN;
  const int w = x % kTuneWtrN; x /= kTuneWtrN;
  const int bm = kTuneS4Bm[x % kTuneS4BmN];
  return TuneGeom{bm, kTuneWtr[w][0], stg, /*subs=*/2, kTuneWtr[w][1],
                  kTuneWtr[w][2]};
}

constexpr ConfigSpec tune_s4_at(int m_tile, int i) {
  const TuneGeom g = tune_s4_geom_at(i);
  ConfigSpec s = canonical_for(4, m_tile, g.bm);   // 继承 S4 的 bool 特性
  s.bm = g.bm; s.wg = g.wg; s.stg = g.stg;
  s.subs = g.subs; s.tiles = g.tiles; s.math_regs = g.regs;
  s.prepack = true; s.pdl = true; s.epi_overlap = false; s.splitk_cta = false;
  s.downgraded = false;
  if (m_tile >= 16 && g.tiles != g.wg)
    return make_spec(-1, 0, 1, 24, 1, 1, 1, false, false, false, false, 32,
                     false, false);                        // feasible()==false
  if (s.tiles > s.wg) s.tiles = s.wg;
  if (s.math_regs > reg_cap(s.wg))
    return make_spec(-1, 0, 1, 24, 1, 1, 1, false, false, false, false, 32,
                     false, false);
  return s;
}

constexpr TuneGeom tune_geom_at(int i) {
  int x = i;
  const int subs = kTuneSubs[x % kTuneSubsN]; x /= kTuneSubsN;
  const int stg = kTuneStg[x % kTuneStgN]; x /= kTuneStgN;
  const int w = x % kTuneWtrN; x /= kTuneWtrN;
  const int bm = kTuneBm[x % kTuneBmN];
  return TuneGeom{bm, kTuneWtr[w][0], stg, subs, kTuneWtr[w][1], kTuneWtr[w][2]};
}

// tune 变体 = 该 step 的布尔特性（PREPACK/PDL/EPI/SPLITK）+ 网格几何
constexpr ConfigSpec tune_at(int step, int m_tile, int i) {
  const TuneGeom g = tune_geom_at(i);
  ConfigSpec s = canonical_for(step, m_tile, g.bm);   // 继承 step 的 bool 特性
  s.bm = g.bm; s.wg = g.wg; s.stg = g.stg;
  s.subs = g.subs; s.tiles = g.tiles; s.math_regs = g.regs;
  s.downgraded = false;
  // CONTRACT §3 / README §6.3：M_TILE>=16 时 TILES 恒等于 WG，网格里 tiles!=wg
  // 的组合直接判为不可行（否则会重复实例化同一个 Cfg）
  if (m_tile >= 16 && g.tiles != g.wg)
    return make_spec(-1, 0, 1, 24, 1, 1, 1, false, false, false, false, 32,
                     false, false);                        // feasible()==false
  if (s.tiles > s.wg) s.tiles = s.wg;
  // ptxas 上限之外的 REG 是重复配置，删掉（kernel agent 2026-09-13）
  if (s.math_regs > reg_cap(s.wg))
    return make_spec(-1, 0, 1, 24, 1, 1, 1, false, false, false, false, 32,
                     false, false);
  // （旧：epi 无 pdl 时强制关；现已允许独立组合）
  if (s.preload_act && !(m_tile == 8 && s.prepack)) s.preload_act = false;
  return s;
}

// ---------------------------------------------------------------------------
// tune 网格的 S 维（step 6 专用，splitk-S agent 2026-09-13）
// ---------------------------------------------------------------------------
// 索引空间：step==6 时 i = base_idx * kTuneSkSlots + slot
//   slot 0      = 基础网格自带的 S=2（canonical_switches 在 step6 把 splitk_cta
//                 置真，经 sk_factor_of 归一化就是 S=2），
//   slot 1 / 2  = kTuneSkFactors 里的 4 / 8。
// 于是 {2,4,8} 三档都覆盖，且不会把同一个 Cfg 实例化两遍（重复实例化会让
// tune CSV 出现同 config 的两行）。
//
// 展开闸门（控制实例化数，指令原话「仅当 splitk_worth_it 或形状 N<=4096 时展开」）：
// 网格本身是形状无关的，所以形状必须由编译期宏给：
//   make tune-bin TUNE_SKF=1 TUNE_SHAPE_N=2048 TUNE_SHAPE_K=4096 TUNE_GRID=s4 TUNE_STEPS=6
// TUNE_SHAPE_N 缺省 0 -> kTuneSkExpand=false -> 索引空间与实例化数和以前完全一样。
// 每个 (bm, subs, wg/tiles, S) 还要过 sk_factor_expand / sk_factor_ok_k 两道筛，
// 不合法的直接返回 feasible()==false 的 spec（不实例化）。
#ifndef BENCH_TUNE_SHAPE_N
#define BENCH_TUNE_SHAPE_N 0
#endif
#ifndef BENCH_TUNE_SHAPE_K
#define BENCH_TUNE_SHAPE_K 0
#endif
inline constexpr int kTuneShapeN = BENCH_TUNE_SHAPE_N;
inline constexpr int kTuneShapeK = BENCH_TUNE_SHAPE_K;
inline constexpr int kTuneSkFactors[] = {4, 8};          // slot 1..2（slot 0 = S=2）
inline constexpr int kTuneSkSlots = 3;

constexpr bool tune_sk_expand_any() {
  bool any = false;
  if (kTuneShapeN > 0) {            // 形状未知（默认）-> 一律不展开，实例化数不变
    for (int i = 0; i < kTuneBmN && !any; ++i)
      any = sk_factor_expand(kTuneShapeN, kTuneBm[i]);
    for (int i = 0; i < kTuneS4BmN && !any; ++i)
      any = sk_factor_expand(kTuneShapeN, kTuneS4Bm[i]);
  }
  return any;
}
inline constexpr bool kTuneSkExpand = tune_sk_expand_any();

constexpr ConfigSpec tune_infeasible() {
  return make_spec(-1, 0, 1, 24, 1, 1, 1, false, false, false, false, 32,
                   false, false);
}

// 基础网格（slot 维折叠前）
constexpr ConfigSpec tune_base_at(int step, int m_tile, int i) {
#if defined(BENCH_TUNE_GRID_S4)
  (void)step;
  return tune_s4_at(m_tile, i);
#else
  return tune_at(step, m_tile, i);
#endif
}

constexpr ConfigSpec tune_slot_at(int step, int m_tile, int i) {
  if (step == 5 && kTuneSkExpand) {
    const ConfigSpec base = tune_base_at(step, m_tile, i / kTuneSkSlots);
    const int slot = i % kTuneSkSlots;
    if (slot == 0) return base;                       // S=2（基础网格自带）
    if (!feasible(base)) return base;
    const int factor = kTuneSkFactors[slot - 1];
    if (!sk_factor_expand(kTuneShapeN, base.bm)) return tune_infeasible();
    if (!sk_factor_ok_k(kTuneShapeK, base.subs, wgs_per_tile(base), factor))
      return tune_infeasible();
    return with_splitk_factor(base, factor);
  }
  return tune_base_at(step, m_tile, i);
}

constexpr int tune_grid_size_for(int step) {
#if defined(BENCH_TUNE_GRID_S4)
  const int base = kTuneS4GridSize;
#else
  const int base = kTuneGridSize;
#endif
  return (step == 5 && kTuneSkExpand) ? base * kTuneSkSlots : base;
}

// -------------------------------------------------------------- 文本化 ------
// CONTRACT §6：config 字段内部用 ';' 分隔键值对，不许出现逗号
inline std::string config_to_string(const ConfigSpec& s) {
  std::string r;
  r += "M_TILE=" + std::to_string(s.m_tile);
  r += ";BM=" + std::to_string(s.bm);
  r += ";WG=" + std::to_string(s.wg);
  r += ";REG=" + std::to_string(s.math_regs);
  r += ";STG=" + std::to_string(s.stg);
  r += ";TILES=" + std::to_string(s.tiles);
  r += ";SUBS=" + std::to_string(s.subs);
  r += ";PREPACK=" + std::to_string(static_cast<int>(s.prepack));
  r += ";PDL=" + std::to_string(static_cast<int>(s.pdl));
  r += ";EPI=" + std::to_string(static_cast<int>(s.epi_overlap));
  // SPLITK 打**有效**开关（plot.py 认 "SPLITK=1"），SK 打有效宽度 S
  r += ";SPLITK=" + std::to_string(static_cast<int>(sk_active(s)));
  r += ";SK=" + std::to_string(sk_factor_of(s));
  r += ";PR=" + std::to_string(s.prod_regs);
  r += ";SPW=" + std::to_string(static_cast<int>(s.single_producer_warp));
  r += ";PLA=" + std::to_string(static_cast<int>(s.preload_act));
  r += ";SISM=" + std::to_string(static_cast<int>(s.scale_in_smem));
  if (s.wg1reg232) r += ";wg1reg232=1";      // orchestrator 2026-09-13 要求的标记
  if (s.downgraded) r += ";DOWNGRADED=1";
  return r;
}

}  // namespace bench
