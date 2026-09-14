// SPDX-License-Identifier: MIT
//
// ============================================================================
//  decode_gemm smoke test — the completion gate for kernels/
// ============================================================================
//  Instantiates every step of the CONTRACT §3 ladder (S0..S6) at M_TILE=8, plus
//  M_TILE=16/32/64/128, both weight layouts (raw [N,K] and tile-major packed),
//  SPLIT_K_CTA, and every optional template switch, then checks all of them
//  against a naive fp32 host reference (rel_l2 <= 2e-3, the CONTRACT §4.9 gate).
//
//  Shapes: N=6144/K=7168 (CONTRACT §4.9), N=7168/K=16384 (kimi_k3/dpsk family,
//  also exercises the "N not divisible by BM" padding predicate),
//  N=2048/K=4096 (qwen36, small N -> split-K's home turf) and
//  N=7168/K=12288 (kimi_k3 o_proj, the real model shape).
//
//  The S6 gate covers the *generalised* S-way CTA split-K: SPLITK_FACTOR
//  in {2,4,8} on both the small-N shape (where split-K is supposed to win) and
//  the big-N shape (where it is supposed to lose), at M=1 and M=8.  Every case
//  also re-reads the semaphore array and requires it to be back at 0 (the
//  self-reset contract), and re-launches to catch stale handshake state.
//
//  Build:  build_smoke.sh      Run: run_smoke.sh   (both go through gpurun_dg.sh)
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "decode_gemm.cuh"
#include "weight_prepack.cuh"

namespace {

constexpr float kFp8Max = 448.0f;
constexpr float kQuantizationEps = 1.0e-4f;
constexpr double kRelL2Threshold = 2.0e-3;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    const cudaError_t status_ = (expr);                                         \
    if (status_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                   cudaGetErrorString(status_), __FILE__, __LINE__);            \
      std::exit(2);                                                             \
    }                                                                           \
  } while (0)

float g_fp8_lut[256];

void init_fp8_lut() {
  for (int i = 0; i < 256; ++i) {
    __nv_fp8_e4m3 value;
    value.__x = static_cast<uint8_t>(i);
    const float f = static_cast<float>(value);
    g_fp8_lut[i] = std::isfinite(f) ? f : 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Host problem generation (same recipe as the reference harness bench_v2.cu:
// randn(0, 1/sqrt(K)) rounded to bf16, then per-128(K)-block fp8 quantisation
// with scale = max(amax, 1e-4)/448).
// ---------------------------------------------------------------------------
struct ShapeData {
  int N = 0;
  int K = 0;
  int num_scale_rows = 0;
  std::vector<uint8_t> weight;        // [N,K] fp8 bytes, raw layout
  std::vector<float> w_scales;        // [ceil(N/128), K/128]
  __nv_fp8_e4m3* d_weight_raw = nullptr;
  float* d_w_scales = nullptr;
  std::map<int, __nv_fp8_e4m3*> d_weight_packed;   // BM -> packed device buffer
};

struct MData {
  int M = 0;
  std::vector<uint8_t> act;           // [M,K] fp8 bytes
  std::vector<float> a_scales;        // [M, K/128]
  std::vector<float> reference;       // [M,N] fp32, bf16-rounded
  __nv_fp8_e4m3* d_act = nullptr;
  float* d_a_scales = nullptr;
  __nv_bfloat16* d_output = nullptr;
  float* d_splitk_ws = nullptr;
  int* d_splitk_sem = nullptr;
  int sem_ints = 0;
};

float round_to_bf16(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }

void generate_weight(ShapeData* sd, uint64_t seed) {
  const int N = sd->N;
  const int K = sd->K;
  const int kb_count = K / 128;
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(K));
  sd->weight.assign(static_cast<size_t>(N) * K, 0);
  sd->w_scales.assign(static_cast<size_t>(sd->num_scale_rows) * kb_count, 1.0f);

  std::vector<float> block(128 * 128);
  for (int nb = 0; nb < sd->num_scale_rows; ++nb) {
    for (int kb = 0; kb < kb_count; ++kb) {
      float amax = 0.0f;
      for (int i = 0; i < 128 * 128; ++i) {
        block[i] = round_to_bf16(normal(rng) * inv_sqrt_k);
        amax = std::max(amax, std::fabs(block[i]));
      }
      const float scale = std::max(amax, kQuantizationEps) / kFp8Max;
      sd->w_scales[static_cast<size_t>(nb) * kb_count + kb] = scale;
      const int rows = std::min(128, N - nb * 128);
      for (int nn = 0; nn < rows; ++nn) {
        uint8_t* dst = &sd->weight[static_cast<size_t>(nb * 128 + nn) * K +
                                   static_cast<size_t>(kb) * 128];
        const float* src = &block[static_cast<size_t>(nn) * 128];
        for (int kk = 0; kk < 128; ++kk) {
          const float q = std::min(std::max(src[kk] / scale, -kFp8Max), kFp8Max);
          dst[kk] = __nv_fp8_e4m3(q).__x;
        }
      }
    }
  }
  CUDA_CHECK(cudaMalloc(&sd->d_weight_raw, sd->weight.size()));
  CUDA_CHECK(cudaMemcpy(sd->d_weight_raw, sd->weight.data(), sd->weight.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&sd->d_w_scales, sd->w_scales.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(sd->d_w_scales, sd->w_scales.data(),
                        sd->w_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void generate_activation(MData* md, const ShapeData& sd, uint64_t seed) {
  const int M = md->M;
  const int K = sd.K;
  const int kb_count = K / 128;
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(K));
  md->act.assign(static_cast<size_t>(M) * K, 0);
  md->a_scales.assign(static_cast<size_t>(M) * kb_count, 1.0f);
  for (int m = 0; m < M; ++m) {
    for (int kb = 0; kb < kb_count; ++kb) {
      float values[128];
      float amax = 0.0f;
      for (int kk = 0; kk < 128; ++kk) {
        values[kk] = round_to_bf16(normal(rng) * inv_sqrt_k);
        amax = std::max(amax, std::fabs(values[kk]));
      }
      const float scale = std::max(amax, kQuantizationEps) / kFp8Max;
      md->a_scales[static_cast<size_t>(m) * kb_count + kb] = scale;
      uint8_t* dst = &md->act[static_cast<size_t>(m) * K + static_cast<size_t>(kb) * 128];
      for (int kk = 0; kk < 128; ++kk) {
        const float q = std::min(std::max(values[kk] / scale, -kFp8Max), kFp8Max);
        dst[kk] = __nv_fp8_e4m3(q).__x;
      }
    }
  }
}

// Naive fp32 host reference, accumulated per 128-wide k block and scaled by
// (w_scale * a_scale) — i.e. exactly the numerical path the kernel implements,
// so the residual is accumulation order + the final bf16 rounding only.
void compute_reference(const MData& md, const ShapeData& sd) {
  const int M = md.M, N = sd.N, K = sd.K;
  const int kb_count = K / 128;
  std::vector<float>& ref = const_cast<std::vector<float>&>(md.reference);
  ref.assign(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> w_row(128);
  std::vector<float> act_block(static_cast<size_t>(M) * 128);
  for (int kb = 0; kb < kb_count; ++kb) {
    for (int m = 0; m < M; ++m) {
      const uint8_t* a = &md.act[static_cast<size_t>(m) * K + static_cast<size_t>(kb) * 128];
      float* dst = &act_block[static_cast<size_t>(m) * 128];
      for (int j = 0; j < 128; ++j) dst[j] = g_fp8_lut[a[j]];
    }
    for (int n = 0; n < N; ++n) {
      const uint8_t* w = &sd.weight[static_cast<size_t>(n) * K + static_cast<size_t>(kb) * 128];
      for (int j = 0; j < 128; ++j) w_row[j] = g_fp8_lut[w[j]];
      const float ws = sd.w_scales[static_cast<size_t>(n / 128) * kb_count + kb];
      for (int m = 0; m < M; ++m) {
        const float* a = &act_block[static_cast<size_t>(m) * 128];
        float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
        for (int j = 0; j < 128; j += 4) {
          r0 += w_row[j] * a[j];
          r1 += w_row[j + 1] * a[j + 1];
          r2 += w_row[j + 2] * a[j + 2];
          r3 += w_row[j + 3] * a[j + 3];
        }
        const float raw = (r0 + r1) + (r2 + r3);
        const float combined = ws * md.a_scales[static_cast<size_t>(m) * kb_count + kb];
        float* acc = &ref[static_cast<size_t>(m) * N + n];
        *acc = std::fma(combined, raw, *acc);
      }
    }
  }
  for (float& v : ref) v = round_to_bf16(v);
}

// Largest SPLITK_FACTOR any case below instantiates; the workspace is [S][M][N]
// fp32, so size it once for the worst case (8 * M * N * 4 B).
constexpr int kMaxSplitKFactor = 8;

void upload_m(MData* md, const ShapeData& sd, int max_output_tiles) {
  CUDA_CHECK(cudaMalloc(&md->d_act, md->act.size()));
  CUDA_CHECK(cudaMemcpy(md->d_act, md->act.data(), md->act.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&md->d_a_scales, md->a_scales.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(md->d_a_scales, md->a_scales.data(),
                        md->a_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&md->d_output,
                        static_cast<size_t>(md->M) * sd.N * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&md->d_splitk_ws,
                        static_cast<size_t>(kMaxSplitKFactor) *
                            static_cast<size_t>(md->M) * sd.N * sizeof(float)));
  CUDA_CHECK(cudaMemset(md->d_splitk_ws, 0,
                        static_cast<size_t>(kMaxSplitKFactor) *
                            static_cast<size_t>(md->M) * sd.N * sizeof(float)));
  md->sem_ints = std::max(max_output_tiles, 1);
  CUDA_CHECK(cudaMalloc(&md->d_splitk_sem, static_cast<size_t>(md->sem_ints) * sizeof(int)));
  CUDA_CHECK(cudaMemset(md->d_splitk_sem, 0, static_cast<size_t>(md->sem_ints) * sizeof(int)));
}

// ---------------------------------------------------------------------------
// Config naming
// ---------------------------------------------------------------------------
template <typename Cfg>
std::string describe_config() {
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer),
                "MT=%u BM=%u WG=%u REG=%u STG=%u TILES=%u SUBS=%u PACK=%d PDL=%d EPI=%d "
                "SPLITK=%d SK=%u PREG=%u SPW=%d PRE=%d SIS=%d",
                Cfg::kMTile, Cfg::kOutputRows, Cfg::kNumMathWgs, Cfg::kMathRegisters,
                Cfg::kStagesPerMathWg, Cfg::kOutputTilesPerCta, Cfg::kSubsPerStage,
                static_cast<int>(Cfg::kPrepack), static_cast<int>(Cfg::kPdl),
                static_cast<int>(Cfg::kEpiOverlap), static_cast<int>(Cfg::kSplitKCta),
                Cfg::kSplitKFactor,
                Cfg::kProducerRegs, static_cast<int>(Cfg::kSingleProducerWarp),
                static_cast<int>(Cfg::kPreloadActivation), static_cast<int>(Cfg::kScaleInSmem));
  return buffer;
}

struct CaseStats {
  std::string label;
  std::string config;
  int N, K, M;
  uint32_t grid_x, grid_y, block, smem;
  double rel_l2 = 0.0;
  double max_abs = 0.0;
  int nonfinite = 0;
  int checked = 0;
  bool pass = false;
  std::string note;
};

std::vector<CaseStats> g_results;
int g_failures = 0;

// ---------------------------------------------------------------------------
// One case: prepack if needed, launch, compare against the host reference.
// ---------------------------------------------------------------------------
template <typename Cfg>
void run_case(const char* label, const char* note, ShapeData& sd, MData& md) {
  using decode_gemm::Problem;
  CaseStats stats;
  stats.label = label;
  stats.config = describe_config<Cfg>();
  stats.N = sd.N;
  stats.K = sd.K;
  stats.M = md.M;

  Problem problem;
  problem.M = md.M;
  problem.N = sd.N;
  problem.K = sd.K;
  problem.activation = md.d_act;
  problem.weight_scales = sd.d_w_scales;
  problem.activation_scales = md.d_a_scales;
  problem.output = md.d_output;
  problem.splitk_ws = md.d_splitk_ws;
  problem.splitk_sem = md.d_splitk_sem;

  const int bm = static_cast<int>(Cfg::kOutputRows);
  if constexpr (Cfg::kPrepack) {
    auto it = sd.d_weight_packed.find(bm);
    if (it == sd.d_weight_packed.end()) {
      const size_t bytes = decode_gemm::prepack::packed_weight_bytes(sd.N, sd.K, bm);
      __nv_fp8_e4m3* packed = nullptr;
      CUDA_CHECK(cudaMalloc(&packed, bytes));
      const cudaError_t prepack_error =
          decode_gemm::prepack::launch_prepack_weight<bm, Cfg::kOutputTilesPerCta>(
              sd.d_weight_raw, packed, sd.N, sd.K);
      CUDA_CHECK(prepack_error);
      CUDA_CHECK(cudaDeviceSynchronize());
      it = sd.d_weight_packed.emplace(bm, packed).first;
    }
    problem.weight = it->second;
  } else {
    problem.weight = sd.d_weight_raw;
  }

  const decode_gemm::LaunchPlan plan = decode_gemm::make_launch_plan<Cfg>(problem);
  stats.grid_x = plan.grid.x;
  stats.grid_y = plan.grid.y;
  stats.block = plan.block.x;
  stats.smem = plan.smem_bytes;
  if (plan.error != nullptr) {
    stats.note = std::string("unsupported: ") + plan.error;
    stats.pass = false;
    ++g_failures;
    g_results.push_back(stats);
    return;
  }

  CUDA_CHECK(cudaMemset(md.d_output, 0,
                        static_cast<size_t>(md.M) * sd.N * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemset(md.d_splitk_sem, 0,
                        static_cast<size_t>(md.sem_ints) * sizeof(int)));
  const cudaError_t launch_error = decode_gemm::launch<Cfg>(problem, nullptr);
  if (launch_error != cudaSuccess) {
    stats.note = std::string("launch: ") + cudaGetErrorString(launch_error);
    ++g_failures;
    g_results.push_back(stats);
    return;
  }
  const cudaError_t sync_error = cudaDeviceSynchronize();
  if (sync_error != cudaSuccess) {
    stats.note = std::string("exec: ") + cudaGetErrorString(sync_error);
    ++g_failures;
    g_results.push_back(stats);
    return;
  }

  // The split-K semaphore must have self-reset, otherwise the next launch (or
  // the next back-to-back iteration in the benchmark) would silently corrupt.
  if constexpr (Cfg::kSplitKCta) {
    std::vector<int> sem(static_cast<size_t>(md.sem_ints), -1);
    CUDA_CHECK(cudaMemcpy(sem.data(), md.d_splitk_sem, sem.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    int dirty = 0;
    for (int v : sem) dirty += (v != 0);
    if (dirty != 0) {
      stats.note = "semaphore not self-reset (" + std::to_string(dirty) + " dirty)";
      ++g_failures;
      g_results.push_back(stats);
      return;
    }
  }

  std::vector<__nv_bfloat16> actual(static_cast<size_t>(md.M) * sd.N);
  CUDA_CHECK(cudaMemcpy(actual.data(), md.d_output, actual.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));

  double diff_energy = 0.0;
  double ref_energy = 0.0;
  stats.checked = static_cast<int>(actual.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    const float got = __bfloat162float(actual[i]);
    const float want = md.reference[i];
    if (!std::isfinite(got) || !std::isfinite(want)) ++stats.nonfinite;
    const double diff = static_cast<double>(got) - want;
    stats.max_abs = std::max(stats.max_abs, std::fabs(diff));
    diff_energy += diff * diff;
    ref_energy += static_cast<double>(want) * want;
  }
  stats.rel_l2 = std::sqrt(diff_energy / std::max(ref_energy, 1.0e-30));
  stats.pass = stats.nonfinite == 0 && stats.rel_l2 <= kRelL2Threshold &&
               stats.checked == static_cast<int>(actual.size());
  if (!stats.pass) ++g_failures;
  if (note != nullptr) stats.note = note;
  g_results.push_back(stats);

  // Run the same launch a second time: catches handshake/semaphore state that
  // only breaks when the buffers are not freshly zeroed.
  const cudaError_t again = decode_gemm::launch<Cfg>(problem, nullptr);
  if (again == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess) {
    CUDA_CHECK(cudaMemcpy(actual.data(), md.d_output,
                          actual.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    double d2 = 0.0, r2 = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
      const double diff = static_cast<double>(__bfloat162float(actual[i])) - md.reference[i];
      d2 += diff * diff;
      r2 += static_cast<double>(md.reference[i]) * md.reference[i];
    }
    const double rel2 = std::sqrt(d2 / std::max(r2, 1.0e-30));
    if (!(rel2 <= kRelL2Threshold)) {
      std::fprintf(stderr, "second launch of %s degraded: rel_l2=%.3e\n", label, rel2);
      ++g_failures;
      g_results.back().pass = false;
      g_results.back().note += " SECOND-LAUNCH-FAIL";
    }
  } else {
    std::fprintf(stderr, "second launch of %s failed\n", label);
    ++g_failures;
    g_results.back().pass = false;
    g_results.back().note += " SECOND-LAUNCH-ERROR";
  }
}

// Shorthand for a KernelConfig instantiation.
template <int MT, int BM, int WG, int REG, int STG, int TILES, int SUBS, bool PACK, bool PDL,
          bool EPI, bool SPLITK, int PREG = 32, bool SPW = false, bool PRE = false,
          bool SIS = (MT <= 16), int SKF = 0>
using Cfg = decode_gemm::KernelConfig<MT, BM, WG, REG, STG, TILES, SUBS, PACK, PDL, EPI, SPLITK,
                                      PREG, SPW, PRE, SIS, SKF>;

// ---- CONTRACT §3 canonical ladder, M_TILE=8 (BM=48 for N%48==0 shapes) -----
using S0_m8 = Cfg<8, 64, 4, 112, 2, 1, 1, false, false, false, false>;
using S1_m8 = Cfg<8, 64, 4, 112, 2, 1, 1, false, true, false, false>;
using S2_m8 = Cfg<8, 64, 4, 112, 2, 1, 2, false, true, false, false>;
using S3_m8 = Cfg<8, 64, 4, 112, 2, 1, 2, true, true, false, false>;
using S4_m8 = Cfg<8, 48, 2, 80, 3, 1, 2, true, true, false, false>;
using S5_m8 = Cfg<8, 48, 2, 80, 3, 1, 2, true, true, true, false>;
using S6_m8 = Cfg<8, 48, 2, 80, 3, 1, 2, true, true, true, true>;
// ---- same ladder with the BM that divides N=7168 exactly (56 -> 128 CTAs) ---
using S4_m8_bm56 = Cfg<8, 56, 2, 80, 3, 1, 2, true, true, false, false>;
using S5_m8_bm56 = Cfg<8, 56, 2, 80, 3, 1, 2, true, true, true, false>;
using S6_m8_bm56 = Cfg<8, 56, 2, 80, 3, 1, 2, true, true, true, true>;
// ---- N=7168 with BM=48 on purpose: 150 tiles, padded_n=7200 > N -> padding --
using S5_m8_pad = Cfg<8, 48, 2, 80, 3, 1, 2, true, true, true, false>;
using S0_m8_pad = Cfg<8, 48, 4, 112, 2, 1, 1, false, false, false, false>;
// ---- small N (2048) shapes: BM=32 -> 64 tiles, SPLIT_K_CTA doubles the CTAs --
using S6_m8_n2048 = Cfg<8, 32, 2, 80, 3, 1, 2, true, true, true, true>;
using S2_m8_n2048_splitk = Cfg<8, 64, 4, 112, 2, 1, 2, false, true, false, true>;
using S5_m8_n2048 = Cfg<8, 32, 2, 80, 3, 1, 2, true, true, true, false>;
// ---- S6 generalised: S-way CTA split-K (SPLITK_FACTOR = 2/4/8) --------------
// qwen36 2048x4096 (KB=32, SUBS=2 -> 16 chunks), BM=32 -> 64 output tiles,
// WGsPerTile=2: S=2/4/8 -> 8/4/2 chunks per CTA, all multiples of 2.  Note that
// S=4/8 leave the per-CTA pipeline shallower than STG=3; that is legal (unused
// stages are never waited on) and is exactly what the measurement has to price.
using S6F2_m8_qwen = Cfg<8, 32, 2, 80, 3, 1, 2, true, true, true, true, 32, false, false, true, 2>;
using S6F4_m8_qwen = Cfg<8, 32, 2, 80, 3, 1, 2, true, true, true, true, 32, false, false, true, 4>;
using S6F8_m8_qwen = Cfg<8, 32, 2, 80, 3, 1, 2, true, true, true, true, 32, false, false, true, 8>;
// kimi_k3 7168x12288 (KB=96, SUBS=2 -> 48 chunks), BM=56 -> 128 output tiles
// (7168 = 56*128, zero pad), WGsPerTile=2: S=2/4/8 -> 24/12/6 chunks per CTA.
using S6F2_m8_kimi = Cfg<8, 56, 2, 80, 3, 1, 2, true, true, true, true, 32, false, false, true, 2>;
using S6F4_m8_kimi = Cfg<8, 56, 2, 80, 3, 1, 2, true, true, true, true, 32, false, false, true, 4>;
using S6F8_m8_kimi = Cfg<8, 56, 2, 80, 3, 1, 2, true, true, true, true, 32, false, false, true, 8>;
// The CONTRACT's divisibility rule, pinned at compile time:
//   K % (128 * SUBS * WGsPerTile * S) == 0
static_assert(S6F2_m8_qwen::template k_ok_for_splitk<4096>(), "qwen S=2");
static_assert(S6F4_m8_qwen::template k_ok_for_splitk<4096>(), "qwen S=4");
static_assert(S6F8_m8_qwen::template k_ok_for_splitk<4096>(), "qwen S=8");
static_assert(S6F2_m8_kimi::template k_ok_for_splitk<12288>(), "kimi S=2");
static_assert(S6F4_m8_kimi::template k_ok_for_splitk<12288>(), "kimi S=4");
static_assert(S6F8_m8_kimi::template k_ok_for_splitk<12288>(), "kimi S=8");
static_assert(S6F8_m8_qwen::kSplitKFactor == 8 && S6F8_m8_qwen::kSplitKCta, "S=8 on");
static_assert(S6_m8_n2048::kSplitKFactor == 2, "legacy SPLIT_K_CTA bool still means S=2");

// ---- optional switches ----------------------------------------------------
using S5_m8_nosmem_scale = Cfg<8, 48, 2, 80, 3, 1, 2, true, true, true, false, 32, false, false,
                               false>;
using S5_m8_preload = Cfg<8, 48, 2, 80, 3, 1, 2, true, true, true, false, 32, false, true>;
using S5_m8_spw = Cfg<8, 48, 2, 80, 3, 1, 2, true, true, true, false, 32, true>;
using S5_m8_tiles2 = Cfg<8, 48, 4, 80, 2, 2, 2, true, true, true, false>;
using S5_m8_wg1 = Cfg<8, 64, 1, 112, 4, 1, 4, true, true, true, false>;
// ---- M_TILE >= 16 (TILES_PER_CTA == NUM_MATH_WG per CONTRACT §3) ------------
using S5_m16 = Cfg<16, 64, 2, 112, 2, 2, 2, true, true, true, false>;
using S5_m32 = Cfg<32, 64, 2, 168, 2, 2, 2, true, true, true, false, 32, false, false, false>;
using S5_m64 = Cfg<64, 48, 2, 168, 2, 2, 2, true, true, true, false, 32, false, false, false>;
using S5_m128 = Cfg<128, 48, 2, 168, 2, 2, 1, true, true, true, false, 32, false, false, false>;
using S2_m16_raw = Cfg<16, 64, 2, 112, 2, 2, 2, false, true, false, false>;
using S6_m32 = Cfg<32, 48, 2, 168, 2, 2, 2, true, true, true, true, 32, false, false, false>;
using S1_m128_raw = Cfg<128, 64, 1, 232, 2, 1, 1, false, true, false, false, 32, false, false,
                        false>;

struct ShapeCases {
  const char* name;
  int N;
  int K;
  uint64_t seed;
};

}  // namespace

int main(int argc, char** argv) {
  init_fp8_lut();
  const bool quick = (argc > 1 && std::strcmp(argv[1], "--quick") == 0);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("# decode_gemm smoke test\n");
  std::printf("# device: %s  sm_%d%d  SMs=%d  clock=%d MHz\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount, prop.clockRate / 1000);
  std::printf("# gate: rel_l2 <= %.1e vs naive fp32 host reference (bf16-rounded)\n\n",
              kRelL2Threshold);
  std::printf("| case | shape (N x K) | M | grid | block | smem B | rel_l2 | max_abs | "
              "elements | verdict | config / note |\n");
  std::printf("|---|---|---|---|---|---|---|---|---|---|---|\n");
  std::fflush(stdout);

  const ShapeCases shapes[] = {
      {"A_contract", 6144, 7168, 0x5eed1234ull},
      {"B_big", 7168, 16384, 0xabcdef01ull},
      {"C_smalln", 2048, 4096, 0x0f0f5678ull},
      {"D_kimi3", 7168, 12288, 0x1234abcdull},
  };

  for (const ShapeCases& sc : shapes) {
    ShapeData sd;
    sd.N = sc.N;
    sd.K = sc.K;
    sd.num_scale_rows = (sc.N + 127) / 128;
    generate_weight(&sd, sc.seed);

    const int max_tiles = (sc.N + 15) / 16;   // >= ceil(N/BM) for every legal BM
    std::map<int, MData> mdata;
    auto get_m = [&](int m) -> MData& {
      auto it = mdata.find(m);
      if (it != mdata.end()) return it->second;
      MData md;
      md.M = m;
      generate_activation(&md, sd, sc.seed ^ (0x9e3779b97f4a7c15ull * static_cast<uint64_t>(m)));
      compute_reference(md, sd);
      upload_m(&md, sd, max_tiles);
      return mdata.emplace(m, std::move(md)).first->second;
    };

    auto report = []() {
      const CaseStats& s = g_results.back();
      const std::string note = s.note.empty() ? "" : (" — " + s.note);
      char line[1280];
      std::snprintf(line, sizeof(line),
                    "| %s | %d x %d | %d | %u x %u | %u | %u | %.3e | %.2e | %d | %s | `%s`%s |\n",
                    s.label.c_str(), s.N, s.K, s.M, s.grid_x, s.grid_y, s.block, s.smem, s.rel_l2,
                    s.max_abs, s.checked, s.pass ? "PASS" : "**FAIL**", s.config.c_str(),
                    note.c_str());
      std::printf("%s", line);
      std::fflush(stdout);
    };

    if (std::strcmp(sc.name, "A_contract") == 0) {
      // CONTRACT §4.9: the whole S0..S6 ladder at M=1 and M=8.
      MData& m1 = get_m(1);
      run_case<S0_m8>("S0 baseline", "raw, no PDL", sd, m1); report();
      run_case<S1_m8>("S1 pdl", "producer trigger", sd, m1); report();
      run_case<S2_m8>("S2 blockk", "SUBS=2, raw", sd, m1); report();
      run_case<S3_m8>("S3 prepack", "SUBS=2, packed", sd, m1); report();
      run_case<S4_m8>("S4 tile_stage", "BM48/WG2/STG3", sd, m1); report();
      run_case<S5_m8>("S5 epi_overlap", "second trigger", sd, m1); report();
      run_case<S6_m8>("S6 splitk", "gridDim.y=2", sd, m1); report();
      run_case<S5_m8_nosmem_scale>("S5 scale_in_reg", "SCALE_IN_SMEM=false", sd, m1); report();
      run_case<S5_m8_preload>("S5 preload_act", "PRELOAD_ACT=true", sd, m1); report();
      run_case<S5_m8_spw>("S5 single_prod_warp", "SINGLE_PRODUCER_WARP", sd, m1); report();
      run_case<S5_m8_tiles2>("S5 tiles2", "TILES=2/WG=4", sd, m1); report();
      run_case<S5_m8_wg1>("S5 wg1_sub4", "WG=1, SUBS=4, STG=4", sd, m1); report();
      MData& m8 = get_m(8);
      run_case<S0_m8>("S0 baseline", "M=8", sd, m8); report();
      run_case<S1_m8>("S1 pdl", "M=8", sd, m8); report();
      run_case<S2_m8>("S2 blockk", "M=8", sd, m8); report();
      run_case<S3_m8>("S3 prepack", "M=8", sd, m8); report();
      run_case<S4_m8>("S4 tile_stage", "M=8", sd, m8); report();
      run_case<S5_m8>("S5 epi_overlap", "M=8", sd, m8); report();
      run_case<S6_m8>("S6 splitk", "M=8", sd, m8); report();
      run_case<S5_m8_nosmem_scale>("S5 scale_in_reg", "M=8", sd, m8); report();
      if (!quick) {
        MData& m16 = get_m(16);
        run_case<S5_m16>("M16 S5", "M_TILE=16", sd, m16); report();
        run_case<S2_m16_raw>("M16 S2 raw", "M_TILE=16, raw layout", sd, m16); report();
        MData& m32 = get_m(32);
        run_case<S5_m32>("M32 S5", "M_TILE=32", sd, m32); report();
        MData& m64 = get_m(64);
        run_case<S5_m64>("M64 S5", "M_TILE=64", sd, m64); report();
        MData& m128 = get_m(128);
        run_case<S5_m128>("M128 S5", "M_TILE=128", sd, m128); report();
        run_case<S6_m32>("M32 S6 splitk", "M_TILE=32 split-K", sd, m32); report();
      }
    } else if (std::strcmp(sc.name, "B_big") == 0) {
      MData& m1 = get_m(1);
      MData& m8 = get_m(8);
      run_case<S0_m8_pad>("S0 baseline bm48 pad", "raw, 150 tiles, padded_n=7200", sd, m1); report();
      run_case<S4_m8_bm56>("S4 tile_stage bm56", "7168/56 = 128 CTAs, no pad", sd, m1); report();
      run_case<S5_m8_bm56>("S5 epi_overlap bm56", nullptr, sd, m8); report();
      run_case<S5_m8_pad>("S5 epi_overlap bm48 pad", "padding predicate", sd, m8); report();
      run_case<S6_m8_bm56>("S6 splitk bm56", nullptr, sd, m1); report();
      if (!quick) {
        MData& m32 = get_m(32);
        run_case<S5_m32>("M32 S5 bm64", "M_TILE=32, big K", sd, m32); report();
        MData& m128 = get_m(128);
        run_case<S1_m128_raw>("M128 S1 raw wg1", "raw layout, M_TILE=128, WG=1", sd, m128); report();
      }
    } else if (std::strcmp(sc.name, "C_smalln") == 0) {
      MData& m1 = get_m(1);
      MData& m8 = get_m(8);
      run_case<S5_m8_n2048>("S5 epi_overlap bm32", "small N: 64 tiles", sd, m1); report();
      run_case<S6_m8_n2048>("S6 splitk bm32", "small N: 64x2 = 128 CTAs", sd, m1); report();
      run_case<S6_m8_n2048>("S6 splitk bm32", "small N, M=8", sd, m8); report();
      run_case<S2_m8_n2048_splitk>("S2+splitk raw bm64", "raw layout + split-K", sd, m1); report();
      // ---- S-way split-K gate (small N: 64 tiles -> 128/256/512 CTAs) ----
      run_case<S6F2_m8_qwen>("S6 splitk S=2 bm32", "64x2 = 128 CTAs", sd, m1); report();
      run_case<S6F4_m8_qwen>("S6 splitk S=4 bm32", "64x4 = 256 CTAs, 4 chunks/CTA", sd, m1); report();
      run_case<S6F8_m8_qwen>("S6 splitk S=8 bm32", "64x8 = 512 CTAs, 2 chunks/CTA", sd, m1); report();
      run_case<S6F2_m8_qwen>("S6 splitk S=2 bm32", "M=8", sd, m8); report();
      run_case<S6F4_m8_qwen>("S6 splitk S=4 bm32", "M=8", sd, m8); report();
      run_case<S6F8_m8_qwen>("S6 splitk S=8 bm32", "M=8", sd, m8); report();
    } else if (std::strcmp(sc.name, "D_kimi3") == 0) {
      // kimi_k3 o_proj (7168x12288): N already gives 128 CTAs at BM=56, so
      // split-K has nothing to win here -- it is the negative control.
      MData& m1 = get_m(1);
      MData& m8 = get_m(8);
      run_case<S4_m8_bm56>("S4 tile_stage bm56", "128 CTAs, no pad", sd, m1); report();
      run_case<S6F2_m8_kimi>("S6 splitk S=2 bm56", "128x2 = 256 CTAs", sd, m1); report();
      run_case<S6F4_m8_kimi>("S6 splitk S=4 bm56", "128x4 = 512 CTAs", sd, m1); report();
      run_case<S6F8_m8_kimi>("S6 splitk S=8 bm56", "128x8 = 1024 CTAs", sd, m1); report();
      run_case<S6F2_m8_kimi>("S6 splitk S=2 bm56", "M=8", sd, m8); report();
      run_case<S6F4_m8_kimi>("S6 splitk S=4 bm56", "M=8", sd, m8); report();
      run_case<S6F8_m8_kimi>("S6 splitk S=8 bm56", "M=8", sd, m8); report();
    }

    for (auto& kv : sd.d_weight_packed) CUDA_CHECK(cudaFree(kv.second));
    for (auto& kv : mdata) {
      CUDA_CHECK(cudaFree(kv.second.d_act));
      CUDA_CHECK(cudaFree(kv.second.d_a_scales));
      CUDA_CHECK(cudaFree(kv.second.d_output));
      CUDA_CHECK(cudaFree(kv.second.d_splitk_ws));
      CUDA_CHECK(cudaFree(kv.second.d_splitk_sem));
    }
    CUDA_CHECK(cudaFree(sd.d_weight_raw));
    CUDA_CHECK(cudaFree(sd.d_w_scales));
  }

  std::printf("\n# total cases: %d, failures: %d -> %s\n",
              static_cast<int>(g_results.size()), g_failures,
              g_failures == 0 ? "ALL PASS" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
