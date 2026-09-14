// SPDX-License-Identifier: MIT
//
// bench/bench_decode.cu —— decode_gemm 主 benchmark harness（CONTRACT §6）。
//
//   bench_decode --models all|kimi_k3,glm52 --steps 0-6|all --ms 1,2,4,8,16,32,64,128
//                [--tune] [--reps N] [--out results/]
//
// 本 TU *不* include kernel 头：所有 kernel 实例化都在 bench_m8.cu ... bench_m128.cu
// 里，通过 bench::VariantOps 注册表调用（CONTRACT §6「编译拆分」）。
//
// 口径（CONTRACT §6 + orchestrator 2026-09-13 两次补充）：
//   * correctness：dequantize 到 fp32 后 cublasLt SGEMM 参考，rel_l2 <= 2e-3，
//     每 (model,M) 只算一次（参考与 step 无关）
//   * ISO：单发 eventSync，每次 256MB L2 flush + 旋转冷权重集，p50/p30
//   * B2B：同 stream back-to-back（默认 90 连发，K>=16384 降 50），旋转权重集
//   * PDL：B2B + cudaLaunchAttributeProgrammaticStreamSerialization
//     （harness 自己 cudaLaunchKernelEx，attr 只在这个协议打开 -> B2B/PDL 只差一个 attr）
//   * 文章曲线口径：step>=1 用 PDL，step0 用 B2B
//   * bandwidth_gbps = (N*K + M*K + 2*M*N) bytes / latency
//   * pct_of_spec_peak = bandwidth_gbps / 4000（CONTRACT §6 的公式，是比值不是百分数）
//   * 冷权重：sets = max(5, ceil(512MB / weight_bytes))，每 set 独立 prepack
//   * CSV 16 列，列名/顺序严格按 CONTRACT §6；config 内部用 ';' 分隔；
//     protocol 大写 ISO/B2B/PDL；latency >=4 位小数、bandwidth >=3 位小数；
//     clocks_sm_mhz 每行记实测值
//   * GPU：只用 tools/gpurun_dg.sh 分到的卡（池 {0,1}，跨项目 flock + 争用守卫 +
//     锁频 1830）。GPU2/3 被 weave_v1 keeper 预留，禁止使用（CONTRACT §0 修订版）。
//     实际用的物理卡号写进 CSV 的 clocks 行来源与 summary JSON。

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <sys/file.h>
#include <map>
#include <set>
#include <sys/stat.h>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "bench_common.cuh"
#include "configs.cuh"
#include "json_min.h"

// ------------------------------------------------------------------ 常量 ----
// 全部 helper 放在 namespace bench 的内部链接块里：main() 里统一用 bench:: 前缀调用
namespace bench {
namespace {

constexpr double kSpecPeakGbps = 4000.0;               // CONTRACT §7 主参考线
constexpr size_t kWorkingSetBytes = 512ull * 1024 * 1024;
constexpr size_t kMinWeightSets = 5;                   // orchestrator 补充（原 13）
constexpr double kRelL2Threshold = 2e-3;               // CONTRACT §6
constexpr int kWarmupLaunches = 10;
constexpr int kBigKThreshold = 16384;                  // ISO/B2B 降档阈值
constexpr size_t kLtWorkspaceBytes = 64ull * 1024 * 1024;

#define CUDA_CHECK(expr)                                                      \
  do {                                                                       \
    const cudaError_t st_ = (expr);                                          \
    if (st_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "[bench] CUDA error %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(st_));                       \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

#define LT_CHECK(expr)                                                       \
  do {                                                                       \
    const cublasStatus_t st_ = (expr);                                       \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "[bench] cublasLt error %s:%d: %d\n", __FILE__,    \
                    __LINE__, static_cast<int>(st_));                         \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

// ------------------------------------------------------------- CLI 选项 -----
struct Options {
  std::vector<std::string> models;      // 空 = all
  std::vector<int> steps;               // 空 = all
  std::vector<int> ms;                  // 空 = 1,2,4,8,16,32,64,128
  bool tune = false;
  bool tune_only = false;
  int reps = 3;
  int iso_samples = 0;                  // 0 = auto（30；K>=16384 -> 20）
  int b2b_batch = 0;                    // 0 = auto（90；K>=16384 -> 50）
  std::string out_dir;
  std::string models_json;
  uint64_t seed = 20260913ull;
  bool demo = false;
  bool no_correctness = false;
  bool check_launcher = false;
  bool list_only = false;
  bool verbose = false;
  // 冷权重 set 组的显存预算（LRU 淘汰）。canonical 主 sweep 只需 raw+packed 两组
  // ≈1.2GiB；默认压到 2.5GiB 保证单卡总占用 <3GiB（orchestrator 预算）。
  // --tune 全网格会长到 ~10 组，需要时显式加大 --weight-cache-gb。
  double weight_cache_gb = 2.5;
  bool force_api_launch = false;
  bool allow_shared = false;
  int l2_flush_mb = 256;
  std::vector<std::string> protocols;      // 空 = ISO,B2B,PDL 全跑
  bool splitk_ablation = false;            // S5 的 split-K on/off 消融
  bool pdl_placement = false;              // S4 几何的触发点放置消融（producer/both/store）
  std::string ablation_out;                // 默认 <out>/ablation_splitk.csv
  // S 路 split-K 扫描（splitk-S agent 2026-09-13）：S6 上把 S 当自变量扫
  bool splitk_factor_sweep = false;
  std::vector<int> splitk_factors;         // 空 = 1,2,4,8
  std::string factor_out;                  // 默认 <out>/splitk_factor_sweep.csv
  // split-K semaphore 的轮转槽数（按冷权重 set 索引取模）。1 = 关（旧行为）。
  // 为什么需要：PDL 协议下相邻两次 launch 真的会重叠，共用一份 sem 时后一次
  // launch 的 atomicAdd 可能落在前一次 reducer 的 atomicExch(0) 之前而被抹掉，
  // 于是该 tile 选不出 reducer（少做一次 reduce -> 计时偏乐观）且 sem 停在非 0，
  // 之后所有 launch 的握手全乱（实测 BM56/S=4 的 PDL 批次跑完后，BM32/S=2 的
  // rel_l2 从 1.65e-3 变成 1.71e-1）。轮转后重叠窗口内的 launch 用不同的槽。
  int sem_rotate = 16;
  std::string tune_out;                    // --tune-out PATH（每模型一个文件）
  std::string purge_only;                  // --purge-only PATH：只清理不完整组就退出
  bool resume = false;              // 断点续跑：跳过已存在的 (model,step,M,protocol)
  std::string resume_path;          // 显式指定要续跑的 CSV（空 = out_dir 里最新的一个）
  bool no_resume = false;
};

void print_usage(const char* prog) {
  std::printf(
      "usage: %s [--models all|id1,id2] [--steps all|0-6|0,1,3] "
      "[--ms 1,2,4,8,16,32,64,128]\n"
      "          [--tune] [--tune-only] [--reps N] [--iso-samples N] "
      "[--b2b-batch N]\n"
      "          [--out DIR] [--models-json PATH] [--seed N] [--demo] "
      "[--verbose]\n"
      "          [--no-correctness] [--check-launcher] [--list]\n"
      "          [--weight-cache-gb X] [--force-api-launch] [--allow-shared]\n"
      "          [--l2-flush-mb N]\n"
      "\n"
      "  --models      models.json 里的 id，或 all（默认 all）\n"
      "  --steps       0-6 / all / 逗号列表（默认 all）\n"
      "  --ms          M 阶梯，默认 1,2,4,8,16,32,64,128（M<=8 -> M_TILE=8）\n"
      "  --tune        canonical 之外再扫 CONTRACT §6 的 config 网格 -> tune_*.csv\n"
      "  --tune-only   只扫网格，不跑 canonical\n"
      "  --reps N      ISO 轮数 / B2B、PDL 批次数（默认 3）\n"
      "  --demo        小 sweep 自测：1 模型 x step0,1 x M=1,8，reps=1\n"
      "  --out DIR     结果目录（默认 <exe>/../results，否则 ./results）\n"
      "  --list        只打印模型/已实例化 variant 表，不碰 GPU\n"
      "  --check-launcher  额外用 decode_gemm::launch<Cfg> 复算 rel_l2 交叉验证\n"
      "  --force-api-launch 计时也走 kernel 自带 launcher（attr 不受 harness 控制，\n"
      "                  B2B/PDL 会变得等价，仅用于排查）\n"
      "  --allow-shared  即使 nvidia-smi 显示别的 compute app 也照跑（默认拒绝）\n"
      "  --protocols ISO,B2B,PDL  只跑指定协议（tune 建议只跑 PDL，省 2/3 时间）\n"
      "  --tune-out PATH   tune 结果写到 PATH（每模型一个文件，存在则 purge+append）\n"
      "  --splitk-ablation S5 的 split-K on/off 都跑，写 --ablation-out（+列 splitk_on）\n"
      "  --pdl-placement S4 几何跑触发点三变体（producer/both/store），写 ablation_pdl_placement.csv\n"
      "  --ablation-out PATH  默认 <out>/ablation_splitk.csv\n"
      "  --splitk-factor-sweep S6 上扫 S 路 split-K 宽度，写 --factor-out\n"
      "                  （+列 splitk_factor）；每个 M 取两条基准几何：\n"
      "                  base0 = 该形状该 M 现役 canonical（可能是 tune override），\n"
      "                  base1 = 形状无关 canonical BM（README §6.2 的 S6 几何）\n"
      "  --splitk-factors 1,2,4,8  要扫的 S（默认 1,2,4,8）\n"
      "  --factor-out PATH    默认 <out>/splitk_factor_sweep.csv\n"
      "  --sem-rotate N       split-K semaphore 的轮转槽数（默认 16；1 = 关，会重现\n"
      "                  PDL 连发时 sem 被跨 launch 污染 -> 数值错 + 计时偏乐观）\n"
      "  --purge-only PATH 不跑 GPU，只把 PATH 里不完整的 (model,step,M[,config]) 组删掉\n"
      "  --resume [PATH] 断点续跑：读 PATH（默认 out_dir 里最新的 bench_*.csv）\n"
      "                  已有的 (model,step,M,protocol) 组合直接跳过，只补缺口；\n"
      "                  同一个 CSV 以 append 方式继续写（被 kill/timeout 后重跑用）\n"
      "  --no-resume     即使 run_sweep.sh 传了 --resume 也强制新开一个 CSV\n"
      "\n"
      "  GPU 必须通过 tools/gpurun_dg.sh 启动（它负责选卡/flock/锁频/"
      "CUDA_VISIBLE_DEVICES）。\n",
      prog);
}

std::vector<int> parse_int_ranges(const std::string& text, int lo, int hi) {
  std::vector<int> out;
  if (text == "all" || text.empty()) {
    for (int v = lo; v <= hi; ++v) out.push_back(v);
    return out;
  }
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    if (item == "all") {
      for (int v = lo; v <= hi; ++v) out.push_back(v);
      continue;
    }
    const size_t dash = item.find('-');
    if (dash == std::string::npos) {
      const int v = std::atoi(item.c_str());
      if (v >= lo && v <= hi) out.push_back(v);
    } else {
      const int a = std::atoi(item.substr(0, dash).c_str());
      const int b = std::atoi(item.substr(dash + 1).c_str());
      for (int v = std::min(a, b); v <= std::max(a, b); ++v)
        if (v >= lo && v <= hi) out.push_back(v);
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<std::string> parse_csv_strings(const std::string& text) {
  std::vector<std::string> out;
  if (text == "all" || text.empty()) return out;      // 空 = all
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty() && item != "all") out.push_back(item);
  return out;
}

Options parse_args(int argc, char** argv) {
  Options o;
  auto need = [&](int& i) -> const char* {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "[bench] %s 缺参数\n", argv[i]);
      std::exit(2);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--models") o.models = parse_csv_strings(need(i));
    else if (a == "--steps") o.steps = parse_int_ranges(need(i), 0, 6);
    else if (a == "--ms") o.ms = parse_int_ranges(need(i), 1, 128);
    else if (a == "--tune") o.tune = true;
    else if (a == "--tune-only") { o.tune = true; o.tune_only = true; }
    else if (a == "--reps") o.reps = std::max(1, std::atoi(need(i)));
    else if (a == "--iso-samples") o.iso_samples = std::max(1, std::atoi(need(i)));
    else if (a == "--b2b-batch") o.b2b_batch = std::max(1, std::atoi(need(i)));
    else if (a == "--out") o.out_dir = need(i);
    else if (a == "--models-json") o.models_json = need(i);
    else if (a == "--seed") o.seed = std::strtoull(need(i), nullptr, 10);
    else if (a == "--demo") o.demo = true;
    else if (a == "--no-correctness") o.no_correctness = true;
    else if (a == "--check-launcher") o.check_launcher = true;
    else if (a == "--list") o.list_only = true;
    else if (a == "--verbose") o.verbose = true;
    else if (a == "--weight-cache-gb") o.weight_cache_gb = std::atof(need(i));
    else if (a == "--force-api-launch") o.force_api_launch = true;
    else if (a == "--allow-shared") o.allow_shared = true;
    else if (a == "--protocols") {
      std::stringstream ss(need(i));
      std::string item;
      while (std::getline(ss, item, ',')) {
        for (char& c : item) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (!item.empty()) o.protocols.push_back(item);
      }
    }
    else if (a == "--splitk-ablation") o.splitk_ablation = true;
    else if (a == "--pdl-placement") o.pdl_placement = true;
    else if (a == "--ablation-out") o.ablation_out = need(i);
    else if (a == "--splitk-factor-sweep") o.splitk_factor_sweep = true;
    else if (a == "--splitk-factors") {
      for (int f : parse_int_ranges(need(i), 1, 8))
        if (f == 1 || f == 2 || f == 4 || f == 8) o.splitk_factors.push_back(f);
    }
    else if (a == "--factor-out") o.factor_out = need(i);
    else if (a == "--sem-rotate") o.sem_rotate = std::max(1, std::atoi(need(i)));
    else if (a == "--tune-out") { o.tune_out = need(i); o.tune = true; }
    else if (a == "--purge-only") { o.purge_only = need(i); }
    else if (a == "--resume") {
      o.resume = true;
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
        o.resume_path = argv[++i];
    }
    else if (a == "--resume-from") { o.resume = true; o.resume_path = need(i); }
    else if (a == "--no-resume") { o.no_resume = true; o.resume = false; }
    else if (a == "--l2-flush-mb") o.l2_flush_mb = std::max(1, std::atoi(need(i)));
    else if (a == "-h" || a == "--help") { print_usage(argv[0]); std::exit(0); }
    else {
      std::fprintf(stderr, "[bench] 未知参数: %s\n", a.c_str());
      print_usage(argv[0]);
      std::exit(2);
    }
  }
  if (o.no_resume) o.resume = false;
  if (o.splitk_factor_sweep && o.splitk_factors.empty())
    o.splitk_factors = {1, 2, 4, 8};
  if (o.steps.empty()) o.steps = parse_int_ranges("all", 0, 6);
  if (o.ms.empty()) o.ms = {1, 2, 4, 8, 16, 32, 64, 128};
  if (o.demo) {
    o.steps = {0, 1};
    o.ms = {1, 8};
    o.reps = 1;
    o.iso_samples = 10;
    o.b2b_batch = 30;
    if (o.models.size() > 1) o.models.resize(1);
  }
  return o;
}

// ---------------------------------------------------------------- 模型表 ----
struct ModelShape {
  std::string id;
  std::string display;
  std::string layer;
  int hidden = 0;
  int N = 0;
  int K = 0;
};

// models.json 缺失时的兜底（= orchestrator 2026-09-13 定稿形状），会打 WARNING
std::vector<ModelShape> builtin_models() {
  return {
      {"kimi_k3", "Kimi-K3", "o_proj", 7168, 7168, 12288},
      {"qwen36", "Qwen3.6-35B-A3B", "o_weight", 2048, 2048, 4096},
      {"glm52", "GLM-5.2", "wo", 6144, 6144, 16384},
      {"deepseek_v4_pro", "DeepSeek-V4-Pro", "wo_b", 7168, 7168, 16384},
      {"minimax_m3", "MiniMax-M3", "w_o", 6144, 6144, 8192},
  };
}

std::vector<ModelShape> load_models(const std::string& path, bool* fallback,
                                    std::string* note) {
  *fallback = false;
  std::string err;
  jsonmin::ValuePtr root = jsonmin::parse_file(path, &err);
  if (!root) {
    *fallback = true;
    *note = "models.json 读取失败(" + err + ") -> 用内置兜底形状";
    return builtin_models();
  }
  const jsonmin::Value* arr = root->find("models");
  if (!arr || arr->size() == 0) {
    *fallback = true;
    *note = "models.json 里没有 models[] -> 用内置兜底形状";
    return builtin_models();
  }
  std::vector<ModelShape> out;
  for (size_t i = 0; i < arr->size(); ++i) {
    const jsonmin::Value* m = arr->at(i);
    if (!m) continue;
    ModelShape s;
    if (m->find("id")) s.id = m->find("id")->as_string();
    if (m->find("display")) s.display = m->find("display")->as_string();
    if (m->find("hidden")) s.hidden = m->find("hidden")->as_int();
    const jsonmin::Value* g = m->find("gemm");
    if (g) {
      if (g->find("layer")) s.layer = g->find("layer")->as_string();
      if (g->find("N")) s.N = g->find("N")->as_int();
      if (g->find("K")) s.K = g->find("K")->as_int();
    }
    if (s.display.empty()) s.display = s.id;
    if (s.id.empty() || s.N <= 0 || s.K <= 0) {
      *note = "models.json 第 " + std::to_string(i) + " 项缺 id/N/K，已跳过";
      continue;
    }
    out.push_back(s);
  }
  if (out.empty()) {
    *fallback = true;
    *note = "models.json 无有效条目 -> 用内置兜底形状";
    return builtin_models();
  }
  *note = "loaded " + std::to_string(out.size()) + " models from " + path;
  return out;
}

// ------------------------------------------------------------ host 数据 -----
constexpr float kFp8Max = 448.0f;
constexpr float kQuantEps = 1.0e-4f;

struct HostWeights {
  std::vector<uint8_t> weight;        // [n_ker, K] fp8 e4m3 bytes
  std::vector<float> scales;          // [scale_rows_alloc, kb]
  int n_log = 0, n_ker = 0, K = 0, kb = 0, scale_rows_alloc = 0;
};

struct HostAct {
  std::vector<uint8_t> act;           // [rows_alloc, K]
  std::vector<float> scales;          // [rows_alloc, kb]
  int rows_alloc = 0, K = 0, kb = 0;
};

inline uint8_t quant_one(float v, float scale) {
  const float q = std::max(-kFp8Max, std::min(kFp8Max, v / scale));
  const __nv_fp8_e4m3 f8(q);
  return *reinterpret_cast<const uint8_t*>(&f8);
}

// per-128(K) amax 量化，权重按 128x128 block 生成（与参考 harness 同一口径）
HostWeights make_weights(int n_log, int K, uint64_t seed) {
  const int n_ker = ((n_log + 127) / 128) * 128;     // CONTRACT §2: harness pad
  const int kb = K / 128;
  const int nb = n_ker / 128;
  const int scale_rows_alloc = nb + 1;               // pad tile 越界保护
  HostWeights h;
  h.n_log = n_log; h.n_ker = n_ker; h.K = K; h.kb = kb;
  h.scale_rows_alloc = scale_rows_alloc;
  h.weight.assign(static_cast<size_t>(n_ker) * K, 0);
  h.scales.assign(static_cast<size_t>(scale_rows_alloc) * kb, 0.0f);

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(K));
  std::vector<float> block(128 * 128);

  for (int nbi = 0; nbi < nb; ++nbi) {
    for (int kbi = 0; kbi < kb; ++kbi) {
      float amax = 0.0f;
      for (int r = 0; r < 128; ++r) {
        const int n = nbi * 128 + r;
        for (int c = 0; c < 128; ++c) {
          const float v = (n < n_log)
                              ? static_cast<float>(normal(rng) * inv_sqrt_k)
                              : 0.0f;
          block[r * 128 + c] = v;
          amax = std::max(amax, std::abs(v));
        }
      }
      const float scale = std::max(amax, kQuantEps) / kFp8Max;
      h.scales[static_cast<size_t>(nbi) * kb + kbi] = scale;
      for (int r = 0; r < 128; ++r) {
        const int n = nbi * 128 + r;
        uint8_t* dst = &h.weight[static_cast<size_t>(n) * K + kbi * 128];
        for (int c = 0; c < 128; ++c)
          dst[c] = quant_one(block[r * 128 + c], scale);
      }
    }
  }
  return h;
}

HostAct make_activation(int rows_alloc, int K, uint64_t seed) {
  const int kb = K / 128;
  HostAct h;
  h.rows_alloc = rows_alloc; h.K = K; h.kb = kb;
  h.act.assign(static_cast<size_t>(rows_alloc) * K, 0);
  h.scales.assign(static_cast<size_t>(rows_alloc) * kb, 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(K));
  std::vector<float> row(K);
  for (int m = 0; m < rows_alloc; ++m) {
    for (int k = 0; k < K; ++k)
      row[k] = static_cast<float>(normal(rng) * inv_sqrt_k);
    for (int kbi = 0; kbi < kb; ++kbi) {
      float amax = 0.0f;
      for (int c = 0; c < 128; ++c)
        amax = std::max(amax, std::abs(row[kbi * 128 + c]));
      const float scale = std::max(amax, kQuantEps) / kFp8Max;
      h.scales[static_cast<size_t>(m) * kb + kbi] = scale;
      uint8_t* dst = &h.act[static_cast<size_t>(m) * K + kbi * 128];
      for (int c = 0; c < 128; ++c) dst[c] = quant_one(row[kbi * 128 + c], scale);
    }
  }
  return h;
}

// --------------------------------------------------------------- kernels ----
__global__ void flush_l2_kernel(float4* buf, size_t n4) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n4) buf[i] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
}

__global__ void dequant_weight_kernel(const __nv_fp8_e4m3* __restrict__ w,
                                      const float* __restrict__ scales,
                                      float* __restrict__ out, int N, int K) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(N) * K;
  if (i >= total) return;
  const int k = static_cast<int>(i % K);
  const int n = static_cast<int>(i / K);
  const int kb = K >> 7;
  out[i] = static_cast<float>(w[i]) *
           scales[static_cast<size_t>(n >> 7) * kb + (k >> 7)];
}

__global__ void dequant_act_kernel(const __nv_fp8_e4m3* __restrict__ a,
                                   const float* __restrict__ scales,
                                   float* __restrict__ out, int M, int K) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(M) * K;
  if (i >= total) return;
  const int k = static_cast<int>(i % K);
  const int m = static_cast<int>(i / K);
  const int kb = K >> 7;
  out[i] = static_cast<float>(a[i]) *
           scales[static_cast<size_t>(m) * kb + (k >> 7)];
}

// ------------------------------------------------------------- cublasLt -----
struct LtContext {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  ~LtContext() {
    if (workspace) cudaFree(workspace);
    if (handle) cublasLtDestroy(handle);
  }
};

// C[M,N] row-major = act[M,K] row-major @ W[N,K]^T
// col-major 等价式：C_cm[N,M] = op_T(W_cm[K,N]) * op_N(act_cm[K,M])
// 注意：w_deq / act_deq / d_c **全部是 device 指针**（cublasLt 是 device GEMM，
// 把 host 指针当 C 传进去会直接 illegal memory access —— 踩过一次）。
bool lt_sgemm_ref(LtContext& lt, const float* w_deq, const float* act_deq,
                  float* d_c, int N, int M, int K, cudaStream_t stream) {
  if (!lt.handle) {
    LT_CHECK(cublasLtCreate(&lt.handle));
    CUDA_CHECK(cudaMalloc(&lt.workspace, kLtWorkspaceBytes));
  }
  cublasLtMatmulDesc_t op = nullptr;
  LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, N));
  cublasLtMatmulPreference_t pref = nullptr;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kLtWorkspaceBytes,
      sizeof(kLtWorkspaceBytes)));
  cublasLtMatmulHeuristicResult_t heur{};
  int found = 0;
  const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
      lt.handle, op, la, lb, lc, lc, pref, 1, &heur, &found);
  cublasLtMatmulPreferenceDestroy(pref);
  if (hs != CUBLAS_STATUS_SUCCESS || found == 0) {
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    return false;
  }
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t ms = cublasLtMatmul(
      lt.handle, op, &alpha, w_deq, la, act_deq, lb, &beta, d_c, lc, d_c, lc,
      &heur.algo, lt.workspace, kLtWorkspaceBytes, stream);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
  return ms == CUBLAS_STATUS_SUCCESS;
}

// --------------------------------------------------------------- 计时 -------
struct TimingPlan {
  int reps = 3;
  int iso_samples = 30;
  int batch = 90;
};

TimingPlan plan_for(int K, const Options& o) {
  TimingPlan p;
  const bool big = K >= kBigKThreshold;   // orchestrator: 大 K 降到 20/50
  p.reps = o.reps;
  p.iso_samples = o.iso_samples > 0 ? o.iso_samples : (big ? 20 : 30);
  p.batch = o.b2b_batch > 0 ? o.b2b_batch : (big ? 50 : 90);
  return p;
}

struct TimingStats {
  double p50_us = std::numeric_limits<double>::quiet_NaN();
  double p30_us = std::numeric_limits<double>::quiet_NaN();
  double min_us = std::numeric_limits<double>::quiet_NaN();
  double mean_us = std::numeric_limits<double>::quiet_NaN();
  int n = 0;
  std::vector<double> rounds;
};

double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return std::numeric_limits<double>::quiet_NaN();
  const double idx = q * static_cast<double>(sorted.size());
  size_t lo = static_cast<size_t>(std::ceil(idx));
  if (lo == 0) lo = 1;
  if (lo > sorted.size()) lo = sorted.size();
  return sorted[lo - 1];
}

TimingStats summarize(std::vector<double> samples) {
  TimingStats s;
  if (samples.empty()) return s;
  double sum = 0.0;
  for (double v : samples) sum += v;
  s.mean_us = sum / static_cast<double>(samples.size());
  s.min_us = *std::min_element(samples.begin(), samples.end());
  std::sort(samples.begin(), samples.end());
  s.p50_us = percentile(samples, 0.50);
  s.p30_us = percentile(samples, 0.30);
  s.n = static_cast<int>(samples.size());
  return s;
}

// ------------------------------------------------------------ shell 探测 ----
std::string run_shell(const std::string& cmd) {
  std::string out;
  FILE* p = ::popen(cmd.c_str(), "r");
  if (!p) return out;
  char buf[4096];
  while (std::fgets(buf, sizeof(buf), p)) out += buf;
  ::pclose(p);
  return out;
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

struct EnvInfo {
  int device_index = 0;
  std::string device_name;
  std::string pci_bus_id;
  int sm_count = 0;
  size_t mem_total = 0;
  int clock_max_khz = 0;
  int clock_cur_mhz = -1;
  int mem_clock_cur_mhz = -1;
  int util_pct = -1;
  int mem_used_mib = -1;
  std::string cuda_visible_devices;
  std::string smi_index;         // 实际用的物理卡号
  std::string gpurun_gpu;        // gpurun_dg.sh 分到的卡（env 传入）
  std::string gpurun_target;
  std::string exclusive_verdict = "unknown";
  std::vector<std::string> other_apps;      // 真正的外来 compute app
  std::vector<std::string> keeper_apps;     // v19_swapab keeper（预留但不算污染）
  std::string pool = "?";                   // BORROW{2,3} / CLEAN{0,1}
  std::string shell_note;
};

// 与 tools/gpurun_dg.sh v2 的 is_keeper() 同口径：cmdline 里有 keeper.py
std::string proc_cmdline(long pid) {
  std::ifstream in("/proc/" + std::to_string(pid) + "/cmdline");
  if (!in) return "";
  std::string all((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  for (char& c : all)
    if (c == '\0') c = ' ';
  return trim(all);
}

// nvidia-smi clocks.sm（gpurun_dg 锁 1830，这里读实测值写进每一行 CSV）
struct ClockProbe {
  std::string smi_index;
  int last_mhz = -1;
  std::chrono::steady_clock::time_point last_t{};
  int read(bool force) {
    const auto now = std::chrono::steady_clock::now();
    if (!force && last_mhz >= 0 &&
        std::chrono::duration<double>(now - last_t).count() < 1.0)
      return last_mhz;
    if (smi_index.empty()) return -1;
    const std::string out = trim(run_shell(
        "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i " +
        smi_index + " 2>/dev/null"));
    last_t = now;
    if (!out.empty() && (std::isdigit(static_cast<unsigned char>(out[0]))))
      last_mhz = std::atoi(out.c_str());
    return last_mhz;
  }
};

EnvInfo collect_env_info(int device_index) {
  EnvInfo e;
  e.device_index = device_index;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
  e.device_name = prop.name;
  e.sm_count = prop.multiProcessorCount;
  e.mem_total = prop.totalGlobalMem;
  char pci[64] = {0};
  if (cudaDeviceGetPCIBusId(pci, sizeof(pci), device_index) == cudaSuccess)
    e.pci_bus_id = pci;
  int clk = 0;
  if (cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, device_index) == cudaSuccess)
    e.clock_max_khz = clk;
  if (const char* v = std::getenv("CUDA_VISIBLE_DEVICES"))
    e.cuda_visible_devices = v;
  if (const char* v = std::getenv("BENCH_GPURUN_GPU")) e.gpurun_gpu = v;
  if (const char* v = std::getenv("BENCH_GPURUN_TARGET")) e.gpurun_target = v;
  if (const char* v = std::getenv("BENCH_EXCLUSIVE_CHECK")) e.shell_note = v;

  // 用 PCI bus id 反查物理 index（CUDA_VISIBLE_DEVICES 会重编号，不能直接信）
  const std::string map = run_shell(
      "nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null");
  std::stringstream ss(map);
  std::string line;
  while (std::getline(ss, line)) {
    const size_t comma = line.find(',');
    if (comma == std::string::npos) continue;
    const std::string idx = trim(line.substr(0, comma));
    const std::string bus = trim(line.substr(comma + 1));
    if (!e.pci_bus_id.empty() &&
        lower(bus).find(lower(e.pci_bus_id)) != std::string::npos) {
      e.smi_index = idx;
      break;
    }
  }
  if (e.smi_index.empty() && !e.gpurun_gpu.empty()) e.smi_index = e.gpurun_gpu;

  if (!e.smi_index.empty()) {
    const std::string q = trim(run_shell(
        "nvidia-smi --query-gpu=clocks.sm,clocks.mem,utilization.gpu,memory.used "
        "--format=csv,noheader,nounits -i " + e.smi_index + " 2>/dev/null"));
    std::vector<std::string> f;
    std::stringstream qs(q);
    std::string item;
    while (std::getline(qs, item, ',')) f.push_back(trim(item));
    if (f.size() >= 4) {
      e.clock_cur_mhz = std::atoi(f[0].c_str());
      e.mem_clock_cur_mhz = std::atoi(f[1].c_str());
      e.util_pct = std::atoi(f[2].c_str());
      e.mem_used_mib = std::atoi(f[3].c_str());
    }
    const std::string apps = run_shell(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
        "--format=csv,noheader -i " + e.smi_index + " 2>/dev/null");
    const long my_pid = static_cast<long>(::getpid());
    std::stringstream as(apps);
    int foreign = 0;
    while (std::getline(as, line)) {
      if (trim(line).empty()) continue;
      const size_t comma = line.find(',');
      const long pid = comma == std::string::npos
                           ? -1
                           : std::atol(trim(line.substr(0, comma)).c_str());
      if (pid == my_pid) continue;
      // keeper 预留进程（GPU2/3）不算外来污染：BORROW 池就是借它们的空闲卡
      if (proc_cmdline(pid).find("keeper.py") != std::string::npos) {
        e.keeper_apps.push_back(trim(line));
        continue;
      }
      ++foreign;
      e.other_apps.push_back(trim(line));
    }
    e.exclusive_verdict = (foreign == 0) ? "pass" : "fail";
    if (foreign == 0 && e.util_pct > 5)
      e.exclusive_verdict = "warn_util_" + std::to_string(e.util_pct);
    e.pool = (e.smi_index == "2" || e.smi_index == "3") ? "BORROW{2,3}"
                                                        : "CLEAN{0,1}";
  } else {
    e.exclusive_verdict = "unknown_no_smi_index";
  }
  return e;
}

// --------------------------------------------------------- 冷权重 set 缓存 --
struct WeightGroup {
  bool prepack = false;
  int bm = 0;
  int tiles = 0;
  size_t set_bytes = 0;
  int sets = 0;
  std::vector<__nv_fp8_e4m3*> bufs;
  uint64_t last_use = 0;
  ~WeightGroup() {
    for (auto* b : bufs) cudaFree(b);
  }
};

class WeightCache {
 public:
  explicit WeightCache(size_t budget_bytes) : budget_(budget_bytes) {}
  ~WeightCache() { clear(); }
  void clear() {
    for (auto* g : groups_) delete g;
    groups_.clear();
    used_ = 0;
  }
  WeightGroup* get(bool prepack, int bm, int tiles, int n_ker, int K,
                   const __nv_fp8_e4m3* raw, const bench::VariantOps* v,
                   cudaStream_t stream, bool verbose);
  size_t used() const { return used_; }
  size_t budget() const { return budget_; }

 private:
  void evict_if_needed(size_t incoming);
  std::vector<WeightGroup*> groups_;
  size_t used_ = 0;
  size_t budget_;
  uint64_t tick_ = 0;
};

void WeightCache::evict_if_needed(size_t incoming) {
  while (used_ + incoming > budget_ && !groups_.empty()) {
    size_t victim = 0;
    for (size_t i = 1; i < groups_.size(); ++i)
      if (groups_[i]->last_use < groups_[victim]->last_use) victim = i;
    WeightGroup* g = groups_[victim];
    used_ -= std::min(used_, g->set_bytes * g->bufs.size());
    delete g;
    groups_.erase(groups_.begin() + victim);
  }
}

WeightGroup* WeightCache::get(bool prepack, int bm, int tiles, int n_ker, int K,
                              const __nv_fp8_e4m3* raw,
                              const bench::VariantOps* v, cudaStream_t stream,
                              bool verbose) {
  for (auto* g : groups_) {
    if (g->prepack == prepack && g->bm == bm && g->tiles == tiles) {
      g->last_use = ++tick_;
      return g;
    }
  }
  const size_t set_bytes =
      v->fn_set_bytes(n_ker, K);
  size_t sets = (kWorkingSetBytes + set_bytes - 1) / set_bytes;   // ceil(512MB/set)
  if (sets < kMinWeightSets) sets = kMinWeightSets;              // orchestrator: max(5,..)
  evict_if_needed(set_bytes * sets);

  auto* g = new WeightGroup();
  g->prepack = prepack; g->bm = bm; g->tiles = tiles;
  g->set_bytes = set_bytes; g->sets = static_cast<int>(sets);
  g->bufs.resize(sets, nullptr);
  for (size_t i = 0; i < sets; ++i) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->bufs[i]), set_bytes));
    if (prepack && v->fn_prepack) {
      const cudaError_t e = v->fn_prepack(raw, g->bufs[i], n_ker, K, stream);
      if (e != cudaSuccess)
        std::fprintf(stderr, "[bench] prepack 失败(bm=%d tiles=%d set=%zu): %s\n",
                     bm, tiles, i, cudaGetErrorString(e));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(g->bufs[i], raw, set_bytes,
                                 cudaMemcpyDeviceToDevice, stream));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  used_ += set_bytes * sets;
  g->last_use = ++tick_;
  groups_.push_back(g);
  if (verbose)
    std::fprintf(stderr,
                 "[bench] weight group prepack=%d bm=%d tiles=%d sets=%zu "
                 "set=%.1fMiB total=%.1fMiB (cache %.2f/%.2f GiB)\n",
                 prepack ? 1 : 0, bm, tiles, sets, set_bytes / 1048576.0,
                 used_ / 1048576.0, used_ / 1073741824.0,
                 budget_ / 1073741824.0);
  return g;
}

// ------------------------------------------------------------- 设备侧结构 ---
struct ModelDevice {
  ModelShape shape;
  int n_log = 0, n_ker = 0, K = 0, kb = 0, scale_rows_alloc = 0;
  HostWeights host_w;
  __nv_fp8_e4m3* d_weight_raw = nullptr;
  float* d_wscale = nullptr;
  float* d_wdeq = nullptr;          // fp32 反量化权重（参考用）
};

struct MDevice {
  int M = 0, m_tile = 0, rows_alloc = 0;
  __nv_fp8_e4m3* d_act = nullptr;
  float* d_ascale = nullptr;
  float* d_adeq = nullptr;
  __nv_bfloat16* d_out = nullptr;
  float* d_ws = nullptr;
  int* d_sem = nullptr;             // [sem_slots][sem_ints] int32
  int sem_ints = 0;                 // 一个槽的 int 数（>= num_output_tiles）
  int sem_slots = 1;                // 轮转槽数（--sem-rotate）
  std::vector<float> ref;           // [M, n_log] host fp32 参考
  bool ref_ok = false;
};

struct RunCtx {
  const bench::VariantOps* v = nullptr;
  bench::DeviceBuffers buf;
  int sem_ints = 0;                 // split-K semaphore 轮转：一个槽的 int 数
  int sem_slots = 1;                // 1 = 不轮转
  WeightGroup* group = nullptr;
  std::vector<CUtensorMap> w_maps;
  CUtensorMap a_map{};
  cudaStream_t stream = nullptr;
  bool use_api = false;
};

struct Globals {
  cudaStream_t stream = nullptr;
  cudaEvent_t ev_start = nullptr, ev_end = nullptr;
  float4* flush_buf = nullptr;
  size_t flush_n4 = 0;
  LtContext lt;
  ClockProbe clocks;
};

cudaError_t do_launch(const RunCtx& ctx, size_t set_idx, bool pdl_attr) {
  bench::DeviceBuffers b = ctx.buf;
  b.weight = ctx.group->bufs[set_idx];
  // split-K semaphore 跟着冷权重 set 一起轮转（见 Options::sem_rotate 的注释）。
  // 相邻 launch 一定落在不同的槽；同一个槽要隔 sem_slots(>=4) 次 launch 才复用，
  // 远超 PDL 的重叠窗口，所以每次 launch 看到的 sem 都是干净的 0。
  if (ctx.sem_slots > 1 && b.splitk_sem != nullptr && ctx.sem_ints > 0)
    b.splitk_sem = ctx.buf.splitk_sem +
                   static_cast<size_t>(set_idx % static_cast<size_t>(ctx.sem_slots)) *
                       ctx.sem_ints;
  bench::LaunchRequest rq;
  rq.buf = b;
  rq.w_map = ctx.w_maps.empty() ? nullptr : &ctx.w_maps[set_idx];
  rq.a_map = &ctx.a_map;
  rq.stream = ctx.stream;
  rq.pdl_attr = pdl_attr;
  rq.use_api = ctx.use_api;
  return ctx.v->fn_launch(rq);
}

void flush_l2(Globals& g) {
  if (!g.flush_buf) return;
  const int threads = 256;
  const size_t blocks = (g.flush_n4 + threads - 1) / threads;
  flush_l2_kernel<<<static_cast<unsigned>(blocks), threads, 0, g.stream>>>(
      g.flush_buf, g.flush_n4);
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
}

TimingStats time_iso(Globals& g, const RunCtx& ctx, const TimingPlan& plan) {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(plan.reps) * plan.iso_samples);
  std::vector<double> round_medians;
  size_t set_idx = 0;
  for (int r = 0; r < plan.reps; ++r) {
    std::vector<double> round;
    round.reserve(plan.iso_samples);
    for (int s = 0; s < plan.iso_samples; ++s) {
      flush_l2(g);                              // 256MB 写，把 L2 冲干净
      CUDA_CHECK(cudaEventRecord(g.ev_start, g.stream));
      CUDA_CHECK(do_launch(ctx, set_idx % ctx.group->sets, false));
      CUDA_CHECK(cudaEventRecord(g.ev_end, g.stream));
      CUDA_CHECK(cudaEventSynchronize(g.ev_end));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, g.ev_start, g.ev_end));
      const double us = ms * 1000.0;
      round.push_back(us);
      samples.push_back(us);
      ++set_idx;
    }
    std::sort(round.begin(), round.end());
    round_medians.push_back(percentile(round, 0.5));
  }
  TimingStats st = summarize(std::move(samples));
  st.rounds = round_medians;
  return st;
}

TimingStats time_batch(Globals& g, const RunCtx& ctx, const TimingPlan& plan,
                       bool pdl_attr) {
  std::vector<double> per_launch;
  per_launch.reserve(plan.reps);
  size_t set_idx = 0;
  for (int r = 0; r < plan.reps; ++r) {
    CUDA_CHECK(cudaStreamSynchronize(g.stream));
    CUDA_CHECK(cudaEventRecord(g.ev_start, g.stream));
    for (int i = 0; i < plan.batch; ++i) {
      CUDA_CHECK(do_launch(ctx, set_idx % ctx.group->sets, pdl_attr));
      ++set_idx;
    }
    CUDA_CHECK(cudaEventRecord(g.ev_end, g.stream));
    CUDA_CHECK(cudaEventSynchronize(g.ev_end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g.ev_start, g.ev_end));
    per_launch.push_back(ms * 1000.0 / plan.batch);
  }
  TimingStats st = summarize(per_launch);
  st.rounds = per_launch;
  return st;
}

// ----------------------------------------------------------- correctness ----
struct CorrResult {
  double rel_l2 = std::numeric_limits<double>::quiet_NaN();
  int checked = 0;
  int nonfinite = 0;
  bool pass = false;
  bool ran = false;
  double rel_l2_launcher = std::numeric_limits<double>::quiet_NaN();
};

double rel_l2_against(const std::vector<__nv_bfloat16>& got,
                      const MDevice& md, const ModelDevice& mdl, double* energy) {
  double de = 0.0;
  for (int m = 0; m < md.M; ++m)
    for (int n = 0; n < mdl.n_log; ++n) {
      const float v =
          __bfloat162float(got[static_cast<size_t>(m) * mdl.n_ker + n]);
      const float r = md.ref[static_cast<size_t>(m) * mdl.n_log + n];
      const double d = static_cast<double>(v) - r;
      de += d * d;
    }
  return std::sqrt(de / std::max(*energy, 1.0e-30));
}

// correctness 一律用**单发**结果：B2B/PDL 连发时 90 个 kernel 写同一个 output
// buffer 属于语义 race（写同值，实践无害，但不能拿来判数值）。计时连发照旧。
CorrResult run_correctness(Globals& g, RunCtx& ctx, const MDevice& md,
                           const ModelDevice& mdl, const Options& o) {
  CorrResult c;
  if (o.no_correctness || !md.ref_ok) return c;
  const size_t out_elems = static_cast<size_t>(md.M) * mdl.n_ker;
  CUDA_CHECK(cudaMemsetAsync(ctx.buf.output, 0,
                             out_elems * sizeof(__nv_bfloat16), g.stream));
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  CUDA_CHECK(do_launch(ctx, 0, false));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(g.stream));

  std::vector<__nv_bfloat16> got(out_elems);
  CUDA_CHECK(cudaMemcpy(got.data(), ctx.buf.output,
                        out_elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  double re = 0.0;
  int nonfinite = 0, checked = 0;
  for (int m = 0; m < md.M; ++m)
    for (int n = 0; n < mdl.n_log; ++n) {
      const float v =
          __bfloat162float(got[static_cast<size_t>(m) * mdl.n_ker + n]);
      const float r = md.ref[static_cast<size_t>(m) * mdl.n_log + n];
      if (!std::isfinite(v) || !std::isfinite(r)) ++nonfinite;
      re += static_cast<double>(r) * r;
      ++checked;
    }
  c.rel_l2 = rel_l2_against(got, md, mdl, &re);
  c.checked = checked;
  c.nonfinite = nonfinite;
  c.pass = (nonfinite == 0) && (c.rel_l2 <= kRelL2Threshold) && checked > 0;
  c.ran = true;

  if (o.check_launcher) {
    // 交叉验证：kernel 自带 decode_gemm::launch<Cfg>（自己 encode tensor map）
    CUDA_CHECK(cudaMemsetAsync(ctx.buf.output, 0,
                               out_elems * sizeof(__nv_bfloat16), g.stream));
    CUDA_CHECK(cudaStreamSynchronize(g.stream));
    bench::DeviceBuffers b = ctx.buf;
    b.weight = ctx.group->bufs[0];
    const cudaError_t e = ctx.v->fn_launch_api(b, g.stream);
    if (e == cudaSuccess) {
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
      std::vector<__nv_bfloat16> got2(out_elems);
      CUDA_CHECK(cudaMemcpy(got2.data(), ctx.buf.output,
                            out_elems * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost));
      c.rel_l2_launcher = rel_l2_against(got2, md, mdl, &re);
    } else {
      std::fprintf(stderr, "[bench] launch<Cfg> 交叉验证失败: %s\n",
                   cudaGetErrorString(e));
    }
  }
  return c;
}

double modeled_bytes(int n_log, int K, int M) {
  return static_cast<double>(n_log) * K + static_cast<double>(M) * K +
         2.0 * M * n_log;                       // CONTRACT §6 分子
}

// ------------------------------------------------------------- CSV 输出 -----
std::string utc_stamp() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

std::string iso_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string exe_dir() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  buf[n] = '\0';
  const std::string p(buf);
  const size_t slash = p.find_last_of('/');
  return slash == std::string::npos ? "." : p.substr(0, slash);
}

bool dir_exists(const std::string& p) {
  struct stat st {};
  return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string default_out_dir() {
  if (const char* env = std::getenv("BENCH_RESULTS_DIR")) {
    if (*env) return env;
  }
  const std::string sibling = exe_dir() + "/../results";
  if (dir_exists(sibling)) return sibling;
  return "results";
}

std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\"\"";
    else out += c;
  }
  out += "\"";
  return out;
}

std::string f4(double v) {   // latency：>=4 位小数
  if (!std::isfinite(v)) return "nan";
  std::ostringstream os; os << std::fixed << std::setprecision(4) << v; return os.str();
}
std::string f3(double v) {   // bandwidth：>=3 位小数
  if (!std::isfinite(v)) return "nan";
  std::ostringstream os; os << std::fixed << std::setprecision(3) << v; return os.str();
}
std::string f6(double v) {
  if (!std::isfinite(v)) return "nan";
  std::ostringstream os; os << std::fixed << std::setprecision(6) << v; return os.str();
}
std::string fe6(double v) {  // rel_l2：科学计数
  if (!std::isfinite(v)) return "nan";
  std::ostringstream os; os << std::scientific << std::setprecision(6) << v; return os.str();
}

// split-K 消融表：主 CSV 的 16 列 + splitk_on（orchestrator 2026-09-13）
const char* const kAblationHeader =
    "model,N,K,M,m_tile,step,step_name,protocol,latency_us_p50,latency_us_p30,"
    "bandwidth_gbps,pct_of_spec_peak,clocks_sm_mhz,rel_l2,pass,config,splitk_on";

// PDL 触发点放置消融表：主 CSV 的 16 列 + pdl_trigger（producer/both/store）
const char* const kPlacementHeader =
    "model,N,K,M,m_tile,step,step_name,protocol,latency_us_p50,latency_us_p30,"
    "bandwidth_gbps,pct_of_spec_peak,clocks_sm_mhz,rel_l2,pass,config,pdl_trigger";

// S 路 split-K 扫描表：主 CSV 的 16 列 + splitk_factor（splitk-S agent）
const char* const kFactorHeader =
    "model,N,K,M,m_tile,step,step_name,protocol,latency_us_p50,latency_us_p30,"
    "bandwidth_gbps,pct_of_spec_peak,clocks_sm_mhz,rel_l2,pass,config,"
    "splitk_factor";

// --tune 的 best config 表（CONTRACT §6「输出 best config 表」）
const char* const kTuneBestHeader =
    "model,N,K,M,m_tile,step,step_name,rank,is_best,rank_protocol,"
    "latency_us_p50,latency_us_p30,bandwidth_gbps,pct_of_spec_peak,"
    "clocks_sm_mhz,rel_l2,pass,config,smem_bytes,num_ctas,weight_sets";

// CONTRACT §6 的 16 列，列名/顺序一字不差
const char* const kCsvHeader =
    "model,N,K,M,m_tile,step,step_name,protocol,latency_us_p50,latency_us_p30,"
    "bandwidth_gbps,pct_of_spec_peak,clocks_sm_mhz,rel_l2,pass,config";
// 追加明细（sidecar / tune 文件用；主 CSV 不写，保证 plot agent 逐列对齐）
const char* const kDetailHeader =
    "model,N,K,M,m_tile,step,step_name,protocol,latency_us_p50,latency_us_p30,"
    "bandwidth_gbps,pct_of_spec_peak,clocks_sm_mhz,rel_l2,pass,config,"
    "latency_us_min,latency_us_mean,samples,reps,iso_samples,b2b_batch,"
    "pdl_batch,variant_kind,tune_index,rank,is_best,padded_n,weight_sets,"
    "weight_set_bytes,smem_bytes,num_ctas,num_threads,launch_mechanism,"
    "rel_l2_launcher";

struct CsvRow {
  std::string model;
  int N = 0, K = 0, M = 0, m_tile = 0, step = -1;
  std::string step_name;
  std::string protocol;                       // ISO / B2B / PDL
  double p50 = std::numeric_limits<double>::quiet_NaN();
  double p30 = std::numeric_limits<double>::quiet_NaN();
  double bw = std::numeric_limits<double>::quiet_NaN();
  double pct = std::numeric_limits<double>::quiet_NaN();
  int clocks_mhz = -1;
  double rel_l2 = std::numeric_limits<double>::quiet_NaN();
  std::string pass = "na";
  std::string config;
  // 明细
  double min_us = std::numeric_limits<double>::quiet_NaN();
  double mean_us = std::numeric_limits<double>::quiet_NaN();
  int samples = 0, reps = 0, iso_samples = 0, b2b_batch = 0, pdl_batch = 0;
  std::string kind = "canonical";
  int tune_index = -1, rank = -1;
  bool is_best = false;
  int padded_n = 0, weight_sets = 0, smem_bytes = 0, num_ctas = 0, num_threads = 0;
  size_t weight_set_bytes = 0;
  std::string mechanism;
  double rel_l2_launcher = std::numeric_limits<double>::quiet_NaN();
};

std::string row_main(const CsvRow& r) {
  std::ostringstream os;
  os << csv_field(r.model) << ',' << r.N << ',' << r.K << ',' << r.M << ','
     << r.m_tile << ',' << r.step << ',' << csv_field(r.step_name) << ','
     << r.protocol << ',' << f4(r.p50) << ',' << f4(r.p30) << ',' << f3(r.bw)
     << ',' << f6(r.pct) << ',' << r.clocks_mhz << ',' << fe6(r.rel_l2) << ','
     << csv_field(r.pass) << ',' << csv_field(r.config);
  return os.str();
}

std::string row_detail(const CsvRow& r) {
  std::ostringstream os;
  os << row_main(r) << ',' << f4(r.min_us) << ',' << f4(r.mean_us) << ','
     << r.samples << ',' << r.reps << ',' << r.iso_samples << ',' << r.b2b_batch
     << ',' << r.pdl_batch << ',' << csv_field(r.kind) << ',' << r.tune_index
     << ',' << r.rank << ',' << (r.is_best ? 1 : 0) << ',' << r.padded_n << ','
     << r.weight_sets << ',' << r.weight_set_bytes << ',' << r.smem_bytes << ','
     << r.num_ctas << ',' << r.num_threads << ',' << csv_field(r.mechanism)
     << ',' << fe6(r.rel_l2_launcher);
  return os.str();
}

class CsvWriter {
 public:
  // append=true：续跑，不写表头，直接往已有文件尾部追加（每行 flush）
  bool open(const std::string& path, const char* header, bool append = false) {
    path_ = path;
    if (append) {
      out_.open(path, std::ios::app);
      if (!out_) return false;
      std::ifstream in(path);
      std::string line;
      while (std::getline(in, line))
        if (!line.empty()) ++rows_;
      --rows_;                                  // 表头不算数据行
      if (rows_ < 0) rows_ = 0;
      return true;
    }
    out_.open(path);
    if (!out_) return false;
    out_ << header << "\n";
    out_.flush();
    return true;
  }
  void write(const std::string& line) {
    if (!out_) return;
    out_ << line << "\n";
    out_.flush();
    ++rows_;
  }
  void close() { if (out_.is_open()) out_.close(); }
  const std::string& path() const { return path_; }
  int rows() const { return rows_; }

 private:
  std::string path_;
  std::ofstream out_;
  int rows_ = 0;
};

// --------------------------------------------------- 断点续跑（resume）-----
// 读已有 CSV，把 (model,N,K,M,m_tile,step,protocol[,config]) 记成 key；
// 只认 latency_us_p50 是有效数字的行（nan/空的行视为没测过，重跑会补上）。
// 注意：容忍 CRLF —— 手工编辑/Windows 工具处理过的 CSV 行尾会带 '\r'，
// 不剥掉的话最后一个字段（config）会多一个 \r，resume 的 key 就永远对不上。
std::string strip_cr(std::string line) {
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
    line.pop_back();
  return line;
}

std::vector<std::string> split_csv_line(const std::string& line_in) {
  const std::string line = strip_cr(line_in);
  std::vector<std::string> out;
  std::string cur;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
        else in_quotes = false;
      } else {
        cur += c;
      }
    } else if (c == '"') {
      in_quotes = true;
    } else if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

struct DoneSet {
  int rows = 0;
  std::string source;

  // key 含 config：换了 config（比如换了二进制/TUNE 设置）就当没测过，重测。
  // canonical 与 tune 用不同前缀，两个文件互不干扰。
  static std::string key(bool is_tune, const CsvRow& r) {
    std::ostringstream os;
    os << (is_tune ? 'T' : 'C') << '|' << r.model << '|' << r.N << '|' << r.K
       << '|' << r.M << '|' << r.m_tile << '|' << r.step << '|' << r.protocol
       << '|' << r.config;
    return os.str();
  }
  bool has(bool is_tune, const CsvRow& r) const {
    return keys_.count(key(is_tune, r)) != 0;
  }
  void add(bool is_tune, const CsvRow& r) { keys_.insert(key(is_tune, r)); }
  size_t canon_count() const { return canon_rows_; }
  size_t tune_count() const { return tune_rows_; }

  bool load(const std::string& path, bool is_tune) {
    std::ifstream in(path);
    if (!in) return false;
    source = path;
    std::string line;
    if (!std::getline(in, line)) return false;
    const std::vector<std::string> hdr = split_csv_line(line);
    std::map<std::string, int> col;
    for (size_t i = 0; i < hdr.size(); ++i) col[hdr[i]] = static_cast<int>(i);
    const char* need[] = {"model", "N", "K", "M", "m_tile", "step", "protocol",
                          "latency_us_p50"};
    for (const char* c : need)
      if (!col.count(c)) {
        std::fprintf(stderr, "[bench] resume: %s 缺列 %s，不做续跑\n",
                     path.c_str(), c);
        source.clear();
        return false;
      }
    const bool has_config = col.count("config") != 0;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      const std::vector<std::string> f = split_csv_line(line);
      if (f.size() < hdr.size()) continue;
      const std::string lat = f[col["latency_us_p50"]];
      if (lat.empty() || lat == "nan" || lat == "NaN") continue;   // 无效行 -> 重测
      CsvRow r;
      r.model = f[col["model"]];
      r.N = std::atoi(f[col["N"]].c_str());
      r.K = std::atoi(f[col["K"]].c_str());
      r.M = std::atoi(f[col["M"]].c_str());
      r.m_tile = std::atoi(f[col["m_tile"]].c_str());
      r.step = std::atoi(f[col["step"]].c_str());
      r.protocol = f[col["protocol"]];
      if (has_config) r.config = f[col["config"]];
      add(is_tune, r);
      if (is_tune) ++tune_rows_; else ++canon_rows_;
      ++rows;
    }
    return true;
  }

 private:
  std::set<std::string> keys_;
  size_t canon_rows_ = 0;
  size_t tune_rows_ = 0;
};

// ---------------------------------------------------------------------------
// 不完整组清理（orchestrator 2026-09-13 要求）
//   看门狗 abort / timeout / kill 可能留下「只写了 1~2 个协议」的组，这些行的
//   测量条件（时钟、L2 状态、是否被打断）不可信，所以整组删掉重测。
//   组 = (model,N,K,M,m_tile,step,config)；完整 = ISO/B2B/PDL 三行都有有效 latency。
//   原子写：先写 <path>.tmp 再 rename，避免清理到一半把文件搞坏。
// ---------------------------------------------------------------------------
struct PurgeResult {
  int kept = 0;
  int dropped = 0;
  int groups_dropped = 0;
  bool ok = false;
  std::string note;
};

PurgeResult purge_incomplete_groups(const std::string& path,
                                    const std::vector<std::string>& need_protocols) {
  // 组的「完整」定义 = --protocols 请求的协议都在（默认 ISO+B2B+PDL）。
  // tune 只跑 PDL 时，若不按请求集判定，会把已完成的 config 行当不完整全删掉
  // （2026-09-13 实测：分块 tune 每块都 purge 掉上一块 => 数据只剩最后一块）。
  std::set<std::string> need_pr;
  if (need_protocols.empty()) {
    need_pr.insert("ISO"); need_pr.insert("B2B"); need_pr.insert("PDL");
  } else {
    for (const auto& pr : need_protocols) {
      std::string u = pr;
      for (auto& c : u) c = static_cast<char>(::toupper(c));
      if (!u.empty()) need_pr.insert(u);
    }
  }
  PurgeResult r;
  std::ifstream in(path);
  if (!in) { r.note = "no such file"; return r; }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    const std::string clean = strip_cr(line);
    if (!clean.empty()) lines.push_back(clean);   // 重写后文件是纯 LF
  }
  in.close();
  if (lines.empty()) { r.note = "empty"; return r; }

  const std::vector<std::string> hdr = split_csv_line(lines[0]);
  std::map<std::string, int> col;
  for (size_t i = 0; i < hdr.size(); ++i) col[hdr[i]] = static_cast<int>(i);
  const char* need[] = {"model", "N", "K", "M", "m_tile", "step", "protocol",
                        "config", "latency_us_p50"};
  for (const char* c : need)
    if (!col.count(c)) { r.note = std::string("缺列 ") + c; return r; }

  struct Group {
    std::set<std::string> have;      // 已见到的有效协议
    std::vector<size_t> rows;
  };
  std::map<std::string, Group> groups;
  for (size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split_csv_line(lines[i]);
    if (f.size() < hdr.size()) continue;
    std::ostringstream ks;
    ks << f[col["model"]] << '|' << f[col["N"]] << '|' << f[col["K"]] << '|'
       << f[col["M"]] << '|' << f[col["m_tile"]] << '|' << f[col["step"]] << '|'
       << f[col["config"]];
    Group& g = groups[ks.str()];
    g.rows.push_back(i);
    const std::string lat = f[col["latency_us_p50"]];
    const bool valid = !lat.empty() && lat != "nan" && lat != "NaN";
    if (valid) g.have.insert(f[col["protocol"]]);
  }

  std::vector<bool> keep(lines.size(), false);
  keep[0] = true;                                   // 表头
  for (const auto& kv : groups) {
    const Group& g = kv.second;
    bool complete = true;
    for (const auto& np : need_pr)
      if (!g.have.count(np)) { complete = false; break; }
    if (complete) {
      for (size_t i : g.rows) keep[i] = true;
      r.kept += static_cast<int>(g.rows.size());
    } else {
      r.dropped += static_cast<int>(g.rows.size());
      ++r.groups_dropped;
    }
  }
  if (r.dropped == 0) { r.ok = true; r.note = "all groups complete"; return r; }

  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) { r.note = "cannot write " + tmp; return r; }
    for (size_t i = 0; i < lines.size(); ++i)
      if (keep[i]) out << lines[i] << "\n";
    out.flush();
  }
  if (::rename(tmp.c_str(), path.c_str()) != 0) {
    r.note = "rename failed";
    ::unlink(tmp.c_str());
    return r;
  }
  r.ok = true;
  r.note = "purged " + std::to_string(r.dropped) + " rows in " +
           std::to_string(r.groups_dropped) + " incomplete groups";
  return r;
}

// 同一个 CSV 只允许一个进程 append：拿不到锁就新开一份，避免两个 sweep 互踩
int g_csv_lock_fd = -1;
bool acquire_csv_lock(const std::string& csv_path) {
  const std::string lp = csv_path + ".lock";
  const int fd = ::open(lp.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd < 0) return false;
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    ::close(fd);
    return false;
  }
  g_csv_lock_fd = fd;      // 持有到进程退出（退出/被 kill 时内核自动释放）
  return true;
}

// out_dir 里最新的 <prefix>*.csv（排除 _detail）；找不到返回空串
std::string newest_csv(const std::string& dir, const std::string& prefix) {
  std::string best;
  time_t best_t = -1;
  DIR* d = ::opendir(dir.c_str());
  if (!d) return best;
  while (struct dirent* ent = ::readdir(d)) {
    const std::string name = ent->d_name;
    if (name.rfind(prefix, 0) != 0) continue;
    if (name.size() < 5 || name.substr(name.size() - 4) != ".csv") continue;
    if (name.find("_detail") != std::string::npos) continue;
    struct stat st {};
    const std::string full = dir + "/" + name;
    if (::stat(full.c_str(), &st) != 0) continue;
    if (st.st_mtime > best_t) { best_t = st.st_mtime; best = full; }
  }
  ::closedir(d);
  return best;
}

// 读 sibling JSON 里的 kernel_backend，避免把 stub 的结果当成真 kernel 的续跑起点
std::string json_backend_of(const std::string& csv_path) {
  std::string jp = csv_path;
  const size_t dot = jp.rfind('.');
  if (dot != std::string::npos) jp = jp.substr(0, dot);
  jp += ".json";
  std::ifstream in(jp);
  if (!in) return "";
  std::string all((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  const std::string key = "\"kernel_backend\": \"";
  const size_t a = all.find(key);
  if (a == std::string::npos) return "";
  const size_t b = all.find('"', a + key.size());
  return b == std::string::npos ? "" : all.substr(a + key.size(), b - a - key.size());
}

// ------------------------------------------------------------- 单点测量 -----
struct PointResult {
  CorrResult corr;
  TimingStats iso, b2b, pdl;
  int padded_n = 0, sets = 0, smem = 0, ctas = 0, threads = 0;
  size_t set_bytes = 0;
  std::string mechanism, skip_reason;
};

bool setup_run_ctx(Globals& g, WeightCache& cache, const ModelDevice& mdl,
                   const MDevice& md, const bench::VariantOps* v, RunCtx* ctx,
                   const Options& o, std::string* why) {
  ctx->v = v;
  ctx->stream = g.stream;
  ctx->use_api = o.force_api_launch;
  ctx->buf.M = md.M;
  ctx->buf.N = mdl.n_ker;
  ctx->buf.K = mdl.K;
  ctx->buf.padded_n = v->fn_padded_n(mdl.n_ker);
  ctx->buf.activation = md.d_act;
  ctx->buf.weight = nullptr;                 // 每次 launch 指向当前 set
  ctx->buf.weight_scales = mdl.d_wscale;
  ctx->buf.activation_scales = md.d_ascale;
  ctx->buf.output = md.d_out;
  ctx->buf.splitk_ws = md.d_ws;
  ctx->buf.splitk_sem = md.d_sem;
  ctx->sem_ints = md.sem_ints;
  ctx->sem_slots = md.sem_slots;

  WeightGroup* grp = cache.get(v->spec.prepack, v->spec.bm, v->spec.tiles,
                               mdl.n_ker, mdl.K, mdl.d_weight_raw, v, g.stream,
                               o.verbose);
  if (!grp) { *why = "weight group alloc failed"; return false; }
  ctx->group = grp;
  ctx->buf.weight = grp->bufs[0];
  ctx->w_maps.assign(grp->sets, CUtensorMap{});
  for (int i = 0; i < grp->sets; ++i) {
    bench::DeviceBuffers b = ctx->buf;
    b.weight = grp->bufs[i];
    const cudaError_t e = v->fn_make_maps(b, &ctx->w_maps[i], &ctx->a_map);
    if (e != cudaSuccess) {
      *why = std::string("make_*_tma failed: ") + cudaGetErrorString(e);
      return false;
    }
  }
  if (v->fn_prepare) {
    const cudaError_t pe = v->fn_prepare(ctx->buf);
    if (pe != cudaSuccess && pe != cudaErrorInvalidValue) {
      *why = std::string("prepare failed: ") + cudaGetErrorString(pe);
      return false;
    }
  }
  for (int i = 0; i < kWarmupLaunches; ++i) {          // timed region 外
    const cudaError_t e = do_launch(*ctx, i % grp->sets, false);
    if (e != cudaSuccess) {
      *why = std::string("warmup launch failed: ") + cudaGetErrorString(e);
      return false;
    }
  }
  const cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) {
    *why = std::string("warmup sticky error: ") + cudaGetErrorString(le);
    return false;
  }
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  return true;
}

// 每测完一个协议就回调一次 -> 立刻 append+flush 进 CSV（被 kill 也不丢已测数据）
using EmitFn = std::function<void(const char* protocol, const TimingStats& st,
                                  const PointResult& pr)>;

PointResult measure_point(Globals& g, WeightCache& cache, const ModelDevice& mdl,
                          const MDevice& md, const bench::VariantOps* v,
                          const Options& o, const TimingPlan& plan,
                          bool with_correctness, const std::array<bool, 3>& want,
                          const EmitFn& emit) {
  PointResult res;
  RunCtx ctx;
  std::string why;
  res.padded_n = v->fn_padded_n(mdl.n_ker);
  res.set_bytes = v->fn_set_bytes(mdl.n_ker, mdl.K);
  res.smem = v->fn_smem_bytes(md.M, mdl.K);
  res.ctas = v->fn_num_ctas(mdl.n_ker);
  res.threads = v->fn_num_threads();
  res.mechanism = v->mechanism;
  if (!setup_run_ctx(g, cache, mdl, md, v, &ctx, o, &why)) {
    res.skip_reason = why;
    return res;
  }
  res.sets = ctx.group->sets;
  if (with_correctness) res.corr = run_correctness(g, ctx, md, mdl, o);
  if (want[0]) { res.iso = time_iso(g, ctx, plan);          emit("ISO", res.iso, res); }
  if (want[1]) { res.b2b = time_batch(g, ctx, plan, false); emit("B2B", res.b2b, res); }
  if (want[2]) { res.pdl = time_batch(g, ctx, plan, true);  emit("PDL", res.pdl, res); }
  return res;
}

struct Skipped {
  std::string model;
  int step, M, m_tile;
  std::string reason;
};

struct BestTune {
  std::string model;
  int step = 0, M = 0;
  std::string config;
  double us = 0.0, gbps = 0.0;
};

CsvRow make_base_row(const ModelShape& shape, const ModelDevice& mdl,
                     const MDevice& md, const bench::VariantOps* v, int step,
                     const PointResult& pr, const TimingPlan& plan,
                     int clocks_mhz) {
  CsvRow r;
  r.model = shape.id;
  r.N = mdl.n_log;                    // CSV 记 logical N（padding 只在明细/JSON）
  r.K = mdl.K;
  r.M = md.M;
  r.m_tile = v->m_tile;
  r.step = step;
  r.step_name = (step >= 0 && step < bench::kNumSteps) ? bench::kStepNames[step]
                                                       : v->step_name;
  r.rel_l2 = pr.corr.rel_l2;
  r.rel_l2_launcher = pr.corr.rel_l2_launcher;
  r.pass = pr.corr.ran ? (pr.corr.pass ? "1" : "0") : "na";
  r.config = v->config_str;
  r.clocks_mhz = clocks_mhz;
  r.reps = plan.reps;
  r.iso_samples = plan.iso_samples;
  r.b2b_batch = plan.batch;
  r.pdl_batch = plan.batch;
  r.kind = v->step >= 0 ? "canonical" : "tune";
  r.tune_index = v->tune_index;
  r.padded_n = pr.padded_n;
  r.weight_sets = pr.sets;
  r.weight_set_bytes = pr.set_bytes;
  r.smem_bytes = pr.smem;
  r.num_ctas = pr.ctas;
  r.num_threads = pr.threads;
  r.mechanism = pr.mechanism;
  return r;
}

// --protocols 过滤（空 = 全跑）
bool proto_wanted(const Options& o, const char* name) {
  if (o.protocols.empty()) return true;
  for (const auto& p : o.protocols)
    if (p == name) return true;
  return false;
}
std::array<bool, 3> proto_mask(const Options& o) {
  return std::array<bool, 3>{proto_wanted(o, "ISO"), proto_wanted(o, "B2B"),
                             proto_wanted(o, "PDL")};
}

// 把一个协议的计时结果填进 CSV 行（bandwidth 分子 = N*K + M*K + 2*M*N）
void fill_row_stats(CsvRow* r, const char* protocol, const TimingStats& st) {
  r->protocol = protocol;
  r->p50 = st.p50_us;
  r->p30 = st.p30_us;
  r->min_us = st.min_us;
  r->mean_us = st.mean_us;
  r->samples = st.n;
  const double bytes = modeled_bytes(r->N, r->K, r->M);
  if (std::isfinite(st.p50_us) && st.p50_us > 0) {
    r->bw = bytes / (st.p50_us * 1.0e-6) / 1.0e9;
    r->pct = r->bw / kSpecPeakGbps;
  }
}

}  // namespace
}  // namespace bench

// ---------------------------------------------------- registry 实现（唯一）--
namespace bench {
std::vector<VariantOps>& registry() {
  static std::vector<VariantOps> r;
  return r;
}
void register_variant(const VariantOps& v) { registry().push_back(v); }

// S6 的 split-K 是**条件启用**（configs.cuh：ctas_no_split < 78 且 M >= 8），
// 所以 on/off 两条都注册了，这里按 (N, M) 挑该跑的那条。
namespace {
bool want_splitk(int step, int m_tile, int n, int k, int m) {
  return splitk_enabled_at_runtime_k(step, m_tile, n, k, m);
}
// shape tag 优先：override variant（shape_n==n && shape_k==k）> 形状无关（shape_n==0）。
// S 路 split-K 泛化后 step6 会同时注册 S=1/2/4/8 四条同几何 variant，所以这里必须
// 按**有效 S** 过滤：主阶梯（CONTRACT §3 的 S6）与 --splitk-ablation 的语义都是
// 「开 = 2 路」，不能被 S=4/8 顶掉。S 扫描走 find_splitk_factor_sweep()，不经过这里。
const VariantOps* pick(int m_tile, int step, int n, int k, int m, int role, int bm) {
  const bool sk = want_splitk(step, m_tile, n, k, m);
  const int want_factor = sk ? 2 : 1;
  const VariantOps* fallback = nullptr;
  const VariantOps* generic = nullptr;
  for (const VariantOps& v : registry()) {
    if (v.m_tile != m_tile || v.step != step || v.tune_index != -1) continue;
    if (v.role != role) continue;
    if (bm > 0 && v.spec.bm != bm) continue;
    const bool shape_ok = (v.shape_n == 0) || (v.shape_n == n && v.shape_k == k);
    if (!shape_ok) continue;
    if (step == 5 && sk_factor_of(v.spec) != want_factor) continue;
    if (v.shape_n == n && v.shape_k == k) return &v;   // 形状精确匹配立即返回
    if (!generic) generic = &v;
    if (!fallback) fallback = &v;
  }
  return generic ? generic : fallback;
}
}  // namespace

const VariantOps* find_canonical(int m_tile, int step, int n, int k, int m) {
  const int want = canonical_bm_shape(step, m_tile, n, k);
  if (const VariantOps* v = pick(m_tile, step, n, k, m, /*role=*/0, want)) return v;
  int rank[kCandidateBmCount] = {};
  bm_rank_list(n, rank);
  for (int i = 0; i < kCandidateBmCount; ++i)          // 回退：按 BM rank
    if (const VariantOps* v = pick(m_tile, step, n, k, m, 0, rank[i])) return v;
  return nullptr;
}

std::vector<const VariantOps*> find_canonical_all(int m_tile, int step, int n,
                                                 int k, int m) {
  std::vector<const VariantOps*> out;
  if (const VariantOps* v = find_canonical(m_tile, step, n, k, m)) out.push_back(v);
  // M_TILE=128 的第二条 canonical（role==1，kernels/README.md §8：WG1/REG232 0 spill）
  if (const VariantOps* a = pick(m_tile, step, n, k, m, /*role=*/1, /*bm=*/0))
    out.push_back(a);
  return out;
}

std::vector<const VariantOps*> find_splitk_pair(int m_tile, int step, int n,
                                               int k) {
  std::vector<const VariantOps*> out;
  const int want = canonical_bm_shape(step, m_tile, n, k);
  for (bool sk : {false, true}) {
    const VariantOps* hit = nullptr;
    for (const VariantOps& v : registry())
      if (v.m_tile == m_tile && v.step == step && v.tune_index == -1 &&
          v.role == 0 && v.spec.bm == want && v.spec.splitk_cta == sk &&
          // 消融的 on/off 一对是 **2 路** vs 关；别让 S=4/8 混进来
          sk_factor_of(v.spec) == (sk ? 2 : 1) &&
          ((v.shape_n == n && v.shape_k == k) || v.shape_n == 0)) {
        if (!hit || (v.shape_n == n)) hit = &v;
      }
    if (hit) out.push_back(hit);
  }
  return out;
}

// 触发点放置消融三变体（role=2，M_TILE=8，bm 与主阶梯 pick 一致）
std::vector<const VariantOps*> find_epi_triple(int m_tile, int bm) {
  std::vector<const VariantOps*> out;
  for (const VariantOps& v : registry()) {
    if (v.m_tile != m_tile || v.step != 4 || v.role != 2 ||
        v.tune_index != -1 || v.spec.bm != bm)
      continue;
    out.push_back(&v);
  }
  // 排序：producer(pdl1,epi0) -> both(1,1) -> store(0,1)
  std::sort(out.begin(), out.end(), [](const VariantOps* a, const VariantOps* b) {
    auto key = [](const VariantOps* v) {
      return v->spec.pdl ? (v->spec.epi_overlap ? 1 : 0) : 2;
    };
    return key(a) < key(b);
  });
  return out;
}

const char* epi_trigger_tag(const VariantOps& v) {
  if (v.spec.pdl && !v.spec.epi_overlap) return "producer";
  if (v.spec.pdl && v.spec.epi_overlap) return "both";
  return "store";
}

// ---- S 路 split-K 扫描：按 (基准几何, S) 找 variant ------------------------
namespace {
struct SpecGeom {
  int m_tile = 0, bm = 0, wg = 0, regs = 0, stg = 0, tiles = 0, subs = 0;
  bool pp = false, pdl = false, epi = false, sism = false;
};
SpecGeom geom_of(const ConfigSpec& s) {
  SpecGeom g;
  g.m_tile = s.m_tile; g.bm = s.bm; g.wg = s.wg; g.regs = s.math_regs;
  g.stg = s.stg; g.tiles = s.tiles; g.subs = s.subs;
  g.pp = s.prepack; g.pdl = s.pdl; g.epi = s.epi_overlap;
  g.sism = s.scale_in_smem;
  return g;
}
bool same_geom(const SpecGeom& a, const SpecGeom& b) {
  return a.m_tile == b.m_tile && a.bm == b.bm && a.wg == b.wg &&
         a.regs == b.regs && a.stg == b.stg && a.tiles == b.tiles &&
         a.subs == b.subs && a.pp == b.pp && a.pdl == b.pdl && a.epi == b.epi &&
         a.sism == b.sism;
}
std::string geom_str(const SpecGeom& g) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "BM%d/WG%d/T%d/REG%d/STG%d/SUBS%d/SISM%d",
                g.bm, g.wg, g.tiles, g.regs, g.stg, g.subs, g.sism ? 1 : 0);
  return buf;
}
// 同几何 + 指定 S；shape-tagged（tune override）优先于形状无关的 canonical
const VariantOps* find_geom_factor(int m_tile, int step, int n, int k,
                                   const SpecGeom& g, int factor) {
  const VariantOps* generic = nullptr;
  for (const VariantOps& v : registry()) {
    if (v.m_tile != m_tile || v.step != step) continue;
    if (v.tune_index != -1 || v.role != 0) continue;
    if (sk_factor_of(v.spec) != factor) continue;
    if (!same_geom(geom_of(v.spec), g)) continue;
    if (v.shape_n == n && v.shape_k == k) return &v;
    if (v.shape_n == 0 && !generic) generic = &v;
  }
  return generic;
}
}  // namespace

std::vector<FactorPick> find_splitk_factor_sweep(int m_tile, int step, int n,
                                                int k, int m,
                                                const std::vector<int>& factors) {
  std::vector<FactorPick> out;
  std::vector<SpecGeom> bases;
  // base0：该形状该 M 的现役 canonical（有 tune override 就是 override 几何）
  if (const VariantOps* v = find_canonical(m_tile, step, n, k, m))
    bases.push_back(geom_of(v->spec));
  // base1：形状无关 canonical（README §6.2 的 S6 几何，BM 按整除 N/CTA 甜区选）
  {
    const ConfigSpec c = canonical_for(step, m_tile, canonical_bm(step, m_tile, n));
    if (feasible(c)) {
      const SpecGeom g = geom_of(c);
      bool dup = false;
      for (const SpecGeom& b : bases) if (same_geom(b, g)) dup = true;
      if (!dup) bases.push_back(g);
    }
  }
  for (size_t bi = 0; bi < bases.size(); ++bi) {
    for (int f : factors) {
      FactorPick fp;
      fp.base = static_cast<int>(bi);
      fp.factor = f;
      fp.geom = geom_str(bases[bi]);
      fp.v = find_geom_factor(m_tile, step, n, k, bases[bi], f);
      out.push_back(fp);
    }
  }
  return out;
}

std::vector<const VariantOps*> find_tune(int m_tile, int step) {
  std::vector<const VariantOps*> out;
  for (const VariantOps& v : registry())
    if (v.m_tile == m_tile && v.step == step && v.tune_index >= 0)
      out.push_back(&v);
  std::sort(out.begin(), out.end(), [](const VariantOps* a, const VariantOps* b) {
    return a->tune_index < b->tune_index;
  });
  return out;
}

bool tune_step_instantiated(int m_tile, int step) {
  return !find_tune(m_tile, step).empty();
}
}  // namespace bench

// ------------------------------------------------------------------ main ----
using namespace bench;   // helper 都在 namespace bench 的内部链接块里

int main(int argc, char** argv) {
  Options o = parse_args(argc, argv);
  if (o.out_dir.empty()) o.out_dir = default_out_dir();

  if (!o.purge_only.empty()) {          // 离线工具模式：不初始化 CUDA
    const PurgeResult r = purge_incomplete_groups(o.purge_only, o.protocols);
    std::printf("[bench_decode] purge %s -> ok=%d kept=%d dropped=%d groups=%d (%s)\n",
                o.purge_only.c_str(), r.ok ? 1 : 0, r.kept, r.dropped,
                r.groups_dropped, r.note.c_str());
    return r.ok ? 0 : 2;
  }

  bench::register_all();
  const size_t n_variants = bench::registry().size();
  size_t n_canon = 0, n_tune = 0;
  for (const auto& v : bench::registry())
    if (v.tune_index >= 0) ++n_tune; else ++n_canon;

  std::string models_path = o.models_json;
  if (models_path.empty())
    models_path = bench::exe_dir() + "/../models/models.json";
  bool fallback = false;
  std::string model_note;
  std::vector<ModelShape> all_models =
      bench::load_models(models_path, &fallback, &model_note);
  if (fallback)
    std::fprintf(stderr, "[bench_decode] WARNING: %s\n", model_note.c_str());

  std::vector<ModelShape> models;
  if (o.models.empty()) {
    models = all_models;
  } else {
    for (const auto& want : o.models) {
      bool found = false;
      for (const auto& m : all_models)
        if (m.id == want) { models.push_back(m); found = true; break; }
      if (!found)
        std::fprintf(stderr, "[bench_decode] WARNING: 模型 %s 不在 models.json，已跳过\n",
                     want.c_str());
    }
  }
  if (o.demo && models.size() > 1) models.resize(1);
  if (models.empty()) {
    std::fprintf(stderr, "[bench_decode] 没有可跑的模型\n");
    return 2;
  }

  if (o.list_only) {
    std::printf("models (%zu, from %s%s):\n", models.size(), models_path.c_str(),
                fallback ? ", FALLBACK" : "");
    for (const auto& m : models)
      std::printf("  %-18s N=%-6d K=%-6d hidden=%-5d bm=%-3d layer=%s\n",
                  m.id.c_str(), m.N, m.K, m.hidden, bench::choose_bm(m.N),
                  m.layer.c_str());
    std::printf("registered variants: %zu (canonical=%zu tune=%zu)\n",
                n_variants, n_canon, n_tune);
    for (int mt : {8, 16, 32, 64, 128}) {
      int canon = 0, tune = 0;
      for (const auto& v : bench::registry()) {
        if (v.m_tile != mt) continue;
        if (v.tune_index >= 0) ++tune; else ++canon;
      }
      std::printf("  M_TILE=%3d canonical=%-3d tune=%d\n", mt, canon, tune);
    }
    std::printf("canonical (M_TILE=8):\n");
    for (int s = 0; s < bench::kNumSteps; ++s) {
      const bench::VariantOps* v = bench::find_canonical(8, s, 6144, 7168, 8);
      if (v) std::printf("  S%d %-12s %s\n", s, bench::kStepNames[s],
                         v->config_str.c_str());
      else std::printf("  S%d %-12s <not instantiated>\n", s, bench::kStepNames[s]);
    for (const auto& v : bench::registry())
      if (v.role == 2)
        std::printf("  [role2] mt=%d step=%d %s\n", v.m_tile, v.step,
                    v.config_str.c_str());
    }
    std::printf("tune steps instantiated (M_TILE=8):");
    for (int s = 0; s < bench::kNumSteps; ++s)
      if (bench::tune_step_instantiated(8, s)) std::printf(" S%d", s);
    std::printf("\n");
    return 0;
  }

  // ---- CUDA init（卡由 tools/gpurun_dg.sh 通过 CUDA_VISIBLE_DEVICES 指定）----
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major != 9) {
    std::fprintf(stderr, "[bench_decode] 需要 SM90（H20/H100），当前 sm_%d%d\n",
                 prop.major, prop.minor);
    return 2;
  }
  const EnvInfo env = bench::collect_env_info(0);
  if (!env.gpurun_gpu.empty() && !env.smi_index.empty() &&
      env.gpurun_gpu != env.smi_index) {
    std::fprintf(stderr,
                 "[bench_decode] WARNING: gpurun_dg 说 GPU%s，但 PCI bus id 反查是 "
                 "GPU%s（以反查为准）\n",
                 env.gpurun_gpu.c_str(), env.smi_index.c_str());
  }
  // CONTRACT §0（2026-09-13 v2）：GPU2/3 是 keeper 预留但物理空闲的卡，用户授权
  // 借用；gpurun_dg.sh v2 的看门狗一有外来 compute 落卡就 abort 让路（exit 75）。
  // 所以这里不再拒绝 2/3，只把「借的哪张卡 + 卡上有哪些 keeper」如实记进 JSON。
  if (env.pool == "BORROW{2,3}")
    std::printf("[bench_decode] BORROW GPU%s（keeper 预留卡，看门狗保护中）keeper_apps=%zu\n",
                env.smi_index.c_str(), env.keeper_apps.size());
  if (env.exclusive_verdict == "fail" && !o.allow_shared) {
    std::fprintf(stderr,
                 "[bench_decode] ERROR: GPU%s 上有别的 compute app（独占检查失败）。\n"
                 "             应该由 tools/gpurun_dg.sh 保证干净卡；确认无误可加 "
                 "--allow-shared 强跑（数据会被污染）。\n",
                 env.smi_index.c_str());
    for (const auto& a : env.other_apps)
      std::fprintf(stderr, "             foreign app: %s\n", a.c_str());
    return 3;
  }

  ::mkdir(o.out_dir.c_str(), 0755);
  const std::string ts = bench::utc_stamp();
  std::string csv_path = o.out_dir + "/bench_" + ts + ".csv";
  std::string detail_path = o.out_dir + "/bench_" + ts + "_detail.csv";
  const std::string json_path = o.out_dir + "/bench_" + ts + ".json";

  // ---- 断点续跑：找同批次 CSV，读已完成组合，之后 append 模式继续写 ----
  DoneSet done;
  bool append_mode = false;
  std::string resume_note = "disabled";
  const std::string my_backend =
#if defined(BENCH_USE_STUB)
      "stub";
#else
      "kernels/decode_gemm.cuh";
#endif
  if (o.resume) {
    std::string src = o.resume_path;
    if (src.empty()) src = newest_csv(o.out_dir, "bench_");
    if (src.empty()) {
      resume_note = "requested，但 " + o.out_dir + " 里没有 bench_*.csv";
    } else {
      const std::string backend = json_backend_of(src);
      const bool stub_mismatch =
          (my_backend == "stub") != (backend.find("stub") != std::string::npos);
      if (!backend.empty() && stub_mismatch) {
        resume_note = "refused：" + src + " 的 kernel_backend=\"" + backend +
                      "\" 与当前二进制(\"" + my_backend + "\")不一致";
        std::fprintf(stderr, "[bench_decode] WARNING: %s -> 新开 CSV\n",
                     resume_note.c_str());
      } else if (!acquire_csv_lock(src)) {
        resume_note = "refused：" + src + " 正被另一个进程写着（.lock 被占）";
        std::fprintf(stderr,
                     "[bench_decode] WARNING: %s -> 本次新开一份 CSV，不互踩\n",
                     resume_note.c_str());
      } else {
        // 先 purge 不完整组，再 load done 集合（顺序不能反，否则被 purge 的行
        // 还会被当成「已完成」而不再补测）
        const PurgeResult pr = purge_incomplete_groups(src, o.protocols);
        if (pr.dropped)
          std::fprintf(stderr, "[bench_decode] resume 清理 %s：%s\n",
                       src.c_str(), pr.note.c_str());
        if (!done.load(src, /*is_tune=*/false)) {
          resume_note = "解析失败：" + src;
        } else {
        append_mode = true;
        csv_path = src;
        const size_t dot = src.rfind('.');
        detail_path = (dot == std::string::npos ? src : src.substr(0, dot)) +
                      "_detail.csv";
        const PurgeResult pd = purge_incomplete_groups(detail_path, o.protocols);
        resume_note = "appending to " + src;
        if (pd.dropped)
          resume_note += " (detail: " + pd.note + ")";
        }
      }
    }
  }

  // ---- split-K 消融输出（固定文件名，存在则 purge + append）----
  std::string ablation_path;
  CsvWriter ablation_csv;
  bool ablation_append = false;
  if (o.splitk_ablation) {
    ablation_path = !o.ablation_out.empty()
                        ? o.ablation_out
                        : o.out_dir + "/ablation_splitk.csv";
    if (::access(ablation_path.c_str(), F_OK) == 0) {
      purge_incomplete_groups(ablation_path, o.protocols);
      if (done.load(ablation_path, /*is_tune=*/true)) ablation_append = true;
    }
    if (!ablation_csv.open(ablation_path, kAblationHeader, ablation_append)) {
      std::fprintf(stderr, "[bench_decode] 打不开 %s\n", ablation_path.c_str());
      return 2;
    }
    std::printf("[bench_decode] split-K 消融 -> %s (%s)\n",
                ablation_path.c_str(), ablation_append ? "append" : "new");
  }

  // ---- S 路 split-K 扫描输出（固定文件名，存在则 purge + append）----
  std::string factor_path;
  CsvWriter factor_csv;
  bool factor_append = false;
  if (o.splitk_factor_sweep) {
    factor_path = !o.factor_out.empty() ? o.factor_out
                                        : o.out_dir + "/splitk_factor_sweep.csv";
    if (::access(factor_path.c_str(), F_OK) == 0) {
      purge_incomplete_groups(factor_path, o.protocols);
      if (done.load(factor_path, /*is_tune=*/true)) factor_append = true;
    }
    if (!factor_csv.open(factor_path, kFactorHeader, factor_append)) {
      std::fprintf(stderr, "[bench_decode] 打不开 %s\n", factor_path.c_str());
      return 2;
    }
    std::printf("[bench_decode] S 路 split-K 扫描 -> %s (%s)\n",
                factor_path.c_str(), factor_append ? "append" : "new");
  }

  std::string placement_path;
  CsvWriter placement_csv;
  bool placement_append = false;
  if (o.pdl_placement) {
    placement_path = o.out_dir + "/ablation_pdl_placement.csv";
    if (::access(placement_path.c_str(), F_OK) == 0) {
      purge_incomplete_groups(placement_path, o.protocols);
      if (done.load(placement_path, /*is_tune=*/true)) placement_append = true;
    }
    if (!placement_csv.open(placement_path, kPlacementHeader, placement_append)) {
      std::fprintf(stderr, "[bench_decode] 打不开 %s\n", placement_path.c_str());
      return 2;
    }
    std::printf("[bench_decode] 触发点放置消融 -> %s (%s)\n",
                placement_path.c_str(), placement_append ? "append" : "new");
  }

  CsvWriter csv, detail, tune_csv, tune_detail, tune_best_csv;
  if (!csv.open(csv_path, kCsvHeader, append_mode) ||
      !detail.open(detail_path, kDetailHeader, append_mode)) {
    std::fprintf(stderr, "[bench_decode] 打不开 %s\n", csv_path.c_str());
    return 2;
  }
  std::string tune_path, tune_detail_path, tune_best_path;
  bool tune_append = false;
  if (o.tune) {
    // --tune-out：固定文件名（例如 results/tune_kimi_k3.csv）。文件已存在就
    // purge 不完整组 + append，这样分批跑 / 被看门狗 abort 后重跑都往同一个文件
    // 补缺口（orchestrator 2026-09-13 要求：每 config 一行，abort 不丢）
    tune_path = !o.tune_out.empty() ? o.tune_out
                                    : o.out_dir + "/tune_" + ts + ".csv";
    const size_t tdot = tune_path.rfind('.');
    tune_detail_path =
        (tdot == std::string::npos ? tune_path : tune_path.substr(0, tdot)) +
        "_detail.csv";
    tune_best_path = o.out_dir + "/tune_best_" + ts + ".csv";
    if (::access(tune_path.c_str(), F_OK) == 0) {
      const PurgeResult pt = purge_incomplete_groups(tune_path, o.protocols);
      purge_incomplete_groups(tune_detail_path, o.protocols);
      if (done.load(tune_path, /*is_tune=*/true)) {
        tune_append = true;
        std::printf("[bench_decode] tune 续跑 %s（已有 %zu 行；purge: %s）\n",
                    tune_path.c_str(), done.tune_count(), pt.note.c_str());
      }
    }
    if (!tune_append && o.resume && append_mode) {
      const std::string tsrc = newest_csv(o.out_dir, "tune_");
      if (!tsrc.empty()) purge_incomplete_groups(tsrc, o.protocols);
      if (!tsrc.empty() && done.load(tsrc, /*is_tune=*/true)) {
        tune_path = tsrc;
        const size_t dot = tsrc.rfind('.');
        tune_detail_path =
            (dot == std::string::npos ? tsrc : tsrc.substr(0, dot)) +
            "_detail.csv";
        tune_append = true;
      }
    }
    if (!tune_csv.open(tune_path, kCsvHeader, tune_append) ||
        !tune_detail.open(tune_detail_path, kDetailHeader, tune_append) ||
        !tune_best_csv.open(tune_best_path, kTuneBestHeader)) {
      std::fprintf(stderr, "[bench_decode] 打不开 %s\n", tune_path.c_str());
      return 2;
    }
  }
  if (o.resume)
    std::printf("[bench_decode] resume: %s (已完成 canonical=%zu tune=%zu 行)\n",
                resume_note.c_str(), done.canon_count(), done.tune_count());

  std::printf(
      "[bench_decode] device=%s cuda_idx=%d physical_gpu=%s pci=%s SMs=%d "
      "mem=%.0fGiB\n",
      env.device_name.c_str(), env.device_index,
      env.smi_index.empty() ? "?" : env.smi_index.c_str(), env.pci_bus_id.c_str(),
      env.sm_count, env.mem_total / 1073741824.0);
  std::printf(
      "[bench_decode] clocks.sm=%dMHz(max %dkHz) mem=%dMHz util=%d%% "
      "mem_used=%dMiB exclusive=%s gpurun_target=%s\n",
      env.clock_cur_mhz, env.clock_max_khz, env.mem_clock_cur_mhz, env.util_pct,
      env.mem_used_mib, env.exclusive_verdict.c_str(),
      env.gpurun_target.empty() ? "-" : env.gpurun_target.c_str());
  if (!env.shell_note.empty())
    std::printf("[bench_decode] shell check: %s\n", env.shell_note.c_str());
  std::printf("[bench_decode] backend=%s launch=%s variants=%zu(canon %zu/tune %zu) out=%s\n",
#if defined(BENCH_USE_STUB)
              "STUB(流程自测,性能无意义)",
#else
              "kernels/decode_gemm.cuh",
#endif
              bench::registry().empty() ? "?" : bench::registry()[0].mechanism,
              n_variants, n_canon, n_tune, o.out_dir.c_str());
  if (o.force_api_launch)
    std::fprintf(stderr,
                 "[bench_decode] WARNING: --force-api-launch -> PSS attr 由 kernel "
                 "launcher 决定，B2B 与 PDL 不再可比\n");

  Globals g;
  g.clocks.smi_index = env.smi_index;
  CUDA_CHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&g.ev_start));
  CUDA_CHECK(cudaEventCreate(&g.ev_end));
  const size_t flush_bytes = static_cast<size_t>(o.l2_flush_mb) << 20;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g.flush_buf), flush_bytes));
  CUDA_CHECK(cudaMemsetAsync(g.flush_buf, 0, flush_bytes, g.stream));
  g.flush_n4 = flush_bytes / sizeof(float4);
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  const int clocks_start = g.clocks.read(true);

  WeightCache cache(static_cast<size_t>(o.weight_cache_gb * 1073741824.0));

  const auto t_start = std::chrono::steady_clock::now();
  std::vector<Skipped> skipped;
  std::vector<BestTune> bests;
  int points_done = 0, points_failed = 0, correctness_fail = 0, tune_points = 0;
  int points_resumed = 0, rows_resumed = 0, rows_written = 0;

  for (const ModelShape& shape : models) {
    std::printf("[bench_decode] === %s  N=%d K=%d (%s) ===\n", shape.id.c_str(),
                shape.N, shape.K, shape.layer.c_str());
    if (shape.K % 128 != 0) {
      std::fprintf(stderr, "[bench_decode] ERROR: %s 的 K=%d 不是 128 的倍数，跳过\n",
                   shape.id.c_str(), shape.K);
      continue;
    }
    if (shape.N % 128 != 0)
      std::fprintf(stderr,
                   "[bench_decode] WARNING: %s 的 N=%d 不是 128 的倍数，harness pad "
                   "到 %d（CSV 仍记 logical N，pad 明细在 _detail.csv/JSON）\n",
                   shape.id.c_str(), shape.N, ((shape.N + 127) / 128) * 128);

    ModelDevice mdl;
    mdl.shape = shape;
    mdl.n_log = shape.N;
    mdl.n_ker = ((shape.N + 127) / 128) * 128;
    mdl.K = shape.K;
    mdl.kb = shape.K / 128;

    const auto tg = std::chrono::steady_clock::now();
    mdl.host_w = bench::make_weights(mdl.n_log, mdl.K, o.seed);
    const HostAct host_a =
        bench::make_activation(128, mdl.K, o.seed ^ 0x9e3779b97f4a7c15ull);
    const double gen_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
    std::printf("[bench_decode] host 数据生成 %.1fs (weight %.1f MiB, sets 目标 >= %zu MiB)\n",
                gen_s, mdl.host_w.weight.size() / 1048576.0,
                kWorkingSetBytes >> 20);

    mdl.scale_rows_alloc = mdl.host_w.scale_rows_alloc;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdl.d_weight_raw),
                          static_cast<size_t>(mdl.n_ker) * mdl.K));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdl.d_wscale),
                          static_cast<size_t>(mdl.scale_rows_alloc) * mdl.kb *
                              sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdl.d_wdeq),
                          static_cast<size_t>(mdl.n_ker) * mdl.K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(mdl.d_weight_raw, mdl.host_w.weight.data(),
                          static_cast<size_t>(mdl.n_ker) * mdl.K,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mdl.d_wscale, mdl.host_w.scales.data(),
                          static_cast<size_t>(mdl.scale_rows_alloc) * mdl.kb *
                              sizeof(float),
                          cudaMemcpyHostToDevice));
    {
      const size_t total = static_cast<size_t>(mdl.n_ker) * mdl.K;
      const int threads = 256;
      const size_t blocks = (total + threads - 1) / threads;
      dequant_weight_kernel<<<static_cast<unsigned>(blocks), threads, 0, g.stream>>>(
          mdl.d_weight_raw, mdl.d_wscale, mdl.d_wdeq, mdl.n_ker, mdl.K);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
    }

    const TimingPlan plan = bench::plan_for(mdl.K, o);
    std::printf("[bench_decode] timing plan: reps=%d iso_samples=%d batch=%d\n",
                plan.reps, plan.iso_samples, plan.batch);

    for (int M : o.ms) {
      MDevice md;
      md.M = M;
      md.m_tile = bench::m_tile_for(M);
      md.rows_alloc = std::max(md.m_tile, 8);
      const int pdn_max = bench::padded_n(mdl.n_ker, 32);   // BM 最小 -> pad 最大
      // kernels/decode_gemm.cuh 的实际粒度：splitk_ws = [S][M][N] fp32（S = 该
      // M_TILE 已实例化 variant 里最大的 splitk_factor，S 路泛化后不再固定 2）；
      // splitk_sem = **每输出 tile 一个** int32（>= num_output_tiles，host 清零一次，
      // kernel atomicExch 自复位）；num_output_tiles 最大 = ceil(N / BM_min=32)。
      int max_sk_factor = 1;
      for (const bench::VariantOps& rv : bench::registry())
        if (rv.m_tile == md.m_tile)
          max_sk_factor = std::max(max_sk_factor, bench::sk_factor_of(rv.spec));
      const size_t ws_elems = static_cast<size_t>(max_sk_factor) *
                                  static_cast<size_t>(M) * mdl.n_ker + 64;
      const size_t sem_elems =
          static_cast<size_t>((mdl.n_ker + 31) / 32) + 64;
      md.sem_ints = static_cast<int>(sem_elems);
      md.sem_slots = std::max(1, o.sem_rotate);
      const size_t sem_total = sem_elems * static_cast<size_t>(md.sem_slots);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&md.d_act),
                            static_cast<size_t>(md.rows_alloc) * mdl.K));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&md.d_ascale),
                            static_cast<size_t>(md.rows_alloc) * mdl.kb *
                                sizeof(float)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&md.d_adeq),
                            static_cast<size_t>(M) * mdl.K * sizeof(float)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&md.d_out),
                            static_cast<size_t>(M) * pdn_max *
                                sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&md.d_ws),
                            ws_elems * sizeof(float)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&md.d_sem),
                            sem_total * sizeof(int)));
      CUDA_CHECK(cudaMemsetAsync(md.d_act, 0,
                                 static_cast<size_t>(md.rows_alloc) * mdl.K,
                                 g.stream));
      CUDA_CHECK(cudaMemsetAsync(md.d_ascale, 0,
                                 static_cast<size_t>(md.rows_alloc) * mdl.kb *
                                     sizeof(float), g.stream));
      CUDA_CHECK(cudaMemsetAsync(md.d_out, 0,
                                 static_cast<size_t>(M) * pdn_max *
                                     sizeof(__nv_bfloat16), g.stream));
      CUDA_CHECK(cudaMemsetAsync(md.d_ws, 0, ws_elems * sizeof(float), g.stream));
      CUDA_CHECK(cudaMemsetAsync(md.d_sem, 0, sem_total * sizeof(int),
                                 g.stream));
      CUDA_CHECK(cudaMemcpyAsync(
          md.d_act, host_a.act.data(),
          static_cast<size_t>(std::min(md.rows_alloc, host_a.rows_alloc)) * mdl.K,
          cudaMemcpyHostToDevice, g.stream));
      CUDA_CHECK(cudaMemcpyAsync(
          md.d_ascale, host_a.scales.data(),
          static_cast<size_t>(std::min(md.rows_alloc, host_a.rows_alloc)) *
              mdl.kb * sizeof(float),
          cudaMemcpyHostToDevice, g.stream));
      CUDA_CHECK(cudaStreamSynchronize(g.stream));

      // ---- fp32 参考：每 (model,M) 一次（与 step 无关）----
      if (!o.no_correctness) {
        const size_t total = static_cast<size_t>(M) * mdl.K;
        const int threads = 256;
        const size_t blocks = (total + threads - 1) / threads;
        dequant_act_kernel<<<static_cast<unsigned>(blocks), threads, 0, g.stream>>>(
            md.d_act, md.d_ascale, md.d_adeq, M, mdl.K);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> ref_full(static_cast<size_t>(M) * mdl.n_ker, 0.0f);
        float* d_ref = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ref),
                              static_cast<size_t>(M) * mdl.n_ker * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_ref, 0,
                                   static_cast<size_t>(M) * mdl.n_ker * sizeof(float),
                                   g.stream));
        const bool ok = bench::lt_sgemm_ref(g.lt, mdl.d_wdeq, md.d_adeq, d_ref,
                                            mdl.n_ker, M, mdl.K, g.stream);
        CUDA_CHECK(cudaStreamSynchronize(g.stream));
        if (ok) {
          CUDA_CHECK(cudaMemcpy(ref_full.data(), d_ref,
                                static_cast<size_t>(M) * mdl.n_ker * sizeof(float),
                                cudaMemcpyDeviceToHost));
        }
        cudaFree(d_ref);
        CUDA_CHECK(cudaGetLastError());
        if (!ok) {
          std::fprintf(stderr, "[bench_decode] WARNING: cublasLt 参考失败(M=%d)\n", M);
        } else {
          md.ref.assign(static_cast<size_t>(M) * mdl.n_log, 0.0f);
          for (int m = 0; m < M; ++m)
            for (int n = 0; n < mdl.n_log; ++n)
              md.ref[static_cast<size_t>(m) * mdl.n_log + n] =
                  ref_full[static_cast<size_t>(m) * mdl.n_ker + n];
          md.ref_ok = true;
        }
      }

      for (int step : o.steps) {
        // ------- 触发点放置消融（--pdl-placement，只对 S4 几何有意义）---------
        if (o.pdl_placement && step == 4) {
          const bench::VariantOps* mainv =
              bench::pick(md.m_tile, 4, mdl.n_ker, mdl.K, M, 0,
                          bench::canonical_bm_shape(4, md.m_tile, mdl.n_ker, mdl.K));
          const int bm = mainv ? mainv->spec.bm
                               : bench::canonical_bm_shape(4, md.m_tile, mdl.n_ker, mdl.K);
          const std::vector<const bench::VariantOps*> tri =
              bench::find_epi_triple(md.m_tile, bm);
          if (tri.size() < 3) {
            skipped.push_back({shape.id, step, M, md.m_tile,
                               "触发点三变体没实例化齐（bm=" + std::to_string(bm) + "）"});
            continue;
          }
          for (const bench::VariantOps* v : tri) {
            if (const char* bad = v->fn_supported(M, mdl.n_ker, mdl.K)) {
              skipped.push_back({shape.id, step, M, md.m_tile,
                                 std::string("placement ") + v->config_str + ": " + bad});
              continue;
            }
            std::array<bool, 3> want = {false, false, true};   // 只跑 PDL 协议
            PointResult pr = bench::measure_point(
                g, cache, mdl, md, v, o, plan, /*with_correctness=*/true, want,
                [&](const char* proto, const TimingStats& st,
                    const PointResult& r) {
                  CsvRow row = bench::make_base_row(shape, mdl, md, v, step, r,
                                                    plan, g.clocks.read(false));
                  fill_row_stats(&row, proto, st);
                  std::ostringstream line;
                  line << row_main(row) << ',' << epi_trigger_tag(*v);
                  placement_csv.write(line.str());
                  done.add(true, row);
                  ++rows_written;
                });
            if (!pr.skip_reason.empty()) {
              skipped.push_back({shape.id, step, M, md.m_tile, pr.skip_reason});
              ++points_failed;
            } else {
              ++points_done;
              std::printf("[bench_decode] PLACEMENT %-16s M=%3d trigger=%-8s "
                          "rel_l2=%.2e PDL=%.2f us  %s\n",
                          shape.id.c_str(), M, epi_trigger_tag(*v),
                          pr.corr.rel_l2, pr.pdl.p50_us, v->config_str.c_str());
            }
          }
          continue;
        }

        // ---------------- split-K 消融（--splitk-ablation，只对 S5 有意义）------
        if (o.splitk_ablation && step == 5) {
          const std::vector<const bench::VariantOps*> pair =
              bench::find_splitk_pair(md.m_tile, 5, mdl.n_ker, mdl.K);
          if (pair.size() < 2) {
            skipped.push_back({shape.id, step, M, md.m_tile,
                               "split-K on/off 两条没都实例化，做不了消融"});
          }
          for (const bench::VariantOps* v : pair) {
            if (const char* bad = v->fn_supported(M, mdl.n_ker, mdl.K)) {
              skipped.push_back({shape.id, step, M, md.m_tile,
                                 std::string("ablation ") + v->config_str + ": " +
                                     bad});
              continue;
            }
            std::array<bool, 3> want = proto_mask(o);
            if (ablation_append) {
              const char* pn[3] = {"ISO", "B2B", "PDL"};
              CsvRow probe_row = bench::make_base_row(shape, mdl, md, v, step,
                                                      PointResult{}, plan, -1);
              bool all = true;
              for (int pi = 0; pi < 3; ++pi) {
                probe_row.protocol = pn[pi];
                if (!want[pi]) continue;
                if (done.has(true, probe_row)) { want[pi] = false; ++rows_resumed; }
                else all = false;
              }
              if (all) { ++points_resumed; continue; }
            }
            PointResult pr = bench::measure_point(
                g, cache, mdl, md, v, o, plan, /*with_correctness=*/true, want,
                [&](const char* proto, const TimingStats& st,
                    const PointResult& r) {
                  CsvRow row = bench::make_base_row(shape, mdl, md, v, step, r,
                                                    plan, g.clocks.read(false));
                  fill_row_stats(&row, proto, st);
                  std::ostringstream line;
                  line << row_main(row) << ','
                       << (v->spec.splitk_cta ? 1 : 0);
                  ablation_csv.write(line.str());
                  done.add(true, row);
                  ++rows_written;
                });
            if (!pr.skip_reason.empty()) {
              skipped.push_back({shape.id, step, M, md.m_tile, pr.skip_reason});
              ++points_failed;
            } else {
              ++points_done;
              std::printf(
                  "[bench_decode] ABLATION %-16s M=%3d splitk=%d rel_l2=%.2e "
                  "pass=%s ISO=%.2f B2B=%.2f PDL=%.2f us  %s\n",
                  shape.id.c_str(), M, v->spec.splitk_cta ? 1 : 0,
                  pr.corr.rel_l2, pr.corr.ran ? (pr.corr.pass ? "Y" : "N") : "-",
                  pr.iso.p50_us, pr.b2b.p50_us, pr.pdl.p50_us,
                  v->config_str.c_str());
            }
          }
          continue;      // 消融模式不再跑常规 canonical
        }

        // --------- S 路 split-K 扫描（--splitk-factor-sweep，只对 S6 有意义）----
        if (o.splitk_factor_sweep && step == 5) {
          const std::vector<bench::FactorPick> picks =
              bench::find_splitk_factor_sweep(md.m_tile, 5, mdl.n_ker, mdl.K, M,
                                              o.splitk_factors);
          for (const bench::FactorPick& fp : picks) {
            const bench::VariantOps* v = fp.v;
            if (v == nullptr) {
              skipped.push_back({shape.id, step, M, md.m_tile,
                                 "S=" + std::to_string(fp.factor) + " base" +
                                     std::to_string(fp.base) + " (" + fp.geom +
                                     ") 未实例化"});
              continue;
            }
            if (const char* bad = v->fn_supported(M, mdl.n_ker, mdl.K)) {
              skipped.push_back({shape.id, step, M, md.m_tile,
                                 "S=" + std::to_string(fp.factor) + " " +
                                     v->config_str + ": " + bad});
              continue;
            }
            std::array<bool, 3> want = proto_mask(o);
            if (factor_append) {
              const char* pn[3] = {"ISO", "B2B", "PDL"};
              CsvRow probe_row = bench::make_base_row(shape, mdl, md, v, step,
                                                      PointResult{}, plan, -1);
              bool all = true;
              for (int pi = 0; pi < 3; ++pi) {
                probe_row.protocol = pn[pi];
                if (!want[pi]) continue;
                if (done.has(true, probe_row)) { want[pi] = false; ++rows_resumed; }
                else all = false;
              }
              if (all) { ++points_resumed; continue; }
            }
            PointResult pr = bench::measure_point(
                g, cache, mdl, md, v, o, plan, /*with_correctness=*/true, want,
                [&](const char* proto, const TimingStats& st,
                    const PointResult& r) {
                  CsvRow row = bench::make_base_row(shape, mdl, md, v, step, r,
                                                    plan, g.clocks.read(false));
                  fill_row_stats(&row, proto, st);
                  std::ostringstream line;
                  line << row_main(row) << ',' << bench::sk_factor_of(v->spec);
                  factor_csv.write(line.str());
                  done.add(true, row);
                  ++rows_written;
                });
            // 门禁（CONTRACT §4.5 / 本次任务）：split-K 跑完 semaphore 必须
            // 自复位回 0。整块回读（含所有轮转槽），任何一格非 0 都算脏。
            int sem_dirty = -1;
            if (bench::sk_factor_of(v->spec) >= 2) {
              std::vector<int> sem_dbg(static_cast<size_t>(md.sem_ints) *
                                       md.sem_slots);
              CUDA_CHECK(cudaMemcpy(sem_dbg.data(), md.d_sem,
                                    sem_dbg.size() * sizeof(int),
                                    cudaMemcpyDeviceToHost));
              sem_dirty = 0;
              for (int sv : sem_dbg) sem_dirty += (sv != 0);
              if (sem_dirty != 0) {
                std::fprintf(stderr,
                             "[bench_decode] SEM-DIRTY %s M=%d S=%d: %d 个 semaphore "
                             "没自复位（sem_rotate=%d）\n",
                             shape.id.c_str(), M, fp.factor, sem_dirty,
                             md.sem_slots);
              }
            }
            if (!pr.skip_reason.empty()) {
              skipped.push_back({shape.id, step, M, md.m_tile,
                                 "S=" + std::to_string(fp.factor) + ": " +
                                     pr.skip_reason});
              ++points_failed;
            } else {
              ++points_done;
              if (sem_dirty > 0) ++points_failed;       // 脏 sem = 数据不可信
              std::printf(
                  "[bench_decode] SKF %-16s M=%3d S=%d base%d(%s) ctas=%d "
                  "rel_l2=%.2e pass=%s sem_dirty=%d ISO=%.2f B2B=%.2f PDL=%.2f us\n",
                  shape.id.c_str(), M, fp.factor, fp.base, fp.geom.c_str(),
                  pr.ctas, pr.corr.rel_l2,
                  pr.corr.ran ? (pr.corr.pass ? "Y" : "N") : "-", sem_dirty,
                  pr.iso.p50_us, pr.b2b.p50_us, pr.pdl.p50_us);
            }
          }
          continue;      // 扫描模式不再跑常规 canonical
        }

        // ---------------- canonical ----------------
        if (!o.tune_only) {
          const std::vector<const bench::VariantOps*> canon =
              bench::find_canonical_all(md.m_tile, step, mdl.n_ker, mdl.K, M);
          if (canon.empty()) {
            skipped.push_back({shape.id, step, M, md.m_tile,
                               "canonical variant 未实例化"});
            continue;
          }
          // M_TILE=128 会拿到两条（WG2/REG168 与 WG1/REG232），两条都跑都出 CSV 行
          for (const bench::VariantOps* v : canon) {
          if (const char* bad = v->fn_supported(M, mdl.n_ker, mdl.K)) {
            skipped.push_back({shape.id, step, M, md.m_tile,
                               std::string("Cfg::validate: ") + bad});
            continue;
          }
          // 断点续跑：三个协议都有数就整点跳过（连 warmup/prepack 都不做）
          std::array<bool, 3> want = proto_mask(o);
          if (!want[0] && !want[1] && !want[2]) continue;
          if (o.resume && append_mode) {
            const char* pn[3] = {"ISO", "B2B", "PDL"};
            CsvRow probe_row = bench::make_base_row(shape, mdl, md, v, step,
                                                    PointResult{}, plan, -1);
            for (int pi = 0; pi < 3; ++pi) {
              probe_row.protocol = pn[pi];
              if (done.has(false, probe_row)) { want[pi] = false; ++rows_resumed; }
            }
            if (!want[0] && !want[1] && !want[2]) { ++points_resumed; continue; }
          }
          // 增量落盘：每个协议测完立刻 append+flush（被 kill 也不丢已测数据）
          PointResult pr = bench::measure_point(
              g, cache, mdl, md, v, o, plan, /*with_correctness=*/true, want,
              [&](const char* proto, const TimingStats& st,
                  const PointResult& r) {
                CsvRow row = bench::make_base_row(shape, mdl, md, v, step, r,
                                                  plan, g.clocks.read(false));
                fill_row_stats(&row, proto, st);
                csv.write(row_main(row));
                detail.write(row_detail(row));
                done.add(false, row);
                ++rows_written;
              });
          if (!pr.skip_reason.empty()) {
            skipped.push_back({shape.id, step, M, md.m_tile, pr.skip_reason});
            ++points_failed;
            continue;
          }
          if (pr.corr.ran && !pr.corr.pass) ++correctness_fail;
          ++points_done;
          const double bytes = bench::modeled_bytes(mdl.n_log, mdl.K, M);
          const TimingStats& curve = (step >= 1) ? pr.pdl : pr.b2b;  // 文章口径
          std::printf(
              "[bench_decode] %-16s M=%3d S%d %-11s rel_l2=%.2e pass=%s "
              "ISO=%.2f B2B=%.2f PDL=%.2f us | curve(%s)=%.0f GB/s (%.1f%%) "
              "sets=%d ctas=%d smem=%dKiB clk=%dMHz\n",
              shape.id.c_str(), M, step, bench::kStepNames[step], pr.corr.rel_l2,
              pr.corr.ran ? (pr.corr.pass ? "Y" : "N") : "-", pr.iso.p50_us,
              pr.b2b.p50_us, pr.pdl.p50_us, (step >= 1) ? "PDL" : "B2B",
              std::isfinite(curve.p50_us) && curve.p50_us > 0
                  ? bytes / (curve.p50_us * 1e-6) / 1e9 : 0.0,
              std::isfinite(curve.p50_us) && curve.p50_us > 0
                  ? 100.0 * bytes / (curve.p50_us * 1e-6) / 1e9 / kSpecPeakGbps
                  : 0.0,
              pr.sets, pr.ctas, pr.smem / 1024, g.clocks.last_mhz);
          if (o.check_launcher && std::isfinite(pr.corr.rel_l2_launcher))
            std::printf("[bench_decode]   launch<Cfg> 交叉验证 rel_l2=%.2e\n",
                        pr.corr.rel_l2_launcher);
          }   // for canonical variants（M_TILE=128 有两条）
        }

        // ---------------- tune ----------------
        if (o.tune) {
          const std::vector<const bench::VariantOps*> grid =
              bench::find_tune(md.m_tile, step);
          if (grid.empty()) {
            skipped.push_back(
                {shape.id, step, M, md.m_tile,
                 "该 step 的 tune 网格没烘进这个二进制（Makefile: TUNE=1 "
                 "TUNE_STEPS=all，或直接用 bench_decode_tune）"});
            continue;
          }
          std::printf("[bench_decode] tune %s M=%d S%d: %zu configs\n",
                      shape.id.c_str(), M, step, grid.size());
          struct RankEntry { CsvRow row; double us; };
          std::vector<RankEntry> ranking;
          for (size_t i = 0; i < grid.size(); ++i) {
            const bench::VariantOps* v = grid[i];
            if (const char* bad = v->fn_supported(M, mdl.n_ker, mdl.K)) {
              skipped.push_back({shape.id, step, M, md.m_tile,
                                 std::string("tune ") + v->config_str + ": " + bad});
              continue;
            }
            std::array<bool, 3> want = proto_mask(o);
            if (!want[0] && !want[1] && !want[2]) continue;
            if (o.resume && tune_append) {
              const char* pn[3] = {"ISO", "B2B", "PDL"};
              CsvRow probe_row = bench::make_base_row(shape, mdl, md, v, step,
                                                      PointResult{}, plan, -1);
              probe_row.kind = "tune";
              bool all = true;
              for (int pi = 0; pi < 3; ++pi) {
                probe_row.protocol = pn[pi];
                if (done.has(true, probe_row)) { want[pi] = false; ++rows_resumed; }
                else all = false;
              }
              if (all) { ++points_resumed; continue; }
            }
            CsvRow last_row;
            // 前 8 个 config 顺带验 correctness（同一数值路径，抽样即可）
            PointResult pr = bench::measure_point(
                g, cache, mdl, md, v, o, plan, /*with_correctness=*/i < 8, want,
                [&](const char* proto, const TimingStats& st,
                    const PointResult& r) {
                  CsvRow row = bench::make_base_row(shape, mdl, md, v, step, r,
                                                    plan, g.clocks.read(false));
                  row.kind = "tune";
                  fill_row_stats(&row, proto, st);
                  tune_csv.write(row_main(row));
                  tune_detail.write(row_detail(row));
                  done.add(true, row);
                  ++rows_written;
                  last_row = row;
                });
            if (!pr.skip_reason.empty()) {
              skipped.push_back({shape.id, step, M, md.m_tile, pr.skip_reason});
              continue;
            }
            if (pr.corr.ran && !pr.corr.pass) {
              std::fprintf(stderr, "[bench_decode]   tune rel_l2=%.2e FAIL %s\n",
                           pr.corr.rel_l2, v->config_str.c_str());
              ++correctness_fail;
              continue;                        // 数值不对的 config 不参与排名
            }
            ++tune_points;
            const TimingStats& rank_st = (step >= 1) ? pr.pdl : pr.b2b;  // 文章口径
            if (std::isfinite(rank_st.p50_us)) {
              last_row.rank = -1;
              ranking.push_back({last_row, rank_st.p50_us});
            }
          }
          std::stable_sort(ranking.begin(), ranking.end(),
                           [](const RankEntry& a, const RankEntry& b) {
                             return a.us < b.us;
                           });
          for (size_t r = 0; r < ranking.size(); ++r) {
            const CsvRow& row = ranking[r].row;
            std::ostringstream line;
            line << csv_field(row.model) << ',' << row.N << ',' << row.K << ','
                 << row.M << ',' << row.m_tile << ',' << row.step << ','
                 << csv_field(row.step_name) << ',' << (r + 1) << ','
                 << (r == 0 ? 1 : 0) << ',' << ((step >= 1) ? "PDL" : "B2B")
                 << ',' << f4(row.p50) << ',' << f4(row.p30) << ',' << f3(row.bw)
                 << ',' << f6(row.pct) << ',' << row.clocks_mhz << ','
                 << fe6(row.rel_l2) << ',' << csv_field(row.pass) << ','
                 << csv_field(row.config) << ',' << row.smem_bytes << ','
                 << row.num_ctas << ',' << row.weight_sets;
            tune_best_csv.write(line.str());
          }
          if (!ranking.empty()) {
            const CsvRow& b = ranking[0].row;
            bests.push_back({shape.id, step, M, b.config, ranking[0].us, b.bw});
            std::printf("[bench_decode]   BEST %s M=%d S%d: %.2f us (%.0f GB/s) %s\n",
                        shape.id.c_str(), M, step, ranking[0].us, b.bw,
                        b.config.c_str());
          }
        }
      }

      cudaFree(md.d_act); cudaFree(md.d_ascale); cudaFree(md.d_adeq);
      cudaFree(md.d_out); cudaFree(md.d_ws); cudaFree(md.d_sem);
    }

    cache.clear();
    cudaFree(mdl.d_weight_raw); cudaFree(mdl.d_wscale); cudaFree(mdl.d_wdeq);
  }

  const int clocks_end = g.clocks.read(true);
  const double elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
          .count();
  csv.close(); detail.close();
  if (o.tune) { tune_csv.close(); tune_detail.close(); tune_best_csv.close(); }
  if (o.splitk_ablation) ablation_csv.close();
  if (o.pdl_placement) placement_csv.close();
  if (o.splitk_factor_sweep) factor_csv.close();

  // ------------------------------------------------------------- summary ----
  {
    std::ofstream j(json_path);
    if (!j) {
      std::fprintf(stderr, "[bench_decode] 写不了 %s\n", json_path.c_str());
      return 2;
    }
    const char* mechanism =
        bench::registry().empty() ? "none" : bench::registry()[0].mechanism;
    j << "{\n";
    j << "  \"harness\": \"bench/bench_decode.cu\",\n";
    j << "  \"timestamp_utc\": \"" << bench::iso_now() << "\",\n";
    j << "  \"stamp\": \"" << ts << "\",\n";
    j << "  \"kernel_backend\": \""
#if defined(BENCH_USE_STUB)
      << "stub (bench/stub_kernel.h) - 仅流程自测，性能数字无意义"
#else
      << "kernels/decode_gemm.cuh"
#endif
      << "\",\n";
    j << "  \"launch_mechanism\": \"" << mechanism << "\",\n";
    j << "  \"launch_mechanism_note\": \""
      << (std::string(mechanism) == "kernel_ex"
              ? "harness 自己 cudaLaunchKernelEx；PSS attr 只在 PDL 协议打开，三协议可比"
              : "退回 kernel launcher：PSS attr 不受 harness 控制，B2B 与 PDL 等价，不能讲 PDL 收益")
      << "\",\n";
    j << "  \"force_api_launch\": " << (o.force_api_launch ? "true" : "false") << ",\n";
    j << "  \"tune_steps_mask\": " << BENCH_TUNE_STEPS_MASK << ",\n";
    j << "  \"registered_variants\": " << n_variants
      << ", \"registered_canonical\": " << n_canon
      << ", \"registered_tune\": " << n_tune << ",\n";
    j << "  \"gpu\": {\n";
    j << "    \"name\": \"" << env.device_name << "\",\n";
    j << "    \"physical_index\": \"" << env.smi_index << "\",\n";
    j << "    \"gpurun_gpu\": \"" << env.gpurun_gpu << "\",\n";
    j << "    \"gpurun_target\": \"" << env.gpurun_target << "\",\n";
    j << "    \"gpurun_wrapper\": \"tools/gpurun_dg.sh v2\",\n";
    j << "    \"pool\": \"" << env.pool << "\",\n";
    j << "    \"cuda_device_index\": " << env.device_index << ",\n";
    j << "    \"pci_bus_id\": \"" << env.pci_bus_id << "\",\n";
    j << "    \"sm_count\": " << env.sm_count << ",\n";
    j << "    \"memory_total_bytes\": " << env.mem_total << ",\n";
    j << "    \"CUDA_VISIBLE_DEVICES\": \"" << env.cuda_visible_devices << "\"\n";
    j << "  },\n";
    j << "  \"clocks\": {\n";
    j << "    \"sm_start_mhz\": " << clocks_start << ",\n";
    j << "    \"sm_end_mhz\": " << clocks_end << ",\n";
    j << "    \"sm_current_mhz\": " << env.clock_cur_mhz << ",\n";
    j << "    \"sm_max_khz\": " << env.clock_max_khz << ",\n";
    j << "    \"mem_current_mhz\": " << env.mem_clock_cur_mhz << ",\n";
    j << "    \"utilization_gpu_pct\": " << env.util_pct << ",\n";
    j << "    \"memory_used_mib\": " << env.mem_used_mib << ",\n";
    j << "    \"locked_by_gpurun_mhz\": 1830,\n";
    j << "    \"per_row_column\": \"clocks_sm_mhz\"\n";
    j << "  },\n";
    j << "  \"exclusivity\": {\n";
    j << "    \"verdict\": \"" << env.exclusive_verdict << "\",\n";
    j << "    \"policy\": \"CONTRACT §0 v2(2026-09-13)：BORROW 池 {2,3} 优先"
         "(keeper 预留但物理空闲，用户授权借用；util<=5 且除 keeper 白名单外无 "
         "compute pid；运行中 2s 看门狗，外来 compute 落卡即 abort 让路 exit 75)，"
         "CLEAN 池 {0,1} 兜底(util<=5/mem<=2000MiB/完全无 compute pid)；"
         "两池都抢 _lockbench_20260913/locks/gpu<N>.lock 跨项目 flock + 锁频 1830\",\n";
    j << "    \"shell_check\": \"" << env.shell_note << "\",\n";
    j << "    \"keeper_apps\": [";
    for (size_t i = 0; i < env.keeper_apps.size(); ++i)
      j << (i ? ", " : "") << "\"" << env.keeper_apps[i] << "\"";
    j << "],\n";
    j << "    \"foreign_compute_apps\": [";
    for (size_t i = 0; i < env.other_apps.size(); ++i)
      j << (i ? ", " : "") << "\"" << env.other_apps[i] << "\"";
    j << "]\n  },\n";
    j << "  \"protocol\": {\n";
    j << "    \"ISO\": \"单发 cudaEvent + eventSync，每次 " << o.l2_flush_mb
      << "MB L2 flush + 旋转冷权重集；p50/p30 over reps*iso_samples 个样本\",\n";
    j << "    \"B2B\": \"同 stream back-to-back batch，一个 event 窗口，"
         "per-launch=total/batch；p50/p30 over reps 个 batch 均值\",\n";
    j << "    \"PDL\": \"B2B + cudaLaunchAttributeProgrammaticStreamSerialization\",\n";
    j << "    \"article_curve\": \"step>=1 用 PDL，step0 用 B2B\",\n";
    j << "    \"warmup_launches\": " << kWarmupLaunches << ",\n";
    j << "    \"l2_flush_bytes\": " << flush_bytes << ",\n";
    j << "    \"reps\": " << o.reps << ",\n";
    j << "    \"big_k_auto_scale\": \"K>=" << kBigKThreshold
      << " -> iso_samples 20 / batch 50（orchestrator 补充）\",\n";
    j << "    \"cold_weight_working_set_bytes\": " << kWorkingSetBytes << ",\n";
    j << "    \"cold_weight_min_sets\": " << kMinWeightSets << ",\n";
    j << "    \"cold_weight_sets_formula\": \"max(5, ceil(512MiB / weight_bytes))\"\n";
    j << "  },\n";
    j << "  \"correctness\": {\n";
    j << "    \"reference\": \"dequantize->fp32 cublasLt SGEMM (CUBLAS_COMPUTE_32F)\",\n";
    j << "    \"reference_cache\": \"每 (model,M) 一次（参考与 step 无关）\",\n";
    j << "    \"threshold_rel_l2\": " << std::setprecision(3) << kRelL2Threshold << ",\n";
    j << "    \"failures\": " << correctness_fail << "\n";
    j << "  },\n";
    j << "  \"bandwidth_convention\": {\n";
    j << "    \"numerator_bytes\": \"N*K + M*K + 2*M*N\",\n";
    j << "    \"spec_peak_gbps\": " << std::setprecision(6) << kSpecPeakGbps << ",\n";
    j << "    \"pct_of_spec_peak\": \"bandwidth_gbps / 4000（比值，不是百分数）\"\n";
    j << "  },\n";
    j << "  \"csv_schema\": {\n";
    j << "    \"columns\": \"" << kCsvHeader << "\",\n";
    j << "    \"contract_columns\": 16,\n";
    j << "    \"config_separator\": \";\",\n";
    j << "    \"protocols\": [\"ISO\", \"B2B\", \"PDL\"],\n";
    j << "    \"detail_sidecar_columns\": \"" << kDetailHeader << "\"\n";
    j << "  },\n";
    j << "  \"cli\": {\"models\": [";
    for (size_t i = 0; i < models.size(); ++i)
      j << (i ? ", " : "") << "\"" << models[i].id << "\"";
    j << "], \"steps\": [";
    for (size_t i = 0; i < o.steps.size(); ++i) j << (i ? ", " : "") << o.steps[i];
    j << "], \"ms\": [";
    for (size_t i = 0; i < o.ms.size(); ++i) j << (i ? ", " : "") << o.ms[i];
    j << "], \"tune\": " << (o.tune ? "true" : "false")
      << ", \"tune_only\": " << (o.tune_only ? "true" : "false")
      << ", \"demo\": " << (o.demo ? "true" : "false")
      << ", \"seed\": " << o.seed
      << ", \"weight_cache_gb\": " << o.weight_cache_gb << "},\n";
    j << "  \"models_json\": {\"path\": \"" << models_path
      << "\", \"fallback_used\": " << (fallback ? "true" : "false")
      << ", \"note\": \"" << model_note << "\", \"shapes\": [";
    for (size_t i = 0; i < all_models.size(); ++i) {
      const ModelShape& m = all_models[i];
      j << (i ? ", " : "") << "{\"id\": \"" << m.id << "\", \"N\": " << m.N
        << ", \"K\": " << m.K << ", \"layer\": \"" << m.layer << "\", \"bm\": "
        << bench::choose_bm(m.N) << "}";
    }
    j << "]},\n";
    j << "  \"outputs\": {\"csv\": \"" << csv_path << "\", \"detail_csv\": \""
      << detail_path << "\", \"summary\": \"" << json_path << "\", \"tune_csv\": \""
      << tune_path << "\", \"tune_detail_csv\": \"" << tune_detail_path
      << "\", \"ablation_csv\": \"" << ablation_path
      << "\", \"factor_csv\": \"" << factor_path
      << "\", \"tune_best_csv\": \"" << tune_best_path
      << "\", \"csv_rows\": " << csv.rows()
      << ", \"tune_csv_rows\": " << tune_csv.rows()
      << ", \"appended\": " << (append_mode ? "true" : "false") << "},\n";
    j << "  \"resume\": {\"policy\": \"启动时先 purge 不完整组（组="
         "(model,N,K,M,m_tile,step,config) 必须有 ISO/B2B/PDL 三行有效 latency），"
         "再对 CSV 上 flock 独占，拿不到锁就新开一份\",\n";
    j << "    \"requested\": " << (o.resume ? "true" : "false")
      << ", \"append_mode\": " << (append_mode ? "true" : "false")
      << ", \"source\": \"" << done.source << "\", \"note\": \""
      << resume_note << "\", \"rows_skipped\": " << rows_resumed
      << ", \"points_skipped\": " << points_resumed
      << ", \"rows_written_this_run\": " << rows_written << "},\n";
    j << "  \"counts\": {\"points_done\": " << points_done
      << ", \"points_failed\": " << points_failed << ", \"tune_points\": "
      << tune_points << ", \"points_resumed\": " << points_resumed
      << ", \"rows_written\": " << rows_written
      << ", \"skipped\": " << skipped.size() << "},\n";
    j << "  \"skipped\": [";
    for (size_t i = 0; i < skipped.size(); ++i) {
      const Skipped& s = skipped[i];
      j << (i ? ", " : "") << "{\"model\": \"" << s.model << "\", \"step\": "
        << s.step << ", \"M\": " << s.M << ", \"m_tile\": " << s.m_tile
        << ", \"reason\": \"" << s.reason << "\"}";
      if (i >= 199) { j << ", {\"truncated\": true}"; break; }
    }
    j << "],\n";
    j << "  \"tune_best\": [";
    for (size_t i = 0; i < bests.size(); ++i) {
      const BestTune& b = bests[i];
      j << (i ? ", " : "") << "{\"model\": \"" << b.model << "\", \"step\": "
        << b.step << ", \"M\": " << b.M << ", \"latency_us\": "
        << std::setprecision(6) << b.us << ", \"bandwidth_gbps\": " << b.gbps
        << ", \"config\": \"" << b.config << "\"}";
    }
    j << "],\n";
    j << "  \"elapsed_seconds\": " << std::setprecision(6) << elapsed_s << "\n";
    j << "}\n";
  }

  std::printf("[bench_decode] done: %d points, %d failed, %d tune points, "
              "%zu skipped, %d rows written, %d rows resumed, %.1fs, "
              "clocks %d->%d MHz\n",
              points_done, points_failed, tune_points, skipped.size(),
              rows_written, rows_resumed, elapsed_s, clocks_start, clocks_end);
  std::printf("[bench_decode] csv    -> %s (%d rows)\n", csv_path.c_str(), csv.rows());
  std::printf("[bench_decode] detail -> %s (%d rows)\n", detail_path.c_str(),
              detail.rows());
  std::printf("[bench_decode] json   -> %s\n", json_path.c_str());
  if (o.splitk_ablation)
    std::printf("[bench_decode] ablation -> %s (%d rows)\n",
                ablation_path.c_str(), ablation_csv.rows());
  if (o.splitk_factor_sweep)
    std::printf("[bench_decode] splitk-factor -> %s (%d rows)\n",
                factor_path.c_str(), factor_csv.rows());
  if (o.tune) {
    std::printf("[bench_decode] tune   -> %s (%d rows)\n", tune_path.c_str(),
                tune_csv.rows());
    std::printf("[bench_decode] tune detail -> %s\n", tune_detail_path.c_str());
    std::printf("[bench_decode] tune best  -> %s (%d rows)\n",
                tune_best_path.c_str(), tune_best_csv.rows());
  }
  if (!skipped.empty()) {
    std::fprintf(stderr, "[bench_decode] skipped %zu 个 (model,step,M)：\n",
                 skipped.size());
    int shown = 0;
    for (const auto& s : skipped) {
      if (shown++ >= 15) {
        std::fprintf(stderr, "  ... 其余 %zu 条见 JSON\n", skipped.size() - 15);
        break;
      }
      std::fprintf(stderr, "  %s step%d M=%d m_tile=%d: %s\n", s.model.c_str(),
                   s.step, s.M, s.m_tile, s.reason.c_str());
    }
  }
  CUDA_CHECK(cudaStreamDestroy(g.stream));
  CUDA_CHECK(cudaEventDestroy(g.ev_start));
  CUDA_CHECK(cudaEventDestroy(g.ev_end));
  cudaFree(g.flush_buf);
  return (correctness_fail == 0 && points_failed == 0 && points_done > 0) ? 0 : 1;
}
