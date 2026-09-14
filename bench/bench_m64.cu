// SPDX-License-Identifier: MIT
//
// bench/bench_m64.cu —— M_TILE=64 的实例化 TU。
// 实例化内容：该 M_TILE 下 7 个 step 的 canonical config（每个 step 遍历
// 5 个候选 BM，运行时按 N 选）+ --tune 网格（BENCH_TUNE / BENCH_TUNE_STEPS_MASK
// 控制范围）。主程序通过 bench::register_m64() 拿到注册表。
// 拆分理由：CONTRACT §6「编译拆分」——避免单 TU 模板实例化爆炸，make -j 可并行。

#include "variant_impl.cuh"

namespace bench {
void register_m64() { register_m_tile<64>(); }
}  // namespace bench
