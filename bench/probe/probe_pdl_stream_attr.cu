// Probe: does cudaStreamSetAttribute(cudaLaunchAttributeProgrammaticStreamSerialization)
// enable PDL overlap for kernels launched with plain <<<>>> afterwards?
// Harness needs this so the PDL protocol can be toggled WITHOUT knowing the kernel symbol.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::printf("CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__device__ __forceinline__ uint64_t gt() {
  uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__global__ void spin_kernel(uint64_t before_ns, uint64_t after_ns, uint64_t* stamps, int idx) {
  const uint64_t t0 = gt();
  if (threadIdx.x == 0 && blockIdx.x == 0) stamps[2 * idx] = t0;
  while (gt() - t0 < before_ns) {}
  asm volatile("griddepcontrol.launch_dependents;");
  while (gt() - t0 < before_ns + after_ns) {}
  if (threadIdx.x == 0 && blockIdx.x == 0) stamps[2 * idx + 1] = gt();
}

int main() {
  CK(cudaSetDevice(0));
  cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, 0));
  std::printf("device=%s sm_%d%d\n", p.name, p.major, p.minor);

  const int kBatch = 20;
  const uint64_t before_ns = 20000, after_ns = 80000;
  uint64_t* stamps = nullptr;
  CK(cudaMalloc(&stamps, sizeof(uint64_t) * 2 * kBatch));
  CK(cudaMemset(stamps, 0, sizeof(uint64_t) * 2 * kBatch));

  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

  for (int mode = 0; mode < 3; ++mode) {
    // mode 0: no attribute (plain B2B). mode 1: stream attribute. mode 2: cudaLaunchKernelEx attribute.
    cudaLaunchAttributeValue val{}; val.programmaticStreamSerializationAllowed = 1;
    cudaError_t se = cudaSuccess;
    if (mode == 1) {
      se = cudaStreamSetAttribute(s, cudaLaunchAttributeProgrammaticStreamSerialization, &val);
      cudaLaunchAttributeValue rd{};
      cudaError_t ge = cudaStreamGetAttribute(s, cudaLaunchAttributeProgrammaticStreamSerialization, &rd);
      std::printf("mode1 streamSetAttribute rc=%s ; readback rc=%s val=%d\n",
                  cudaGetErrorString(se), cudaGetErrorString(ge),
                  (int)rd.programmaticStreamSerializationAllowed);
    }
    CK(cudaMemsetAsync(stamps, 0, sizeof(uint64_t) * 2 * kBatch, s));
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < kBatch; ++i) {
      if (mode == 2) {
        cudaLaunchConfig_t cfg{}; cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.gridDim = dim3(128,1,1); cfg.blockDim = dim3(128,1,1); cfg.dynamicSmemBytes = 0;
        cfg.stream = s; cfg.attrs = attrs; cfg.numAttrs = 1;
        CK(cudaLaunchKernelEx(&cfg, spin_kernel, before_ns, after_ns, stamps, i));
      } else {
        spin_kernel<<<128, 128, 0, s>>>(before_ns, after_ns, stamps, i);
      }
    }
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    if (mode == 1) {  // clear attribute again
      cudaLaunchAttributeValue zero{}; zero.programmaticStreamSerializationAllowed = 0;
      cudaError_t ce = cudaStreamSetAttribute(s, cudaLaunchAttributeProgrammaticStreamSerialization, &zero);
      std::printf("mode1 clear rc=%s\n", cudaGetErrorString(ce));
    }
    std::vector<uint64_t> h(2 * kBatch);
    CK(cudaMemcpy(h.data(), stamps, sizeof(uint64_t) * 2 * kBatch, cudaMemcpyDeviceToHost));
    // overlap evidence: start of launch i+1 vs end of launch i
    int overlaps = 0; uint64_t max_overlap_ns = 0;
    for (int i = 0; i + 1 < kBatch; ++i) {
      if (h[2*(i+1)] < h[2*i+1]) { overlaps++; max_overlap_ns = std::max(max_overlap_ns, h[2*i+1] - h[2*(i+1)]); }
    }
    std::printf("mode=%d total_us=%.1f per_launch_us=%.2f expected_serial_us=%.1f expected_pdl_us=%.1f overlaps=%d/%d max_overlap_ns=%llu\n",
                mode, ms * 1000.0f, ms * 1000.0f / kBatch,
                kBatch * (before_ns + after_ns) / 1000.0, (kBatch * before_ns + after_ns + before_ns) / 1000.0,
                overlaps, kBatch - 1, (unsigned long long)max_overlap_ns);
  }
  return 0;
}
