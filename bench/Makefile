# SPDX-License-Identifier: MIT
#
# bench/Makefile —— decode_gemm benchmark harness。
#
#   make                 真 kernel（kernels/decode_gemm.cuh），canonical + 默认 tune 网格
#   make -j8             6 个 TU 并行编（bench_decode.cu + bench_m{8,16,32,64,128}.cu）
#   make TUNE=0          只烘 canonical 七步（最快，跑主 sweep 用这个就够）
#   make TUNE_STEPS=all  7 个 step 全烘 tune 网格（CONTRACT §6 全量，慢：约 20 分钟）
#   make TUNE_STEPS=6    只烘 S6 的 tune 网格
#   make stub            bench_decode_stub：走 bench/stub_kernel.h（仅流程自测）
#   make tune-bin        另出一个 bench_decode_tune（TUNE_STEPS=all，不动主二进制）
#   make tune-bin TUNE_SKF=1 TUNE_SHAPE_N=2048 TUNE_SHAPE_K=4096 TUNE_GRID=s4 TUNE_STEPS=6
#                        S 路 split-K：给 step6 的 tune 网格再加 S∈{2,4,8} 一维。
#                        形状必须由宏给（网格本身形状无关），闸门见 configs.cuh
#                        sk_factor_expand()：splitk_worth_it(N,BM) 或 N<=4096 才展开；
#                        整除性 K % (128*SUBS*WGsPerTile*S)==0 也在编译期筛掉。
#                        不给 TUNE_SHAPE_N 时索引空间与实例化数和以前完全一样。
#   make probe-cfg       编译期校验 configs.cuh 的 feasible() 不比 kernel 的
#                        static_assert 松（几秒，不跑 GPU）
#   make list            打印模型表 + 已实例化 variant 数
#   make clean
#
# 编译命令严格按 CONTRACT §0 的模板：
#   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -DNDEBUG \
#        -I$P/kernels -I$CUTLASS <src> -lcuda -lcublasLt -o <out>
# （额外的 -I. 是为了找到 bench/ 自己的头；-lcublasLt 是 CONTRACT §6 的
#   cublasLt SGEMM 参考所需。）

P       := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
W       := $(abspath $(P)/../../../../..)
CUTLASS ?= $(W)/weave_v2/deepgemm/third-party/cutlass/include
NVCC    ?= /usr/local/cuda/bin/nvcc

ARCH    := -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -DNDEBUG
INC     := -I. -I$(P)/kernels -I$(CUTLASS)
LIBS    := -lcuda -lcublasLt

BIN     ?= bench_decode
OBJDIR  ?= build
TUNE    ?= 1
TUNE_STEPS ?= 0,4,6
# TUNE_GRID=contract -> CONTRACT §6 全网格（BM 5 x WG/TILES/REG 13 x STG 3 x SUBS 2）
# TUNE_GRID=s4       -> orchestrator 2026-09-13 修订版：只扫 S4+ 旋钮，SUBS=2/PREPACK=1
#                       固定，BM{32,48,56,64} x (WG,TILES,REG) 13 x STG{2,3,4} = 156/M_TILE
TUNE_GRID ?= contract
# S 路 split-K 的 tune 维（splitk-S agent）：默认关，实例化数不变
TUNE_SKF ?= 0
TUNE_SHAPE_N ?= 0
TUNE_SHAPE_K ?= 0
EXTRA_DEFS ?=
ifeq ($(TUNE_SKF),1)
  EXTRA_DEFS += -DBENCH_TUNE_SHAPE_N=$(TUNE_SHAPE_N) -DBENCH_TUNE_SHAPE_K=$(TUNE_SHAPE_K)
endif
ifeq ($(TUNE_GRID),s4)
  EXTRA_DEFS += -DBENCH_TUNE_GRID_S4=1
endif

ifeq ($(TUNE_STEPS),all)
  TUNE_MASK := 127
else ifeq ($(TUNE_STEPS),none)
  TUNE_MASK := 0
else
  TUNE_MASK := $(shell python3 -c "print(sum(1<<int(x) for x in '$(TUNE_STEPS)'.replace(' ','').split(',') if x.isdigit()))")
endif

DEFS := $(ARCH) $(INC) -DBENCH_TUNE=$(TUNE) -DBENCH_TUNE_STEPS_MASK=$(TUNE_MASK) $(EXTRA_DEFS)

SRCS := bench_decode.cu bench_m8.cu bench_m16.cu bench_m32.cu bench_m64.cu bench_m128.cu
OBJS := $(addprefix $(OBJDIR)/,$(SRCS:.cu=.o))
DEPS := $(OBJS:.o=.d)

.PHONY: all stub tune-bin probe-cfg probe-pdl list clean help
.SUFFIXES:

all: $(BIN)
	@echo "[bench] built $(BIN)  (TUNE=$(TUNE) TUNE_STEPS=$(TUNE_STEPS) mask=$(TUNE_MASK))"
	@echo "[bench] 跑 GPU 必须经过 wrapper:  ../tools/gpurun_dg.sh <target> ./$(BIN) --demo"

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: %.cu | $(OBJDIR)
	$(NVCC) $(DEFS) -MMD -MP -c $< -o $@

$(BIN): $(OBJS)
	$(NVCC) $(ARCH) $(OBJS) $(LIBS) -o $(BIN)

# ---- stub：同一个源码树，只换 kernel 后端（bench/stub_kernel.h）----
stub:
	@$(MAKE) --no-print-directory BIN=bench_decode_stub OBJDIR=build_stub \
	  EXTRA_DEFS="-DBENCH_USE_STUB" all

# ---- tune 专用二进制（不动主二进制）。默认 s4 网格（orchestrator 修订版）----
tune-bin:
	@$(MAKE) --no-print-directory BIN=bench_decode_tune OBJDIR=build_tune \
	  TUNE=1 TUNE_GRID=$(TUNE_GRID) all

# ---- 编译期一致性闸门：configs.cuh 的 feasible() vs kernel 的 static_assert ----
probe-cfg:
	$(NVCC) $(DEFS) probe/probe_cfg_asserts.cu $(LIBS) -o $(OBJDIR)/probe_cfg_asserts
	./$(OBJDIR)/probe_cfg_asserts

probe-pdl:
	$(NVCC) $(ARCH) probe/probe_pdl_stream_attr.cu -lcuda -o $(OBJDIR)/probe_pdl
	@echo "[bench] 跑它请走 wrapper: ../tools/gpurun_dg.sh probe_pdl ./$(OBJDIR)/probe_pdl"

list: $(BIN)
	./$(BIN) --list

clean:
	rm -rf build build_stub build_tune bench_decode bench_decode_stub bench_decode_tune

help:
	@sed -n '2,30p' Makefile

-include $(DEPS)
