/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include "tc/aten/utils.h"
#include "tc/core/cpu/cpu_tc_executor.h"
#include "tc/core/execution_engine.h"
#include "tc/core/mapping_options.h"
#include "tc/core/polyhedral/codegen_llvm.h"
#include "tc/core/polyhedral/llvm_jit.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/scope_guard.h"

#include "test_harness_aten.h"

using namespace std;

using namespace tc;
using namespace tc::polyhedral;
using namespace tc::polyhedral::detail;

TEST(LLVMCodegen, Basic) {
  string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
  C(i, j) = A(i, j) + B(i, j)
}
)TC";
  auto N = 40;
  auto M = 24;

  auto ctx = isl::with_exceptions::globalIslCtx();
  auto scop = polyhedral::Scop::makeScop(ctx, tc);
  auto context = scop->makeContext(
      std::unordered_map<std::string, int>{{"N", N}, {"M", M}});
  scop = Scop::makeSpecializedScop(*scop, context);
  Jit jit;
  jit.codegenScop("kernel_anon", *scop);
  auto fptr =
      (void (*)(float*, float*, float*))jit.getSymbolAddress("kernel_anon");

  at::Tensor A = at::CPU(at::kFloat).rand({N, M});
  at::Tensor B = at::CPU(at::kFloat).rand({N, M});
  at::Tensor C = at::CPU(at::kFloat).rand({N, M});
  at::Tensor Cc = A + B;
  fptr(A.data<float>(), B.data<float>(), C.data<float>());

  checkRtol(Cc - C, {A, B}, N * M);
}

TEST(LLVMCodegen, BasicParallel) {
  string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
  C(i, j) = A(i, j) + B(i, j)
}
)TC";
  auto N = 40;
  auto M = 24;

  auto ctx = isl::with_exceptions::globalIslCtx();
  auto scop = polyhedral::Scop::makeScop(ctx, tc);
  auto context = scop->makeContext(
      std::unordered_map<std::string, int>{{"N", N}, {"M", M}});
  scop = Scop::makeSpecializedScop(*scop, context);
  scop =
      Scop::makeScheduled(*scop, SchedulerOptionsView(SchedulerOptionsProto()));
  Jit jit;
  auto mod = jit.codegenScop("kernel_anon", *scop);
  auto correct_llvm = R"LLVM(
; Function Attrs: nounwind
define void @kernel_anon([24 x float]* noalias nocapture nonnull readonly %A, [24 x float]* noalias nocapture nonnull readonly %B, [24 x float]* noalias nocapture nonnull %C) local_unnamed_addr #0 {
entry:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #0
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %__cilkrts_enter_frame_1.exit

slowpath.i:                                       ; preds = %entry
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #0
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %entry, %slowpath.i
  %.sink = phi i32 [ 16777344, %slowpath.i ], [ 16777216, %entry ]
  %3 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %entry ]
  %4 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i32*
  store volatile i32 %.sink, i32* %4, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %3, i64 0, i32 9
  %6 = load volatile %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %5, align 8
  %7 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store volatile %struct.__cilkrts_stack_frame* %6, %struct.__cilkrts_stack_frame** %7, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store volatile %struct.__cilkrts_worker* %3, %struct.__cilkrts_worker** %8, align 8
  store volatile %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %5, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  br label %loop_body

loop_body:                                        ; preds = %loop_latch, %__cilkrts_enter_frame_1.exit
  %c09 = phi i64 [ 0, %__cilkrts_enter_frame_1.exit ], [ %23, %loop_latch ]
  %10 = bitcast [5 x i8*]* %9 to i8*
  %11 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr = getelementptr i8, i8* %11, i64 72
  %12 = bitcast i8* %sunkaddr to i32*
  %13 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr16 = getelementptr i8, i8* %13, i64 76
  %14 = bitcast i8* %sunkaddr16 to i16*
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* %12, i16* %14) #0
  %15 = call i8* @llvm.frameaddress(i32 0)
  %16 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr17 = getelementptr i8, i8* %16, i64 32
  %17 = bitcast i8* %sunkaddr17 to i8**
  store volatile i8* %15, i8** %17, align 8
  %18 = call i8* @llvm.stacksave()
  %19 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr18 = getelementptr i8, i8* %19, i64 48
  %20 = bitcast i8* %sunkaddr18 to i8**
  store volatile i8* %18, i8** %20, align 8
  %21 = call i32 @llvm.eh.sjlj.setjmp(i8* %10) #3
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %loop_body.split, label %loop_latch

loop_body.split:                                  ; preds = %loop_body
  call fastcc void @kernel_anon_loop_body2.cilk([24 x float]* %C, i64 %c09, [24 x float]* %B, [24 x float]* %A)
  br label %loop_latch

loop_latch:                                       ; preds = %loop_body.split, %loop_body
  %23 = add nuw nsw i64 %c09, 1
  %exitcond = icmp eq i64 %23, 40
  br i1 %exitcond, label %loop_exit, label %loop_body

loop_exit:                                        ; preds = %loop_latch
  %24 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i32*
  %25 = load volatile i32, i32* %24, align 8
  %26 = and i32 %25, 2
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %__cilk_sync.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %loop_exit
  %28 = bitcast [5 x i8*]* %9 to i8*
  %29 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr19 = getelementptr i8, i8* %29, i64 16
  %30 = bitcast i8* %sunkaddr19 to %struct.__cilkrts_worker**
  %31 = load volatile %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %30, align 8
  %32 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr20 = getelementptr i8, i8* %32, i64 72
  %33 = bitcast i8* %sunkaddr20 to i32*
  %34 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr21 = getelementptr i8, i8* %34, i64 76
  %35 = bitcast i8* %sunkaddr21 to i16*
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %33, i16* nonnull %35) #0
  %36 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr22 = getelementptr i8, i8* %36, i64 32
  %37 = bitcast i8* %sunkaddr22 to i8**
  store volatile i8* %15, i8** %37, align 8
  %38 = call i8* @llvm.stacksave()
  %39 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr23 = getelementptr i8, i8* %39, i64 48
  %40 = bitcast i8* %sunkaddr23 to i8**
  store volatile i8* %38, i8** %40, align 8
  %41 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %28) #3
  %42 = icmp eq i32 %41, 0
  br i1 %42, label %cilk.sync.runtimecall.i, label %cilk.sync.excepting.i

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #0
  br label %__cilk_sync.exit

cilk.sync.excepting.i:                            ; preds = %cilk.sync.savestate.i
  %43 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i32*
  %44 = load volatile i32, i32* %43, align 8
  %45 = and i32 %44, 16
  %46 = icmp eq i32 %45, 0
  br i1 %46, label %__cilk_sync.exit, label %cilk.sync.rethrow.i

cilk.sync.rethrow.i:                              ; preds = %cilk.sync.excepting.i
  call void @__cilkrts_rethrow(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #4
  unreachable

__cilk_sync.exit:                                 ; preds = %loop_exit, %cilk.sync.runtimecall.i, %cilk.sync.excepting.i
  %47 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i32*
  %48 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr24 = getelementptr i8, i8* %48, i64 16
  %49 = bitcast i8* %sunkaddr24 to %struct.__cilkrts_worker**
  %50 = load volatile %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %49, align 8
  %51 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %50, i64 0, i32 12, i32 0
  %52 = load i64, i64* %51, align 8
  %53 = add i64 %52, 1
  store i64 %53, i64* %51, align 8
  %54 = load volatile %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %49, align 8
  %55 = bitcast %struct.__cilkrts_stack_frame* %__cilkrts_sf to i8*
  %sunkaddr25 = getelementptr i8, i8* %55, i64 8
  %56 = bitcast i8* %sunkaddr25 to %struct.__cilkrts_stack_frame**
  %57 = load volatile %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %56, align 8
  %58 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %54, i64 0, i32 9
  store volatile %struct.__cilkrts_stack_frame* %57, %struct.__cilkrts_stack_frame** %58, align 8
  store volatile %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %56, align 8
  %59 = load volatile i32, i32* %47, align 8
  %60 = icmp eq i32 %59, 16777216
  br i1 %60, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #0
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync.exit, %body.i
  ret void
}
)LLVM";
  EXPECT_EQ(correct_llvm, toString(mod->getFunction("kernel_anon")));
  auto fptr =
      (void (*)(float*, float*, float*))jit.getSymbolAddress("kernel_anon");

  at::Tensor A = at::CPU(at::kFloat).rand({N, M});
  at::Tensor B = at::CPU(at::kFloat).rand({N, M});
  at::Tensor C = at::CPU(at::kFloat).rand({N, M});
  at::Tensor Cc = A + B;
  fptr(A.data<float>(), B.data<float>(), C.data<float>());

  checkRtol(Cc - C, {A, B}, N * M);
}

TEST(LLVMCodegen, DISABLED_BasicExecutionEngine) {
  string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
  C(i, j) = A(i, j) + B(i, j)
}
)TC";

  auto N = 40;
  auto M = 24;

  at::Tensor A = at::CPU(at::kFloat).rand({N, M});
  at::Tensor B = at::CPU(at::kFloat).rand({N, M});
  at::Tensor C = at::CPU(at::kFloat).rand({N, M});

  ExecutionEngine<CpuTcExecutor> engine;
  engine.define(tc);
  auto options = tc::MappingOptions::makeNaiveMappingOptions();
  auto inputDLTensorsPair = toConstDlpackTensors({A, B});
  ScopeGuard g([&]() { deleteDlmTensors(inputDLTensorsPair.second); });
  engine.compile(
      "fun", inputDLTensorsPair.first, options.toProtobufSerializedString());
}

TEST(LLVMCodegen, MultiStmt) {
  string tc = R"TC(
 def fun(float(N, M, K, L) A, float(N, M) B, float(N, M) C, float(N, M) D)
 -> (O1, O2, O3)
 {
   O1(i, j) +=! A(i, j, rk, rl) * B(i, j)
   O2(i, j) = C(i, j) * D(i, j)
   O3(i, j) = O1(i, j) + O2(i, j)
 }
 )TC";

  auto N = 40;
  auto M = 24;
  auto K = 21;
  auto L = 33;

  auto ctx = isl::with_exceptions::globalIslCtx();
  auto scop = polyhedral::Scop::makeScop(ctx, tc);
  auto context = scop->makeContext(std::unordered_map<std::string, int>{
      {"N", N}, {"M", M}, {"K", K}, {"L", L}});
  scop = Scop::makeSpecializedScop(*scop, context);

  at::Tensor A = at::CPU(at::kFloat).rand({N, M, K, L});
  at::Tensor B = at::CPU(at::kFloat).rand({N, M});
  at::Tensor C = at::CPU(at::kFloat).rand({N, M});
  at::Tensor D = at::CPU(at::kFloat).rand({N, M});
  at::Tensor O1 = at::CPU(at::kFloat).rand({N, M});
  at::Tensor O2 = at::CPU(at::kFloat).rand({N, M});
  at::Tensor O3 = at::CPU(at::kFloat).rand({N, M});
  at::Tensor O1c = at::CPU(at::kFloat).rand({N, M});
  at::Tensor O2c = at::CPU(at::kFloat).rand({N, M});
  at::Tensor O3c = at::CPU(at::kFloat).rand({N, M});

  Jit jit;
  jit.codegenScop("kernel_anon", *scop);
  auto fptr = (void (*)(float*, float*, float*, float*, float*, float*, float*))
                  jit.getSymbolAddress("kernel_anon");
  fptr(
      A.data<float>(),
      B.data<float>(),
      C.data<float>(),
      D.data<float>(),
      O1.data<float>(),
      O2.data<float>(),
      O3.data<float>());

  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < M; c1 += 1) {
      O1c[c0][c1] = 0;
      for (int c2 = 0; c2 < K; c2 += 1) {
        for (int c3 = 0; c3 < L; c3 += 1) {
          O1c[c0][c1] += A[c0][c1][c2][c3] * B[c0][c1];
        }
      }
    }
  }
  checkRtol(O1c - O1, {A, B}, 2 * N * M * K * L);

  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < M; c1 += 1) {
      O2c[c0][c1] = C[c0][c1] * D[c0][c1];
    }
  }
  checkRtol(O2c - O2, {C, D}, N * M);

  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < M; c1 += 1) {
      O3c[c0][c1] = O1c[c0][c1] + O2c[c0][c1];
    }
  }
  checkRtol(O3c - O3, {O1, O2}, N * M);
}

TEST(LLVMCodegen, BatchMatMul) {
  auto B = 15;
  auto N = 40;
  auto M = 24;
  auto K = 21;
  std::string tc = R"(
  def batch_matmul(float(B, N, M) X, float(B, M, K) Y) -> (Z) {
    Z(b, n, k) +=! X(b, n, mm) * Y(b, mm, k)
  }
)";
  at::Tensor X = at::CPU(at::kFloat).rand({B, N, M});
  at::Tensor Y = at::CPU(at::kFloat).rand({B, M, K});
  at::Tensor O = X.bmm(Y);
  at::Tensor Oc = at::CPU(at::kFloat).zeros_like(O);

  auto ctx = isl::with_exceptions::globalIslCtx();
  auto scop = polyhedral::Scop::makeScop(ctx, tc);
  auto context = scop->makeContext(std::unordered_map<std::string, int>{
      {"N", N}, {"M", M}, {"K", K}, {"B", B}});
  scop = Scop::makeSpecializedScop(*scop, context);

  Jit jit;
  jit.codegenScop("batch_matmul", *scop);
  auto fptr =
      (void (*)(float*, float*, float*))jit.getSymbolAddress("batch_matmul");
  fptr(X.data<float>(), Y.data<float>(), Oc.data<float>());
  checkRtol(O - Oc, {Y, X}, M, 3e-7);
}

TEST(LLVMCodegen, Convolution) {
  auto NN = 12;
  auto C = 4;
  auto O = 5;
  auto W = 14;
  auto H = 13;
  auto KW = 2;
  auto KH = 3;
  std::string tc = R"(
      def convolution(float(N,C,H,W) I, float(O,C,KH,KW) W1, float(O) B)
      -> (tmp, O1) {
        tmp(n, o, h, w) +=! I(n, c, h + kh, w + kw) * W1(o, c, kh, kw)
        O1(n, o, h, w) = tmp(n, o, h, w) + B(o)
      }
    )";

  at::Tensor I = at::CPU(at::kFloat).rand({NN, C, H, W});
  at::Tensor W1 = at::CPU(at::kFloat).rand({O, C, KH, KW});
  at::Tensor B = at::CPU(at::kFloat).rand({O});
  at::Tensor expected = at::conv2d(I, W1, at::IntList{KH, KW}, B);

  auto ctx = isl::with_exceptions::globalIslCtx();
  auto scop = polyhedral::Scop::makeScop(ctx, tc);
  auto context =
      scop->makeContext(std::unordered_map<std::string, int>{{"N", NN},
                                                             {"O", O},
                                                             {"H", H},
                                                             {"KH", KH},
                                                             {"W", W},
                                                             {"KW", KW},
                                                             {"C", C}});
  scop = Scop::makeSpecializedScop(*scop, context);

  Jit jit;
  jit.codegenScop("convolution", *scop);
  auto fptr =
      (void (*)(float*, float*, float*, float*, float*))jit.getSymbolAddress(
          "convolution");
  at::Tensor tmp = at::CPU(at::kFloat).zeros_like(expected);
  at::Tensor output = at::CPU(at::kFloat).zeros_like(expected);

  fptr(
      I.data<float>(),
      W1.data<float>(),
      B.data<float>(),
      tmp.data<float>(),
      output.data<float>());
  CHECK_EQ(output.ndimension(), 4);
  checkRtol(output - expected, {I, W1, B}, C * KH * KW, 1e-6);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  initialize_llvm();
  return RUN_ALL_TESTS();
}
