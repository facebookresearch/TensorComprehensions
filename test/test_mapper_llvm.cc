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

#include "tc/aten/aten.h"
#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/cpu/cpu_tc_executor.h"
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
    C(n, m) = A(n, m) + B(n, m)
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

TEST(LLVMCodegen, MultiStmt) {
  string tc = R"TC(
 def fun(float(N, M, K, L) A, float(N, M) B, float(N, M) C, float(N, M) D)
 -> (O1, O2, O3)
 {
     O1(n, m) +=! A(n, m, r_k, r_l) * B(n, m)
     O2(n, m) = C(n, m) * D(n, m)
     O3(n, m) = O1(n, m) + O2(n, m)
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
    Z(b, n, k) +=! X(b, n, r_m) * Y(b, r_m, k)
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
def convolution(float(N,C,H,W) I, float(O,C,KH,KW) W1, float(O) B) -> (tmp, O1)
{
    tmp(n, o, h, w) +=!  I(n, r_c, h + r_kh, w + r_kw) * W1(o, r_c, r_kh, r_kw)
     O1(n, o, h, w)  = tmp(n, o, h, w) + B(o)
}
    )";

  at::Tensor I = at::CPU(at::kFloat).rand({NN, C, H, W});
  at::Tensor W1 = at::CPU(at::kFloat).rand({O, C, KH, KW});
  at::Tensor B = at::CPU(at::kFloat).rand({O});
  at::Tensor expected = at::conv2d(I, W1, B);

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
