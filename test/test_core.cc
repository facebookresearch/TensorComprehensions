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
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/core/flags.h"
#include "tc/core/halide_utils.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/tensor.h"
#include "tc/external/isl.h"
#include "tc/lang/error_report.h"
#include "tc/library/copy.h"
#include "tc/library/matmul.h"

using namespace std;

using namespace tc;

TEST(PrintDLTensor, Default) {
  auto p = tc::detail::makeDLTensor<DLTensor, int>(
      getCPUDLContext(), DLDataType{DLDataTypeCode::kDLInt, 8, 1}, {3, 4, 5});
  auto expected = R"STR(DLTensor@0:
shape: 3
shape: 4
shape: 5
strides: 20
strides: 5
strides: 1
alignment: 0
dtype {
  code: 0
  bits: 8
  lanes: 1
}
)STR";
  ASSERT_EQ(expected, toString(*p));
}

struct GenericHalideCoreTest : public ::testing::Test {
  void CheckC(const std::string& tc, const std::vector<std::string>& expected) {
    auto curPos = std::string::npos;
    auto halide =
        tc2halide::translate(isl::with_exceptions::globalIslCtx(), tc);
    auto res = tc::halideCodegenC(halide.stmt);
    for (const auto& e : expected) {
      auto newPos = res.find(e);
      if (curPos == std::string::npos) {
        curPos = newPos;
      }
      ASSERT_NE(std::string::npos, res.find(e)) << "No: " << e << " in:\n"
                                                << res;
      ASSERT_GE(newPos, curPos)
          << "Improper ordering of expected outputs in:" << res;
      curPos = newPos;
    }
  }
};

TEST_F(GenericHalideCoreTest, TwoMatmul) {
  string tc = R"TC(
def fun(float(M, K) I, float(K, N) W1, float(N, P) W2) -> (O1, O2) {
    O1(m, n) +=!  I(m, r_k) * W1(r_k, n)
    O2(m, p) +=! O1(m, r_n) * W2(r_n, p)
}
)TC";
  CheckC(
      tc,
      {
          "for (int O1_s0_m = 0; O1_s0_m < M; O1_s0_m++) {",
          "  for (int O1_s0_n = 0; O1_s0_n < N; O1_s0_n++) {",
          "    O1[O1_s0_m][O1_s0_n] = 0.000000f",
          "    for (int O1_s1_r_k = 0; O1_s1_r_k < K; O1_s1_r_k++) {",
          "      O1[O1_s0_m][O1_s0_n] = (O1[O1_s0_m][O1_s0_n] + (I[O1_s0_m][O1_s1_r_k]*W1[O1_s1_r_k][O1_s0_n]))",
          "for (int O2_s0_m = 0; O2_s0_m < M; O2_s0_m++) {",
          "  for (int O2_s0_p = 0; O2_s0_p < P; O2_s0_p++) {",
          "    O2[O2_s0_m][O2_s0_p] = 0.000000f",
          "    for (int O2_s1_r_n = 0; O2_s1_r_n < N; O2_s1_r_n++) {",
          "      O2[O2_s0_m][O2_s0_p] = (O2[O2_s0_m][O2_s0_p] + (O1[O2_s0_m][O2_s1_r_n]*W2[O2_s1_r_n][O2_s0_p]))",
      });
}

TEST_F(GenericHalideCoreTest, Convolution) {
  string tc = R"TC(
def fun(float(N, C, H, W) I1, float(C, F, KH, KW) W1) -> (O1) {
    O1(n, f, h, w) +=! I1(n, r_c, h + r_kh, w + r_kw) * W1(r_c, f, r_kh, r_kw)
}
)TC";
  CheckC(
      tc,
      {"for (int O1_s0_n = 0; O1_s0_n < N; O1_s0_n++) {",
       "  for (int O1_s0_f = 0; O1_s0_f < F; O1_s0_f++) {",
       "    for (int O1_s0_h = 0; O1_s0_h < ((H - KH) + 1); O1_s0_h++) {",
       "      for (int O1_s0_w = 0; O1_s0_w < ((W - KW) + 1); O1_s0_w++) {",
       "        O1[O1_s0_n][O1_s0_f][O1_s0_h][O1_s0_w] = 0.000000f",
       "        for (int O1_s1_r_c = 0; O1_s1_r_c < C; O1_s1_r_c++) {",
       "          for (int O1_s1_r_kh = 0; O1_s1_r_kh < KH; O1_s1_r_kh++) {",
       "            for (int O1_s1_r_kw = 0; O1_s1_r_kw < KW; O1_s1_r_kw++) {",
       "              O1[O1_s0_n][O1_s0_f][O1_s0_h][O1_s0_w] = (O1[O1_s0_n][O1_s0_f][O1_s0_h][O1_s0_w] + (I1[O1_s0_n][O1_s1_r_c][(O1_s0_h + O1_s1_r_kh)][(O1_s0_w + O1_s1_r_kw)]*W1[O1_s1_r_c][O1_s0_f][O1_s1_r_kh][O1_s1_r_kw]))"});
}

TEST_F(GenericHalideCoreTest, Copy) {
  CheckC(
      makeCopyTc(3),
      {"for (int O_s0_i0 = 0; O_s0_i0 < P0; O_s0_i0++) {",
       "  for (int O_s0_i1 = 0; O_s0_i1 < P1; O_s0_i1++) {",
       "    for (int O_s0_i2 = 0; O_s0_i2 < P2; O_s0_i2++) {",
       "      O[O_s0_i0][O_s0_i1][O_s0_i2] = I[O_s0_i0][O_s0_i1][O_s0_i2];"});
}

TEST_F(GenericHalideCoreTest, GroupConvolution) {
  string tc = R"TC(
def fun(float(N, G, C, H, W) I1, float(G, C, F, KH, KW) W1) -> (O1) {
    O1(n, g, f, h, w) +=! I1(n, g, r_c, h + r_kh, w + r_kw) * W1(g, r_c, f, r_kh, r_kw)
}
)TC";
  CheckC(
      tc,
      {"for (int O1_s0_n = 0; O1_s0_n < N; O1_s0_n++) {",
       "  for (int O1_s0_g = 0; O1_s0_g < G; O1_s0_g++) {",
       "    for (int O1_s0_f = 0; O1_s0_f < F; O1_s0_f++) {",
       "      for (int O1_s0_h = 0; O1_s0_h < ((H - KH) + 1); O1_s0_h++) {",
       "        for (int O1_s0_w = 0; O1_s0_w < ((W - KW) + 1); O1_s0_w++) {",
       "          O1[O1_s0_n][O1_s0_g][O1_s0_f][O1_s0_h][O1_s0_w] = 0.000000f",
       "          for (int O1_s1_r_c = 0; O1_s1_r_c < C; O1_s1_r_c++) {",
       "            for (int O1_s1_r_kh = 0; O1_s1_r_kh < KH; O1_s1_r_kh++) {",
       "              for (int O1_s1_r_kw = 0; O1_s1_r_kw < KW; O1_s1_r_kw++) {",
       "                O1[O1_s0_n][O1_s0_g][O1_s0_f][O1_s0_h][O1_s0_w] = (O1[O1_s0_n][O1_s0_g][O1_s0_f][O1_s0_h][O1_s0_w] + (I1[O1_s0_n][O1_s0_g][O1_s1_r_c][(O1_s0_h + O1_s1_r_kh)][(O1_s0_w + O1_s1_r_kw)]*W1[O1_s0_g][O1_s1_r_c][O1_s0_f][O1_s1_r_kh][O1_s1_r_kw]))"});
}

TEST_F(GenericHalideCoreTest, Matmul) {
  CheckC(
      makeMatmulTc(false, false),
      std::vector<std::string>{
          "for (int O_s0_i = 0; O_s0_i < N; O_s0_i++) {",
          "  for (int O_s0_j = 0; O_s0_j < M; O_s0_j++) {",
          "    O[O_s0_i][O_s0_j] = 0.000000f;",
          "    for (int O_s1_k = 0; O_s1_k < K; O_s1_k++) {",
          "      O[O_s0_i][O_s0_j] = (O[O_s0_i][O_s0_j] + (A[O_s0_i][O_s1_k]*B[O_s1_k][O_s0_j]));"});
}

using namespace isl::with_exceptions;

struct TC2Isl : public ::testing::Test {
  void SetUp() {}
  void Check(const string& tc, const std::vector<int64_t>& inputSizes) {
    TensorInfo ti(
        DLDataType{kDLFloat, 32, 1},
        0,
        inputSizes,
        makeStridesFromSizes(inputSizes));
    DLConstTensorUPtr in = makeDLConstTensor(ti);

    // Must reuse the same ctx or memleaks ensue!
    tc2halide::HalideComponents comps =
        tc2halide::translate(isl::with_exceptions::globalIslCtx(), tc);
    auto scop =
        polyhedral::Scop::makeScop(isl::with_exceptions::globalIslCtx(), comps);
    polyhedral::detail::validateSchedule(scop->scheduleRoot());
    // Just check no crashes
    auto outputs = inferOutputTensorInfo(comps, {in.get()});
    // Check schedule construction equality
    auto scheduleHalide = polyhedral::detail::fromIslSchedule(
        polyhedral::detail::toIslSchedule(scop->scheduleRoot()).reset_user());
  }
};

TEST_F(TC2Isl, Copy1D) {
  string tc = R"TC(
def fun(float(M) I) -> (O) {
    O(m) = I(m)
}
)TC";
  Check(tc, {123});
}

TEST_F(TC2Isl, Copy2D) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O) {
    O(m, n) = I(m, n)
}
)TC";
  Check(tc, {123, 1});
}

TEST_F(TC2Isl, Copy3D) {
  string tc = R"TC(
def fun(float(M, N, P) I) -> (O) {
    O(m, n, p) = I(m, n, p)
}
)TC";
  Check(tc, {123, 3, 2});
}

TEST_F(TC2Isl, Copy4D) {
  string tc = R"TC(
def fun(float(M, N, P, Q) I) -> (O) {
    O(m, n, p, q) = I(m, n, p, q)
}
)TC";
  Check(tc, {123, 3, 4, 5});
}

TEST_F(TC2Isl, Copy5D) {
  string tc = R"TC(
def fun(float(M, N, P, Q, R) I) -> (O) {
    O(m, n, p, q, r) = I(m, n, p, q, r)
}
)TC";
  Check(tc, {123, 10, 2, 3, 4});
}

// Invalid TC atm
TEST_F(TC2Isl, DISABLED_Reduction1D) {
  string tc = R"TC(
def fun(float(M) I) -> (O) {
    O(0) +=! I(r_m)
}
)TC";
  Check(tc, {123});
}

TEST_F(TC2Isl, Reduction2D) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O) {
    O(m) +=! I(m, r_n)
}
)TC";
  Check(tc, {123, 12});
}

TEST_F(TC2Isl, Reduction3D) {
  string tc = R"TC(
def fun(float(M, N, P) I) -> (O) {
    O(m) +=! I(m, r_n, r_p)
}
)TC";
  Check(tc, {123, 12, 16});
}

TEST_F(TC2Isl, Copy1D2Stmt) {
  string tc = R"TC(
def fun(float(M) I) -> (O1, O2) {
    O1(m) = I(m)
    O2(m) = O1(m)
}
)TC";
  Check(tc, {123});
}

TEST_F(TC2Isl, Copy2D2Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
    O1(m, n) =  I(m, n)
    O2(m, n) = O1(m, n)
}
)TC";
  Check(tc, {123, 13});
}

TEST_F(TC2Isl, Copy2D3Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2, O3) {
    O1(m, n) =  I(m, n)
    O2(m, n) = O1(m, n)
    O3(m, n) = O2(m, n)
}
)TC";
  Check(tc, {123, 13});
}

// Invalid TC atm
TEST_F(TC2Isl, DISABLED_Reduction1D2Stmt) {
  string tc = R"TC(
def fun(float(M) I) -> (O1, O2) {
    O1(m) =  I(m)
    O2(m) = O1(m)
}
)TC";
  Check(tc, {123});
}

TEST_F(TC2Isl, Reduction2D2StmtA) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
    O1(m) +=! I(m, r_n)
    O2(m)  = O1(m)
}
)TC";
  Check(tc, {123, 13});
}

TEST_F(TC2Isl, Reduction2D2StmtB) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
    O1(m, n) =   I(m, n)
    O2(m)   +=! O1(m, r_n)
}
)TC";
  Check(tc, {123, 13});
}

TEST_F(TC2Isl, Reduction2D3Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2, O3) {
    O1(m, n) =   I(m, n)
    O2(m)   +=! O1(m, r_n)
    O3(m)    =  O2(m)
}
)TC";
  Check(tc, {123, 13});
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
