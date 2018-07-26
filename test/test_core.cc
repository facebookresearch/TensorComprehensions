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
#include "tc/utils/compiler_options.h"

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
    auto curPos = 0;
    auto halide = tc2halide::translate(
        isl::with_exceptions::globalIslCtx(), tc, CompilerOptions());
    auto res = tc::halideCodegenC(halide.stmt);
    for (const auto& e : expected) {
      auto newPos = res.find(e, curPos);
      ASSERT_NE(std::string::npos, newPos)
        << "No: " << e << " in:\n" << res;
      curPos = newPos;
    }
  }
  void CheckC(const std::string& tc, const std::string& expected) {
    std::istringstream stream(expected);
    std::string line;
    std::vector<std::string> split;
    while (std::getline(stream, line)) {
      // Skip lines containing (only) closing brace.
      if (line.find('}') == std::string::npos) {
        split.emplace_back(line);
      }
    }
    CheckC(tc, split);
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
      R"C(
for (int m = 0; m < M; m++) {
  for (int n = 0; n < N; n++) {
    O1[m][n] = 0.000000f;
  }
}
for (int m = 0; m < M; m++) {
  for (int n = 0; n < N; n++) {
    for (int r_k = 0; r_k < K; r_k++) {
      O1[m][n] = (O1[m][n] + (I[m][r_k]*W1[r_k][n]));
    }
  }
}
for (int m = 0; m < M; m++) {
  for (int p = 0; p < P; p++) {
    O2[m][p] = 0.000000f;
  }
}
for (int m = 0; m < M; m++) {
  for (int p = 0; p < P; p++) {
    for (int r_n = 0; r_n < N; r_n++) {
      O2[m][p] = (O2[m][p] + (O1[m][r_n]*W2[r_n][p]));
    }
  }
}
)C");
}

TEST_F(GenericHalideCoreTest, Convolution) {
  string tc = R"TC(
def fun(float(N, C, H, W) I1, float(C, F, KH, KW) W1) -> (O1) {
    O1(n, f, h, w) +=! I1(n, r_c, h + r_kh, w + r_kw) * W1(r_c, f, r_kh, r_kw)
}
)TC";
  CheckC(
      tc,
      R"C(
for (int n = 0; n < N; n++) {
  for (int f = 0; f < F; f++) {
    for (int h = 0; h < ((H - KH) + 1); h++) {
      for (int w = 0; w < ((W - KW) + 1); w++) {
        O1[n][f][h][w] = 0.000000f;
      }
    }
  }
}
for (int n = 0; n < N; n++) {
  for (int f = 0; f < F; f++) {
    for (int h = 0; h < ((H - KH) + 1); h++) {
      for (int w = 0; w < ((W - KW) + 1); w++) {
        for (int r_c = 0; r_c < C; r_c++) {
          for (int r_kh = 0; r_kh < KH; r_kh++) {
            for (int r_kw = 0; r_kw < KW; r_kw++) {
              O1[n][f][h][w] = (O1[n][f][h][w] + (I1[n][r_c][(h + r_kh)][(w + r_kw)]*W1[r_c][f][r_kh][r_kw]));
            }
          }
        }
      }
    }
  }
}
)C");
}

TEST_F(GenericHalideCoreTest, Copy) {
  CheckC(
      makeCopyTc(3),
      {"for (int i0 = 0; i0 < P0; i0++) {",
       "  for (int i1 = 0; i1 < P1; i1++) {",
       "    for (int i2 = 0; i2 < P2; i2++) {",
       "      O[i0][i1][i2] = I[i0][i1][i2];"});
}

TEST_F(GenericHalideCoreTest, GroupConvolution) {
  string tc = R"TC(
def fun(float(N, G, C, H, W) I1, float(G, C, F, KH, KW) W1) -> (O1) {
    O1(n, g, f, h, w) +=! I1(n, g, r_c, h + r_kh, w + r_kw) * W1(g, r_c, f, r_kh, r_kw)
}
)TC";
  CheckC(
      tc,
      R"C(
for (int n = 0; n < N; n++) {
  for (int g = 0; g < G; g++) {
    for (int f = 0; f < F; f++) {
      for (int h = 0; h < ((H - KH) + 1); h++) {
        for (int w = 0; w < ((W - KW) + 1); w++) {
          O1[n][g][f][h][w] = 0.000000f;
        }
      }
    }
  }
}
for (int n = 0; n < N; n++) {
  for (int g = 0; g < G; g++) {
    for (int f = 0; f < F; f++) {
      for (int h = 0; h < ((H - KH) + 1); h++) {
        for (int w = 0; w < ((W - KW) + 1); w++) {
          for (int r_c = 0; r_c < C; r_c++) {
            for (int r_kh = 0; r_kh < KH; r_kh++) {
              for (int r_kw = 0; r_kw < KW; r_kw++) {
                O1[n][g][f][h][w] = (O1[n][g][f][h][w] + (I1[n][g][r_c][(h + r_kh)][(w + r_kw)]*W1[g][r_c][f][r_kh][r_kw]));
              }
            }
          }
        }
      }
    }
  }
}
)C");
}

TEST_F(GenericHalideCoreTest, Matmul) {
  CheckC(
      makeMatmulTc(false, false),
      R"C(
for (int i = 0; i < N; i++) {
  for (int j = 0; j < M; j++) {
    O[i][j] = 0.000000f;
  }
}
for (int i = 0; i < N; i++) {
  for (int j = 0; j < M; j++) {
    for (int k = 0; k < K; k++) {
      O[i][j] = (O[i][j] + (A[i][k]*B[k][j]));
    }
  }
}
)C");
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
    tc2halide::HalideComponents comps = tc2halide::translate(
        isl::with_exceptions::globalIslCtx(), tc, CompilerOptions());
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
