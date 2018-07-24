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

#include "tc/core/tc2halide.h"
#include "tc/lang/error_report.h"

using namespace std;
using namespace lang;
using tc::CompilerOptions;

struct InferenceTest : public ::testing::Test {
  void Check(const string& tc, const string& expected) {
    CompilerOptions compilerOptions;
    compilerOptions.throwWarnings = true;
    auto halideComponents = tc2halide::translate(
        isl::with_exceptions::globalIslCtx(), tc, compilerOptions);

    stringstream ss;
    // Ordered map for repro
    ss << "mins:" << std::endl;
    for (const auto& i : halideComponents.inputs) {
      ss << i.name() << "@[";
      for (int d = 0; d < i.dimensions(); ++d) {
        ss << i.parameter().min_constraint(d) << "; ";
      }
      ss << "]" << std::endl;
    }
    for (const auto& o : halideComponents.outputs) {
      ss << o.name() << "@[";
      for (int d = 0; d < o.dimensions(); ++d) {
        ss << o.parameter().min_constraint(d) << "; ";
      }
      ss << "]" << std::endl;
    }
    ss << "extents:" << std::endl;
    for (const auto& i : halideComponents.inputs) {
      ss << i.name() << "@[";
      for (int d = 0; d < i.dimensions(); ++d) {
        ss << i.parameter().extent_constraint(d) << "; ";
      }
      ss << "]" << std::endl;
    }
    for (const auto& o : halideComponents.outputs) {
      ss << o.name() << "@[";
      for (int d = 0; d < o.dimensions(); ++d) {
        ss << o.parameter().extent_constraint(d) << "; ";
      }
      ss << "]" << std::endl;
    }
    EXPECT_EQ(expected, ss.str());
  }
};

TEST_F(InferenceTest, Copy1D) {
  Check(
      R"(
    def fun(float(I_0) I) -> (O) {
      O(i) = I(i)
    }
  )",
      R"HALIDE(mins:
I@[0; ]
O@[0; ]
extents:
I@[I_0; ]
O@[I_0; ]
)HALIDE");
}

TEST_F(InferenceTest, Copy2D) {
  Check(
      R"(
    def fun(float(I_0, I_1) I) -> (O) {
      O(i, j) = I(i, j)
    }
  )",
      R"HALIDE(mins:
I@[0; 0; ]
O@[0; 0; ]
extents:
I@[I_0; I_1; ]
O@[I_0; I_1; ]
)HALIDE");
}

TEST_F(InferenceTest, Transpose2D) {
  Check(
      R"(
    def fun(float(I_0, I_1) I) -> (O) {
      O(i, j) = I(j, i)
    }
  )",
      R"HALIDE(mins:
I@[0; 0; ]
O@[0; 0; ]
extents:
I@[I_0; I_1; ]
O@[I_1; I_0; ]
)HALIDE");
}

TEST_F(InferenceTest, Transpose4D) {
  Check(
      R"(
    def fun(float(I_0, I_1, I_2, I_3) I) -> (O) {
      O(i, j, l, k) = I(j, i, k, l)
    }
  )",
      R"HALIDE(mins:
I@[0; 0; 0; 0; ]
O@[0; 0; 0; 0; ]
extents:
I@[I_0; I_1; I_2; I_3; ]
O@[I_1; I_0; I_3; I_2; ]
)HALIDE");
}

// This TC is not well formed (m does not appear on the RHS
TEST_F(InferenceTest, Transpose5D) {
  auto tc = R"TC(def fun(float(I_0, I_1, I_2, I_3, I_4) I) -> (O) {
      O(i, j, k, l, m) = I(j, i, k, i, l)
    })TC";
  EXPECT_THROW(Check(tc, ""), ::lang::ErrorReport);
}

TEST_F(InferenceTest, Scale1D) {
  Check(
      R"(
    def fun(float(I_0) I) -> (O) {
      O(i) = I(123 * i)
    }
  )",
      R"HALIDE(mins:
I@[0; ]
O@[0; ]
extents:
I@[I_0; ]
O@[((I_0 + 122)/123); ]
)HALIDE");
}

TEST_F(InferenceTest, Shift1D) {
  Check(
      R"(
    def fun(float(I_0) I) -> (O) {
      O(i) = I(i - 1)
    }
  )",
      R"HALIDE(mins:
I@[0; ]
O@[1; ]
extents:
I@[I_0; ]
O@[I_0; ]
)HALIDE");
}

TEST_F(InferenceTest, Shift1D2) {
  Check(
      R"(
    def fun(float(I_0) I) -> (O) {
      O(i) = I(i + 5)
    }
  )",
      R"HALIDE(mins:
I@[0; ]
O@[-5; ]
extents:
I@[I_0; ]
O@[I_0; ]
)HALIDE");
}

TEST_F(InferenceTest, Conv) {
  Check(
      R"(
      def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(M) B)
      -> (tmp, O)
      {
        tmp(n, m, h, w) +=! I(n, c, h + kh, w + kw) * W1(m, c, kh, kw)
        O(n, m, h, w) +=! tmp(n, m, h, w) + B(m)
      }
    )",
      R"HALIDE(mins:
I@[0; 0; 0; 0; ]
W1@[0; 0; 0; 0; ]
B@[0; ]
tmp@[0; 0; 0; 0; ]
O@[0; 0; 0; 0; ]
extents:
I@[N; C; H; W; ]
W1@[M; C; KH; KW; ]
B@[M; ]
tmp@[N; M; ((H - KH) + 1); ((W - KW) + 1); ]
O@[N; M; ((H - KH) + 1); ((W - KW) + 1); ]
)HALIDE");
}

TEST_F(InferenceTest, ConvGrad) {
  Check(
      R"(
      def convolutionGrad(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(N,M,OH,OW) d_O) -> (d_I, d_W1, d_B) {
        d_I(n, c, h, w) +=! d_O(n, m, h - kh, w - kw) * W1(m, c, kh, kw)
        d_W1(m, c, kh, kw) +=! d_O(n, m, h - kh, w - kw) * I(n, c, h, w)
        d_B(m) +=! d_O(n, m, h, w)
      }
    )",
      R"HALIDE(mins:
I@[0; 0; 0; 0; ]
W1@[0; 0; 0; 0; ]
d_O@[0; 0; 0; 0; ]
d_I@[0; 0; (KH + -1); (KW + -1); ]
d_W1@[0; 0; (H - OH); (W - OW); ]
d_B@[0; ]
extents:
I@[N; C; H; W; ]
W1@[M; C; KH; KW; ]
d_O@[N; M; OH; OW; ]
d_I@[N; C; ((OH - KH) + 1); ((OW - KW) + 1); ]
d_W1@[M; C; ((OH - H) + 1); ((OW - W) + 1); ]
d_B@[M; ]
)HALIDE");
}

// Padded shape should match:
// (I->shape[2] - W->shape[2] + 2 * pad_h) / stride_h + 1,  // H
// (I->shape[3] - W->shape[3] + 2 * pad_w) / stride_w + 1   // W
//
// Spice it up a bit with Pad before / after of (2,3) for H and (3, 4) for W
//
// Note: we only simulate padding by shifting because we have not yet defined
// it in TC.
TEST_F(InferenceTest, PaddedConv) {
  Check(
      R"(
      def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(M) B)
      -> (pad, tmp)
      {
        pad(n, c, h, w)  = I(n, c, h - 2 - 3, w - 3 - 4)
        tmp(n, m, h, w) +=! pad(n, c, 2 * h + kh, 3 * w + kw) * W1(m, c, kh, kw)
      }
    )",
      R"HALIDE(mins:
I@[0; 0; 0; 0; ]
W1@[0; 0; 0; 0; ]
B@[0; ]
pad@[0; 0; 5; 7; ]
tmp@[0; 0; 3; 3; ]
extents:
I@[N; C; H; W; ]
W1@[M; C; KH; KW; ]
B@[M; ]
pad@[N; C; H; W; ]
tmp@[N; M; (((H - KH) + 1)/2); (((W - KW) + 1)/3); ]
)HALIDE");
}

TEST_F(InferenceTest, WeirdConv) {
  Check(
      R"(
    def fun(float(B,IP,H,W) input, float(OP,IP,KH,KW) weight) -> (output) {
      output(b, op, h, w) +=! input(b, ip, 2 * h + 3 * w + kh, 2 * w + kw) *
                             weight(op, ip, kh, kw)
    }
  )",
      // TODO: Double-check this
      R"HALIDE(mins:
input@[0; 0; 0; 0; ]
weight@[0; 0; 0; 0; ]
output@[0; 0; 0; 0; ]
extents:
input@[B; IP; H; W; ]
weight@[OP; IP; KH; KW; ]
output@[B; OP; ((((H - KH) - (((W - KW)/2)*3))/2) + 1); (((W - KW)/2) + 1); ]
)HALIDE");
}

TEST_F(InferenceTest, TCA) {
  Check(
      R"(
    def fun(float(I_0, I_1, I_2) I) -> (O) {
      O(i) +=! I(j, i + j, k)
    }
  )",
      R"HALIDE(mins:
I@[0; 0; 0; ]
O@[0; ]
extents:
I@[I_0; I_1; I_2; ]
O@[((I_1 - I_0) + 1); ]
)HALIDE");
}

TEST_F(InferenceTest, TCB) {
  auto tc = R"TC(
    def fun(float(I_0, I_1, I_2) A) -> (O) {
      O(i) +=! A(i + j, i + j, i + j)
    }
)TC";
  EXPECT_THROW({ Check(tc, ""); }, ::lang::ErrorReport);
}

// TODO: This should fail because it is ambiguous
TEST_F(InferenceTest, TCC) {
  auto tc = R"TC(
    def fun(float(I_0, I_1, I_2) I) -> (O) {
      O(i) +=! I(i, i + j, j)
    }
)TC";
  EXPECT_THROW({ Check(tc, ""); }, ::lang::ErrorReport);
}

// TODO: This should fail because it is ambiguous
//   (2 * i + j) is never solved for so it does not contribute a min to
//   constrain the range, this is unsafe in general
TEST_F(InferenceTest, TCD) {
  auto tc = R"TC(
    def fun(float(I_0, I_1, I_2, I_3, I_4) I) -> (O) {
      O(i) +=! I(i, j, i, k, 2 * i + j)
    }
)TC";
  EXPECT_THROW({ Check(tc, ""); }, ::lang::ErrorReport);
}

// TODO: This should fail because it is ambiguous
//   (2 * i + j) is never solved for so it does not contribute a min to
//   constrain the range, this is unsafe in general
TEST_F(InferenceTest, TCE) {
  auto tc = R"TC(
    def fun(float(I_0, I_1, I_2, I_3, I_4) I) -> (O) {
      O(i) +=! I(j, j + 1, i, i + 3, 2 * i + j + 0)
    }
)TC";
  EXPECT_THROW({ Check(tc, ""); }, ::lang::ErrorReport);
}

TEST_F(InferenceTest, TCF) {
  Check(
      R"(
    def fun(float(A_0, A_1, A_2) A,
            float(B_0, B_1, B_2) B,
            float(C_0, C_1, C_2) C) -> (O2, O3) {
      O2(j, k, l) +=! B(j, k, l) + A(i, j, k)
      O3(k, l, m) +=! C(k, l, m) + O2(j, k, l)
    }
  )",
      R"HALIDE(mins:
A@[0; 0; 0; ]
B@[0; 0; 0; ]
C@[0; 0; 0; ]
O2@[0; 0; 0; ]
O3@[0; 0; 0; ]
extents:
A@[A_0; A_1; A_2; ]
B@[B_0; B_1; B_2; ]
C@[C_0; C_1; C_2; ]
O2@[min(B_0, A_1); min(B_1, A_2); B_2; ]
O3@[min(C_0, min(B_1, A_2)); min(C_1, B_2); C_2; ]
)HALIDE");
}

TEST_F(InferenceTest, TCG) {
  Check(
      R"(
    def fun(float(A_0, A_1, A_2) A,
            float(B_0, B_1, B_2) B,
            float(C_0, C_1, C_2) C) -> (O1, O2, O3) {
      O1(i, j) +=! A(i, j, k)
      O2(j, k, l) +=! B(j, k, l) + O1(i, j)
      O3(k, l, m) +=! C(k, l, m) + O2(j, k, l)
    }
  )",
      R"HALIDE(mins:
A@[0; 0; 0; ]
B@[0; 0; 0; ]
C@[0; 0; 0; ]
O1@[0; 0; ]
O2@[0; 0; 0; ]
O3@[0; 0; 0; ]
extents:
A@[A_0; A_1; A_2; ]
B@[B_0; B_1; B_2; ]
C@[C_0; C_1; C_2; ]
O1@[A_0; A_1; ]
O2@[min(B_0, A_1); B_1; B_2; ]
O3@[min(C_0, B_1); min(C_1, B_2); C_2; ]
)HALIDE");
}

TEST_F(InferenceTest, PartialCopy) {
  string tc = R"TC(
def fun(float(N, M) I) -> (O) {
  O(i, j) = I(i, j) where i in 3:N-1
}
)TC";
  Check(tc, R"HALIDE(mins:
I@[0; 0; ]
O@[3; 0; ]
extents:
I@[N; M; ]
O@[(N + -4); M; ]
)HALIDE");
}

TEST_F(InferenceTest, AmbiguousRotate) {
  string tc = R"TC(
def fun(float(N, M) I) -> (O) {
  O(i, j) = I(10 + i - j, i + j)
}
)TC";
  EXPECT_THROW(Check(tc, {}), ::lang::ErrorReport);
}

TEST_F(InferenceTest, UnambiguousRotate) {
  string tc = R"TC(
def fun(float(N, M) I) -> (O) {
  O(i, j) = I(10 + i - j, i + j) where i in 0:10
}
)TC";
  // In our current semantics this is inferrable. After solving i, we
  // get two sets of constraints on j, and we take the interval that
  // satisfies both.
  Check(tc, R"HALIDE(mins:
I@[0; 0; ]
O@[0; (20 - min(N, 20)); ]
extents:
I@[N; M; ]
O@[10; ((min((M + -10), 10) + min(N, 20)) + -19); ]
)HALIDE");
}

TEST_F(InferenceTest, RuntimeCheckOnDerivedSize) {
  string tc = R"TC(
def fun(float(N, NM, M) I) -> (O) {
  O(i, j) = I(i, i + j, j)
}
)TC";
  // Requires that N + M < NM + 2
  // We don't currently accept things that require additional runtime
  // checks to be valid.
  EXPECT_THROW(Check(tc, {}), ::lang::ErrorReport);
}

TEST_F(InferenceTest, RuntimeCheckOnDerivedSizeStaticallyProvable) {
  string tc = R"TC(
def fun(float(10, 19, 10) I) -> (O) {
  O(i, j) = I(i, i + j, j)
}
)TC";
  Check(tc, R"HALIDE(mins:
I@[0; 0; 0; ]
O@[0; 0; ]
extents:
I@[10; 19; 10; ]
O@[10; 10; ]
)HALIDE");
}

TEST_F(InferenceTest, PartialAssignWithoutWhere) {
  string tc = R"TC(
def fun() -> (O) {
  O(i, j) = 3
}
)TC";
  // i and j are underconstrained
  EXPECT_THROW(Check(tc, {}), ::lang::ErrorReport);
}

TEST_F(InferenceTest, PartialAssignWithWhere) {
  string tc = R"TC(
def fun() -> (O) {
  O(i, j) = 3 where i in 4:8, j in 3:20
}
)TC";
  Check(tc, R"HALIDE(mins:
O@[4; 3; ]
extents:
O@[4; 17; ]
)HALIDE");
}

TEST_F(InferenceTest, EmptySetWhere) {
  string tc = R"TC(
def fun() -> (O) {
  O(i, j) = 3 where i in 8:4, j in 3:20
}
)TC";
  // Is it OK to create zero-element tensors? Currently this code silently
  // infers a negative size.
  Check(tc, R"HALIDE(mins:
O@[8; 3; ]
extents:
O@[-4; 17; ]
)HALIDE");
}

// Currently throws an error in semantic checking, but we may want
// to consider allowing this.
TEST_F(InferenceTest, DISABLED_IndirectWhere) {
  string tc = R"TC(
def fun(float(TWO) size) -> (O) {
  O(i, j) = 3 where i in 0:size(0), j in 0:size(1)
}
)TC";
  Check(tc, "");
}

TEST_F(InferenceTest, IllegalIndirectWhere) {
  string tc = R"TC(
def fun(float(N) size) -> (O) {
  O(i, j) = 3 where i in 0:size(j), j in 0:size(i)
}
)TC";
  // Not a rectangular iteration domain for starters, but that's the
  // least of the problems here.
  EXPECT_THROW(Check(tc, {}), ::lang::ErrorReport);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
