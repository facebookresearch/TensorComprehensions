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
#include <iostream>
#include <sstream>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/tc2halide.h"
#include "tc/core/utils/dlpack.h"

using namespace tc;
using namespace std;
using namespace tc::dlutils;

struct TC2Isl : public ::testing::Test {
  void SetUp() {}
  void Check(const string& tc, const std::vector<long>& inputSizes) {
    auto ctx = getCPUDLContext();
    DLDataType dtype;
    dtype.code = kDLFloat;
    dtype.bits = 32;
    dtype.lanes = 1;
    auto UPtr = makeDLTensorWithSizes(ctx, dtype, inputSizes);
    std::vector<const DLTensor*> inputs{UPtr.get()};

    auto halide =
        tc2halide::translate(isl::with_exceptions::globalIslCtx(), tc);
    auto scop = polyhedral::Scop::makeScop(
        isl::with_exceptions::globalIslCtx(), halide);
    auto scheduleHalide = scop->scheduleRoot();

    polyhedral::detail::validateSchedule(scheduleHalide);
  }
};

TEST_F(TC2Isl, Copy1D) {
  string tc = R"TC(
def fun(float(M) I) -> (O) {
  O(i) = I(i)
}
)TC";
  Check(tc, {123});
}

TEST_F(TC2Isl, Copy2D) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O) {
  O(i, j) = I(i, j)
}
)TC";
  Check(tc, {123, 1});
}

TEST_F(TC2Isl, Copy3D) {
  string tc = R"TC(
def fun(float(M, N, P) I) -> (O) {
  O(i, j, k) = I(i, j, k)
}
)TC";
  Check(tc, {123, 3, 2});
}

TEST_F(TC2Isl, Copy4D) {
  string tc = R"TC(
def fun(float(M, N, P, Q) I) -> (O) {
  O(i, j, k, l) = I(i, j, k, l)
}
)TC";
  Check(tc, {123, 3, 4, 5});
}

TEST_F(TC2Isl, Copy5D) {
  string tc = R"TC(
def fun(float(M, N, P, Q, R) I) -> (O) {
  O(i, j, k, l, m) = I(i, j, k, l, m)
}
)TC";
  Check(tc, {123, 10, 2, 3, 4});
}

// Invalid TC atm
TEST_F(TC2Isl, DISABLED_Reduction1D) {
  string tc = R"TC(
def fun(float(M) I) -> (O) {
  O(0) +=! I(i)
}
)TC";
  Check(tc, {123});
}

TEST_F(TC2Isl, Reduction2D) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O) {
  O(i) +=! I(i, j)
}
)TC";
  Check(tc, {123, 12});
}

TEST_F(TC2Isl, Reduction3D) {
  string tc = R"TC(
def fun(float(M, N, P) I) -> (O) {
  O(i) +=! I(i, j, k)
}
)TC";
  Check(tc, {123, 12, 16});
}

TEST_F(TC2Isl, Copy1D2Stmt) {
  string tc = R"TC(
def fun(float(M) I) -> (O1, O2) {
  O1(i) = I(i)
  O2(i) = O1(i)
}
)TC";
  Check(tc, {123});
}

TEST_F(TC2Isl, Copy2D2Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
  O1(i, j) = I(i, j)
  O2(i, j) = O1(i, j)
}
)TC";
  Check(tc, {123, 13});
}

TEST_F(TC2Isl, Copy2D3Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2, O3) {
  O1(i, j) = I(i, j)
  O2(i, j) = O1(i, j)
  O3(i, j) = O2(i, j)
}
)TC";
  Check(tc, {123, 13});
}

// Invalid TC atm
TEST_F(TC2Isl, DISABLED_Reduction1D2Stmt) {
  string tc = R"TC(
def fun(float(M) I) -> (O1, O2) {
  O1(i) = I(i)
  O2(i) = O1(i)
}
)TC";
  Check(tc, {123});
}

TEST_F(TC2Isl, Reduction2D2StmtA) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
  O1(i) +=! I(i, j)
  O2(i) = O1(i)
}
)TC";
  Check(tc, {123, 13});
}

TEST_F(TC2Isl, Reduction2D2StmtB) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
  O1(i, j) = I(i, j)
  O2(i) +=! O1(i, j)
}
)TC";
  Check(tc, {123, 13});
}

TEST_F(TC2Isl, Reduction2D3Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2, O3) {
  O1(i, j) = I(i, j)
  O2(i) +=! O1(i, j)
  O3(i) = O2(i)
}
)TC";
  Check(tc, {123, 13});
}

TEST_F(TC2Isl, MutableInput) {
  string tc = R"TC(
def foo(float(N) A) -> (B) {
    A(i) = A(i) + 42
    B(k) +=! A(i) where k in 0:1
}
)TC";
  EXPECT_THROW(Check(tc, {123}), ::lang::ErrorReport);
}
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
