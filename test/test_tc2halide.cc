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
#include "tc/core/tensor.h"

using namespace tc;
using namespace std;

struct TC2Isl : public ::testing::Test {
  void SetUp() {}
  void Check(const string& tc) {
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
  Check(tc);
}

TEST_F(TC2Isl, Copy2D) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O) {
  O(i, j) = I(i, j)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Copy3D) {
  string tc = R"TC(
def fun(float(M, N, P) I) -> (O) {
  O(i, j, k) = I(i, j, k)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Copy4D) {
  string tc = R"TC(
def fun(float(M, N, P, Q) I) -> (O) {
  O(i, j, k, l) = I(i, j, k, l)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Copy5D) {
  string tc = R"TC(
def fun(float(M, N, P, Q, R) I) -> (O) {
  O(i, j, k, l, m) = I(i, j, k, l, m)
}
)TC";
  Check(tc);
}

// Invalid TC atm
TEST_F(TC2Isl, DISABLED_Reduction1D) {
  string tc = R"TC(
def fun(float(M) I) -> (O) {
  O(0) +=! I(i)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Reduction2D) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O) {
  O(i) +=! I(i, j)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Reduction3D) {
  string tc = R"TC(
def fun(float(M, N, P) I) -> (O) {
  O(i) +=! I(i, j, k)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Copy1D2Stmt) {
  string tc = R"TC(
def fun(float(M) I) -> (O1, O2) {
  O1(i) = I(i)
  O2(i) = O1(i)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Copy2D2Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
  O1(i, j) = I(i, j)
  O2(i, j) = O1(i, j)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Copy2D3Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2, O3) {
  O1(i, j) = I(i, j)
  O2(i, j) = O1(i, j)
  O3(i, j) = O2(i, j)
}
)TC";
  Check(tc);
}

// Invalid TC atm
TEST_F(TC2Isl, DISABLED_Reduction1D2Stmt) {
  string tc = R"TC(
def fun(float(M) I) -> (O1, O2) {
  O1(i) = I(i)
  O2(i) = O1(i)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Reduction2D2StmtA) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
  O1(i) +=! I(i, j)
  O2(i) = O1(i)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Reduction2D2StmtB) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2) {
  O1(i, j) = I(i, j)
  O2(i) +=! O1(i, j)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, Reduction2D3Stmt) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O1, O2, O3) {
  O1(i, j) = I(i, j)
  O2(i) +=! O1(i, j)
  O3(i) = O2(i)
}
)TC";
  Check(tc);
}

TEST_F(TC2Isl, MutableInput) {
  string tc = R"TC(
def foo(float(N) A) -> (B) {
    A(i) = A(i) + 42
    B(k) +=! A(i) where k in 0:1
}
)TC";
  EXPECT_THROW(Check(tc), ::lang::ErrorReport);
}
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
