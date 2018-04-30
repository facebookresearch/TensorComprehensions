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
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/library/common.h"

#include "test_harness_aten_cuda.h"

using tensor_list = std::vector<at::Tensor>;

template <typename... Args>
static at::Tensor F(Args... args) {
  return at::CUDA(at::kFloat).rand({args...});
}

template <typename... Args>
static at::Tensor I(Args... args) {
  return at::CUDA(at::kInt).ones({args...});
}

// this file is for easy-to-write tests that ensure code either fails well
// (Fail) or at least runs.

// The intention here is not to create long correctness tests with reference
// outputs those are handled in other files. Instead here we want really short
// tests that make sure the compiler doesn't crash in weird circumstances and
// produces reasonable errors in failure cases.

// we expect this to succeed, we don't really care about what the outputs are
// there are other tests that check for correctness
static void Succeed(
    const std::string& str,
    const tensor_list& inputs,
    tensor_list&& outputs,
    std::string fn = "f") {
  auto pExecutor = tc::aten::compile<tc::CudaBackend>(
      str, fn, inputs, tc::CudaMappingOptions::makeNaiveMappingOptions());
  tc::aten::run(*pExecutor, inputs, outputs);
}

static void Fail(
    const std::string& fail_msg,
    const std::string& str,
    const tensor_list& inputs,
    tensor_list&& outputs,
    std::string fn = "f") {
  try {
    Succeed(str, inputs, std::move(outputs), fn);
    FAIL() << "failed to fail on a test that expected a failure.";
  } catch (lang::ErrorReport e) {
    if (std::string::npos == std::string(e.what()).find(fail_msg)) {
      FAIL() << "expected fragment '" << fail_msg << "' but got\n" << e.what();
    }
  }
}

TEST(TestCornerCases, E1) {
  Fail("expected (", " def f{} {}", {}, {});
}
TEST(TestCornerCases, E2) {
  Succeed("def f(float(1) a) -> (b) { b(i) = a(i) }", {F(1)}, {F(1)});
}

// free(): invalid next size (fast): 0x000000003b2d6230 ***
TEST(TestCornerCases, DISABLED_E4) {
  Succeed("def f(float a) -> (b) { b = a }", {F()}, {F()});
}

// main conflicts with program main in nvcc
TEST(TestCornerCases, DISABLED_E3) {
  Succeed(
      "def main(float(1) a) -> (b) { b(i) = a(i) }", {F(1)}, {F(1)}, "main");
}

// segfaults on line:
// src/aten/aten_compiler.cc:123
// 123    at::Backend backend = inputs[0].type().backend();
TEST(TestCornerCases, DISABLED_E5) {
  Succeed("def f() -> (b) { b(i) = 4 where i in 0:10 }", {}, {F(0)});
}

TEST(TestCornerCases, E6) {
  Succeed("def f(float a) -> (b) { b(i) = a where i in 0:10 }", {F()}, {F(10)});
}

TEST(TestCornerCases, E7) {
  Fail(
      "expected 2 inputs",
      "def f(float a, float c) -> (b) { b(i) = a where i in 0:10 }",
      {F()},
      {F(10)});
}

TEST(TestCornerCases, E8) {
  Fail(
      "expected type int32",
      "def f(int32 a) -> (b) { b(i) = a where i in 0:10 }",
      {F()},
      {F(10)});
}

TEST(TestCornerCases, E9) {
  Fail(
      "expected a tensor with 0",
      "def f(int32 a) -> (b) { b(i) = a where i in 0:10 }",
      {I(1, 2)},
      {F(10)});
}

TEST(TestCornerCases, E10) {
  Succeed("def f(int32 a) -> (b) { b(i) = a where i in 0:10 }", {I()}, {I(10)});
}

TEST(TestCornerCases, E11) {
  Fail(
      "expected integral type",
      "def f(int32(N) a) -> (b) { b(i) = a(i + .5) }",
      {I()},
      {I(10, 10)});
}

TEST(TestCornerCases, E12) {
  // this test should eventually work when we can handle non-trivial
  // expressions in where clauses
  Fail(
      "tensor accesses cannot be used in this context",
      "def f(int32 a, float(N) b) -> (c) { c(i) += b(i + j) where j in 0:a, i in 0:10 }",
      {I(), F(12)},
      {I(10)});
}

TEST(TestCornerCases, E13) {
  // this test is harder still, because the bounds of the output
  // depend on the non-trivial expression
  Fail(
      "tensor accesses cannot be used in this context",
      "def f(int32 a, float(N) b) -> (c) { c(i) += b(i + j) where j in 0:a }",
      {I(), F(12)},
      {I(10)});
}

TEST(TestCornerCases, DISABLED_E14) {
  // Currently expressions in where clauses are assumed to be
  // affine. Needs fixing.
  Fail(
      "tensor accesses cannot be used in this context",
      "def f(float(N) b) -> (c) { c(i) += b(i + j) where j in 0:(N*N), i in 0:10 }",
      {F(12)},
      {I(10)});
}

TEST(TestCornerCases, E15){
#define GEN_COMPARATOR(op)                                       \
  {                                                              \
    auto a = F();                                                \
    auto b = F();                                                \
    auto c = F(1);                                               \
    Succeed(                                                     \
        "def f(float a, float b) -> (c) { c(i) = float(a " #op   \
        " b) where i in 0:1 }",                                  \
        {a, b},                                                  \
        {c});                                                    \
    auto r = at::Scalar(a).toFloat() op at::Scalar(b).toFloat(); \
    CHECK_EQ(r, at::Scalar(c[0]).toFloat());                     \
  }

    GEN_COMPARATOR(<=) GEN_COMPARATOR(>=) GEN_COMPARATOR(==) GEN_COMPARATOR(!=)
        GEN_COMPARATOR(<) GEN_COMPARATOR(>)

}

TEST(TestCornerCases, E16){
#define GEN_BOOLS(op)                                                         \
  {                                                                           \
    auto a = F();                                                             \
    auto b = F();                                                             \
    auto c = F(1);                                                            \
    Succeed(                                                                  \
        "def f(float a, float b) -> (c) { c(i) = float(!(a < .5) " #op        \
        " b > .5) where i in 0:1 }",                                          \
        {a, b},                                                               \
        {c});                                                                 \
    auto r = !(at::Scalar(a).toFloat() < .5) op at::Scalar(b).toFloat() > .5; \
    ;                                                                         \
    CHECK_EQ(r, at::Scalar(c[0]).toFloat());                                  \
  }

    GEN_BOOLS(||) GEN_BOOLS(&&)}

TEST(TestCornerCases, E17) {
  auto r = F(1);
  Succeed(
      "def f(float(1) a) -> (b) { b(i) = 4.0 where exists a(i) }", {F(1)}, {r});
  CHECK_EQ(at::Scalar(r[0]).toFloat(), 4);
}

TEST(TestCornerCases, E18) {
  auto a = F(1);
  auto r = F(1);
  Succeed(
      "def f(float(1) a) -> (b) { b(i) = 2*foo where foo = a(i) }", {a}, {r});
  CHECK_EQ(at::Scalar(r[0]).toFloat(), at::Scalar(a[0]).toFloat() * 2);
}
TEST(TestCornerCases, E19) {
  Fail(
      "undefined variable",
      "def f(float(1) a) -> (b) { b(i) = 2*foo where foo = a(i), foo in 1:2 }",
      {F(1)},
      {F(1)});
}
TEST(TestCornerCases, E20) {
  Fail(
      "Temporaries tensors",
      "def f(float(1) a) -> (b) { c(i) = a(i) b(i) = c(i)  }",
      {F(1)},
      {F(1)});
}

TEST(TestCornerCases, E21) {
  auto a = F(1);
  auto b = F(1);
  auto c = F(1);
  Succeed(
      "def f(float(1) a, float(1) b) -> (c) { c(i) = max(a(i), b(i)) }",
      {a, b},
      {c});
  CHECK_EQ(
      fmaxf(at::Scalar(a[0]).toFloat(), at::Scalar(b[0]).toFloat()),
      at::Scalar(c[0]).toFloat());
}

TEST(TestCornerCases, E22) {
  auto a = F(1);
  auto b = F(1);
  auto c = F(1);
  Succeed(
      "def f(float(1) a, float(1) b) -> (c) { c(i) = min(a(i), b(i)) }",
      {a, b},
      {c});
  CHECK_EQ(
      fminf(at::Scalar(a[0]).toFloat(), at::Scalar(b[0]).toFloat()),
      at::Scalar(c[0]).toFloat());
}

TEST(TestCornerCases, E23) {
  auto a = F(1);
  auto b = F(1);
  auto c = F(1);
  auto d = F(1);
  Succeed(
      "def f(float(1) a, float(1) b, float(1) c) -> (d) { d(i) = min(a(i), max(b(i), c(i))) }",
      {a, b, c},
      {d});
  CHECK_EQ(
      fminf(
          at::Scalar(a[0]).toFloat(),
          fmaxf(at::Scalar(b[0]).toFloat(), at::Scalar(c[0]).toFloat())),
      at::Scalar(d[0]).toFloat());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
