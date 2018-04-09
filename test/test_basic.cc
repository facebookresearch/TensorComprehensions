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
#include <functional>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include "tc/core/scope_guard.h"
#include "tc/external/isl.h"

#include "tc/core/utils/math.h"

using namespace std;

DEFINE_int32(B, 32, "Batch size");

TEST(ATenTest, Default) {
  LOG(INFO) << "creating an aten tensor" << std::endl;
  at::Tensor b = at::CPU(at::kFloat).rand({3, 4});
  LOG(INFO) << b.numel() << std::endl;
  LOG(INFO) << "Successfully imported ATen" << std::endl;
}

TEST(GFlags, Default) {
  cout << "cout: " << FLAGS_B << endl;
  VLOG(1) << "VLOG(1): " << FLAGS_B << endl;
  LOG(INFO) << "LOG(INFO): " << FLAGS_B << endl;
  auto l = [&]() { LOG(FATAL) << "LOG(FATAL): " << FLAGS_B << endl; };
  EXPECT_DEATH(l(), "");
}

struct SomeType {
  int i;
  SomeType() : i(0) {}
};

struct SomeOtherType {
  int i;
  SomeOtherType() : i(0) {}
};

int run0_1(int (*callback)(SomeType, void*), void* user) {
  return callback(SomeType(), user);
}

int run1_1(float f, int (*callback)(SomeType, void*), void* user) {
  return callback(SomeType(), user);
}

int run1_2(
    float f,
    int (*callback)(SomeType, SomeOtherType, void*),
    void* user) {
  return callback(SomeType(), SomeOtherType(), user);
}

int run2_2(
    float f,
    SomeType st,
    int (*callback)(SomeType, SomeOtherType, void*),
    void* user) {
  return callback(SomeType(), SomeOtherType(), user);
}

TEST(Math, Median) {
  std::vector<int> v0{};
  EXPECT_THROW(tc::median(v0), std::out_of_range);

  std::vector<int> v1({1});
  EXPECT_EQ(tc::median(v1), 1);

  std::vector<int> v2{{1, 3}};
  EXPECT_EQ(tc::median(v2), 2);

  std::vector<int> v3{{1, 1000, 2}};
  EXPECT_EQ(tc::median(v3), 2);

  std::vector<int> v4{{-42, 517, 4, 2}};
  EXPECT_EQ(tc::median(v4), 3);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
