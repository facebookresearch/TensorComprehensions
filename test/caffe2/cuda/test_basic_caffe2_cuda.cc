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

#include <gtest/gtest.h>

#include <caffe2/core/context_gpu.h>

#include "../test_harness.h"

using namespace caffe2;

TEST(Caffe2, Caffe2MatMulOp) {
  Workspace w;
  AddConstInput<CUDABackend, float>(w, {2, 2}, 1.f, "A");
  AddConstInput<CUDABackend, float>(w, {2, 2}, 1.f, "B");
  AddConstInput<CUDABackend, float>(w, {2, 2}, 0.f, "C");
  OperatorDef def = MakeOperatorDef<CUDABackend>("MatMul", {"A", "B"}, {"C"});
  auto op = CreateOperator(def, &w);
  op->Run();
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
}

TEST(Caffe2, ATenTensor) {
  at::Tensor t = at::CUDA(at::kFloat).rand({1}).fill_(2.0f);
  ASSERT_EQ(t[0].toCFloat(), 2.0f);
}

TEST(Caffe2, ATenTensorFromStorage) {
  auto s = at::getType(at::Backend::CUDA, at::kFloat).storage(1);
  auto t = at::getType(at::Backend::CUDA, at::kFloat).tensor(*s, 0, {1});
  t.fill_(2.0f);
  ASSERT_EQ(t[0].toCFloat(), 2.0f);
}

TEST(Caffe2, Caffe2MatMulOpToATenTensor) {
  Workspace w;
  AddConstInput<CUDABackend, float>(w, {2, 2}, 1.f, "A");
  AddConstInput<CUDABackend, float>(w, {2, 2}, 1.f, "B");
  AddConstInput<CUDABackend, float>(w, {2, 2}, 0.f, "C");
  OperatorDef def = MakeOperatorDef<CUDABackend>("MatMul", {"A", "B"}, {"C"});
  auto op = CreateOperator(def, &w);
  op->Run();
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
  auto c2tensor = GetNamedTensor<CUDABackend>(w, "A");
  auto t = at::getType(at::Backend::CUDA, at::kFloat)
               .tensorFromBlob(c2tensor.raw_mutable_data(), c2tensor.dims());
  ASSERT_EQ(t[0][0].toCFloat(), 1.0f);
  // TODO: without retain there seems to be a memory corruption happening
  // when creating a tensorFromBlob and calling the (noop) deleter. Punt for
  // now until we get a clean official PyTorch + ATen + Caffe2 package.
  t.storage()->retain();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
