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
#include "test_harness.h"

namespace caffe2 {

void CheckEqual(
    const caffe2::Tensor<caffe2::CPUContext>& Texpected,
    const caffe2::Tensor<caffe2::CPUContext>& Ttested,
    float relativePrecision,
    long offsetInExpected,
    long offsetInTested) {
  for (int i = 0; i < Texpected.size() - offsetInExpected; ++i) {
    if (relativePrecision == 0.0) {
      ASSERT_FLOAT_EQ(
          Texpected.data<float>()[i + offsetInExpected],
          Ttested.data<float>()[i + offsetInTested])
          << " for Tensor " << Texpected.DebugString() << " at position " << i;
    } else {
      // From glog's glog/src/glog/logging.h.in
      // #define CHECK_NEAR(val1, val2, margin)
      // CHECK_NEAR is actualy absolute!!!
      ASSERT_NEAR(
          Texpected.data<float>()[i + offsetInExpected],
          Ttested.data<float>()[i + offsetInTested],
          relativePrecision * Texpected.data<float>()[i + offsetInExpected])
          << " for Tensor " << Texpected.DebugString() << " at position " << i;
    }
  }
}

unique_ptr<OpTester> BasicCorrectnessTest(
    const OperatorDef& op_def,
    std::function<void(Workspace&)> ws_init_func,
    float relativePrecision,
    std::map<string, int> reference_args) {
  unique_ptr<OpTester> test(new OpTester(op_def, relativePrecision));
  test->InitializeReference(ws_init_func, reference_args);
  test->RunReference();

  test->InitializeTestedOp(ws_init_func);
  test->Run();

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

  test->Check();
  return test;
}

void BasicCorrectnessTest(
    const NetDef& net_def,
    std::function<void(Workspace&)> ws_init_func,
    float relativePrecision) {
  Workspace w1;
  ws_init_func(w1);
  NetDef ref_net_def =
      caffe2::ReferenceImplementationRegistry::ConvertNet(net_def);
  unique_ptr<NetBase> ref_net(CreateNet(ref_net_def, &w1));
  ASSERT_TRUE(ref_net.get());
  {
    tc::CudaProfiler p;
    ASSERT_TRUE(ref_net->Run());
  }

  Workspace w2;
  ws_init_func(w2);
  unique_ptr<NetBase> net(CreateNet(net_def, &w2));
  ASSERT_TRUE(net.get());
  {
    tc::CudaProfiler p;
    ASSERT_TRUE(net->Run());
  }

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

  // try all output of all ops in original net as they are preserved
  for (const auto& op_def : net_def.op()) {
    for (auto out : op_def.output()) {
      // skip auxiliary blobs
      if (out[0] != '_') {
        CheckEqual(w1, w2, out, relativePrecision);
      }
    }
  }
}

void RunGradient(Workspace& w, const OperatorDef& def) {
  vector<GradientWrapper> g_output(def.output().size());
  for (int i = 0; i < def.output().size(); i++) {
    g_output[i].dense_ = def.output(i);
  }
  GradientOpsMeta meta = GetGradientForOp(def, g_output);
  for (auto& g_op : meta.ops_) {
    unique_ptr<OperatorBase> op_g(CreateOperator(g_op, &w));
    ASSERT_TRUE(op_g.get());
    {
      tc::CudaProfiler p;
      ASSERT_TRUE(op_g->Run());
    }
  }
}

} // namespace caffe2
