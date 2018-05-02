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
#pragma once

#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <vector>

#include "tc/aten/aten.h"

#include "caffe2/core/common.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/net_simple.h"

#include "tc/c2/operator_meta.h"
#include "tc/c2/tc_op.h"
#include "tc/core/cuda/cuda.h"

namespace caffe2 {

caffe2::TensorCPU context2tensor(caffe2::CPUContext& ctx) {
  return caffe2::TensorCPU();
}

caffe2::TensorCUDA context2tensor(caffe2::CUDAContext& ctx) {
  return caffe2::TensorCUDA();
}

template <typename T>
std::unique_ptr<T> makeContext(
    caffe2::DeviceOption opt = caffe2::DeviceOption());

template <>
std::unique_ptr<caffe2::CPUContext> makeContext(caffe2::DeviceOption opt) {
  opt.set_device_type(caffe2::DeviceType::CPU);
  return std::unique_ptr<caffe2::CPUContext>(new caffe2::CPUContext(opt));
}

template <>
std::unique_ptr<caffe2::CUDAContext> makeContext(caffe2::DeviceOption opt) {
  opt.set_device_type(caffe2::DeviceType::CUDA);
  return std::unique_ptr<caffe2::CUDAContext>(new caffe2::CUDAContext(opt));
}

caffe2::Tensor<caffe2::CPUContext> getReferenceHostBlob(
    caffe2::Workspace& ws,
    const std::string& name) {
  // Resolved dynamically
  caffe2::Tensor<caffe2::CPUContext> Texpected(
      ws.GetBlob(name)->Get<caffe2::TensorCUDA>());
  return Texpected;
}

caffe2::Tensor<caffe2::CUDAContext> getReferenceDeviceBlob(
    caffe2::Workspace& ws,
    const std::string& name) {
  caffe2::Tensor<caffe2::CUDAContext> Texpected(
      ws.GetBlob(name)->Get<caffe2::TensorCUDA>());
  return Texpected;
}

// helper functions to construct an ATen tensor from a caffe2 tensor
template <typename C2Context>
at::Tensor makeATenTensor(
    const caffe2::Tensor<C2Context>& c2Tensor,
    at::Backend backend,
    at::ScalarType stype) {
  auto dims = c2Tensor.dims();
  auto ndim = dims.size();
  auto shape = new int64_t[ndim];
  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = dims[i];
  }
  at::Tensor out =
      at::getType(backend, stype)
          .tensorFromBlob(
              const_cast<void*>(c2Tensor.raw_data()), at::IntList(shape, ndim));
  return out;
}

namespace {
std::mutex rng_mutex;
}

struct TestHarness {
  template <
      class IterableInputs = std::initializer_list<string>,
      class IterableOutputs = std::initializer_list<string>,
      class IterableArgs = std::initializer_list<Argument>>
  static OperatorDef Configure(
      std::string type,
      IterableInputs ins,
      IterableOutputs outs,
      IterableArgs args = {},
      caffe2::DeviceType dtype = caffe2::DeviceType::CPU) {
    OperatorDef def = CreateOperatorDef(type, "", ins, outs, args);
    def.mutable_device_option()->set_device_type(dtype);
    return def;
  }

  template <
      class IterableInputs = std::initializer_list<string>,
      class IterableOutputs = std::initializer_list<string>,
      class IterableArgs = std::initializer_list<Argument>>
  static OperatorDef ConfigureCUDA(
      std::string type,
      IterableInputs ins,
      IterableOutputs outs,
      IterableArgs args = {}) {
    return TestHarness::Configure(
        type, ins, outs, args, caffe2::DeviceType::CUDA);
  }

  static DeviceOption getCUDADevice() {
    DeviceOption device;
    device.set_device_type(caffe2::DeviceType::CUDA);
    return device;
  }

  template <typename T>
  static T* NewTensor(
      caffe2::Workspace& ws,
      std::vector<caffe2::TIndex> shape,
      const std::string& name) {
    caffe2::Blob* blob = ws.CreateBlob(name);
    auto* tensor = blob->GetMutable<T>();
    tensor->Resize(shape);
    return tensor;
  }

  template <typename T, typename Context>
  static void AddConstInput(
      caffe2::Workspace& ws,
      const std::vector<caffe2::TIndex>& shape,
      const T value,
      const std::string& name) {
    auto context = makeContext<Context>();
    using TensorType = decltype(context2tensor(*context));
    auto* tensor = TestHarness::NewTensor<TensorType>(ws, shape, name);
    caffe2::math::Set<T, Context>(
        tensor->size(),
        value,
        tensor->template mutable_data<T>(),
        context.get());
    context->FinishDeviceComputation();
  }

  // May need copies because RNG on CPU and GPU do not produce the same
  // values when initialized with the same seed.
  template <typename T, typename DestinationContext, typename SourceContext>
  static void AddCopyOfTensor(
      caffe2::Workspace& ws,
      const std::string& name,
      const caffe2::Workspace& sourceWs,
      const std::string& sourceName) {
    auto destinationContext = makeContext<DestinationContext>();
    using DestinationTensorType = Tensor<DestinationContext>;
    auto sourceContext = makeContext<SourceContext>();
    using SourceTensorType = decltype(context2tensor(*sourceContext));

    const auto& sourceTensor =
        sourceWs.GetBlob(sourceName)->Get<SourceTensorType>();
    auto* destinationTensor = TestHarness::NewTensor<DestinationTensorType>(
        ws, sourceTensor.dims(), name);
    destinationTensor->CopyFrom(sourceTensor);
    sourceContext->FinishDeviceComputation();
    destinationContext->FinishDeviceComputation();
  }

  template <typename T, typename Context>
  static void AddDeterministicallyRandomInputWithRange(
      caffe2::Workspace& ws,
      const std::vector<caffe2::TIndex>& shape,
      const std::string& name,
      T min,
      T max) {
    std::lock_guard<std::mutex> lock{rng_mutex};
    DeviceOption option;
    option.set_random_seed(std::hash<std::string>()(name));
    auto context = makeContext<Context>(option);
    using TensorType = decltype(context2tensor(*context));
    auto* tensor = TestHarness::NewTensor<TensorType>(ws, shape, name);
    caffe2::math::RandUniform<T, Context>(
        tensor->size(),
        min,
        max,
        tensor->template mutable_data<T>(),
        context.get());
    context->FinishDeviceComputation();
  }

  template <typename T, typename Context>
  static void AddDeterministicallyRandomInput(
      caffe2::Workspace& ws,
      const std::vector<caffe2::TIndex>& shape,
      const std::string& name) {
    // 0..2 seems like a nice range for weights
    AddDeterministicallyRandomInputWithRange<T, Context>(ws, shape, name, 0, 2);
  }

  template <typename T>
  static void CheckMatMulOfOnesOnCPU(caffe2::Blob* b, int m, int n, int k) {
    auto& C = b->Get<T>();
    auto cpuC = caffe2::Tensor<caffe2::CPUContext>(C);
    assert(cpuC.size() == m * n);
    for (int i = 0; i < cpuC.size(); ++i) {
      ASSERT_EQ(cpuC.data<float>()[i], static_cast<float>(k));
    }
  }

  static void PrintDualTensor(
      caffe2::Workspace& expected,
      caffe2::Workspace& actual,
      std::string name,
      float relativePrecision = 0.0) {
    // Resolved dynamically
    caffe2::Tensor<caffe2::CPUContext> Texpected(
        expected.GetBlob(name)->Get<caffe2::TensorCUDA>());
    caffe2::Tensor<caffe2::CPUContext> Tactual(
        actual.GetBlob(name)->Get<caffe2::TensorCUDA>());
    for (int i = 0; i < Texpected.size(); ++i) {
      LOG(INFO) << name << "[" << i << "] | E=" << Texpected.data<float>()[i]
                << " \tA=" << Tactual.data<float>()[i] << "\n";
    }
  }

  template <typename T = caffe2::TensorCUDA>
  static void CheckEqual(
      const caffe2::Workspace& expected,
      const caffe2::Workspace& actual,
      const std::string& name,
      float relativePrecision = 0.0,
      long offsetInExpected = 0,
      long offsetInActual = 0) {
    // Resolved dynamically
    caffe2::Tensor<caffe2::CPUContext> Texpected(
        expected.GetBlob(name)->Get<T>());
    caffe2::Tensor<caffe2::CPUContext> Tactual(actual.GetBlob(name)->Get<T>());
    for (int i = 0; i < Texpected.size() - offsetInExpected; ++i) {
      if (relativePrecision == 0.0) {
        ASSERT_FLOAT_EQ(
            Texpected.data<float>()[i + offsetInExpected],
            Tactual.data<float>()[i + offsetInActual])
            << " for Blob " << name << " at position " << i;
      } else {
        // From glog's glog/src/glog/logging.h.in
        // #define CHECK_NEAR(val1, val2, margin)
        // CHECK_NEAR is actualy absolute!!!
        ASSERT_NEAR(
            Texpected.data<float>()[i + offsetInExpected],
            Tactual.data<float>()[i + offsetInActual],
            relativePrecision * Texpected.data<float>()[i + offsetInExpected])
            << " for Blob " << name << " at position " << i;
      }
    }
  }

  class OpTester {
    std::unique_ptr<NetBase> net_ref;
    OperatorDef op_def;
    float relativePrecision;

   public:
    Workspace w_ref;
    Workspace w_test;
    unique_ptr<OperatorBase> op_test;

    OpTester(const OperatorDef& op_def, float relativePrecision = 0.0)
        : op_def{op_def}, relativePrecision{relativePrecision} {}

    void InitializeReference(
        std::function<void(Workspace&)> ws_init_func,
        std::map<string, int> reference_args = {}) {
      ws_init_func(w_ref);
      NetDef net_def;
      caffe2::ReferenceImplementationRegistry::Append(&net_def, op_def);
      for (auto s : reference_args) {
        auto arg = net_def.mutable_op()->Mutable(0)->add_arg();
        arg->set_name(s.first);
        arg->set_i(s.second);
      }
      net_ref = CreateNet(net_def, &w_ref);
    }

    void RunReference() {
      ASSERT_TRUE(net_ref.get());
      tc::CudaProfiler p;
      ASSERT_TRUE(net_ref->Run());
    }

    void InitializeTestedOp(std::function<void(Workspace&)> ws_init_func) {
      ws_init_func(w_test);
      op_test = CreateOperator(op_def, &w_test);
    }

    void Run() {
      ASSERT_TRUE(op_test.get());
      tc::CudaProfiler p;
      ASSERT_TRUE(op_test->Run());
    }

    void Check() const {
      for (auto out : op_def.output()) {
        TestHarness::CheckEqual(w_ref, w_test, out, relativePrecision);
      }
    }

    caffe2::Tensor<caffe2::CPUContext> getReferenceHostBlob(
        const std::string& name) {
      // Resolved dynamically
      caffe2::Tensor<caffe2::CPUContext> Texpected(
          w_ref.GetBlob(name)->Get<caffe2::TensorCUDA>());
      return Texpected;
    }

    caffe2::Tensor<caffe2::CUDAContext> getReferenceDeviceBlob(
        const std::string& name) {
      caffe2::Tensor<caffe2::CUDAContext> Texpected(
          w_ref.GetBlob(name)->Get<caffe2::TensorCUDA>());
      return Texpected;
    }

    void RunAllAndCheck() {
      RunReference();
      Run();
      Check();
    }

    // XXX:stupid gtest macros return void because
    // google doesn't like exceptions
    void GetTcOp(TcOp<float, CUDAContext>** op) {
      *op = dynamic_cast<TcOp<float, CUDAContext>*>(op_test.get());
      ASSERT_NE(*op, nullptr);
    }

    bool cacheRetrievalSuccessful() {
      return false;
      // TODO(ttheodor) FIXME
      // TcOp<float, CUDAContext>* op;
      // GetTcOp(&op);
      // return op->LastCacheRetrievalSuccessful();
    }
  };

  // Compares individual operator
  static unique_ptr<OpTester> BasicCorrectnessTest(
      const OperatorDef& op_def,
      std::function<void(Workspace&)> ws_init_func,
      float relativePrecision = 0.0,
      std::map<string, int> reference_args = {},
      bool check = true) {
    unique_ptr<OpTester> test(new OpTester(op_def, relativePrecision));
    test->InitializeReference(ws_init_func, reference_args);
    test->RunReference();

    test->InitializeTestedOp(ws_init_func);
    test->Run();

    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

    if (check) {
      test->Check();
    }

    return test;
  }

  static void RunGradient(Workspace& w, const OperatorDef& def) {
    vector<GradientWrapper> g_output(def.output().size());
    for (int i = 0; i < def.output().size(); i++) {
      g_output[i].dense_ = def.output(i) + "_grad";
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

  // Compares the entire Net and all intermediate blobs
  static void BasicCorrectnessTest(
      const NetDef& net_def,
      std::function<void(Workspace&)> ws_init_func,
      float relativePrecision = 0.0) {
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
          TestHarness::CheckEqual(w1, w2, out, relativePrecision);
        }
      }
    }
  }

  static void BasicGradientCorrectnessTest(
      const OperatorDef& op_def,
      std::function<void(Workspace&)> ws_init_func,
      std::map<string, int> params = {},
      bool check = true) {
    Workspace w1;
    ws_init_func(w1);
    NetDef net_def;
    caffe2::ReferenceImplementationRegistry::Append(&net_def, op_def);
    for (auto s : params) {
      auto arg = net_def.mutable_op()->Mutable(0)->add_arg();
      arg->set_name(s.first);
      arg->set_i(s.second);
    }
    unique_ptr<NetBase> net(CreateNet(net_def, &w1));
    ASSERT_TRUE(net.get());
    {
      tc::CudaProfiler p;
      ASSERT_TRUE(net->Run());
    }
    RunGradient(w1, *net_def.mutable_op()->Mutable(0));

    Workspace w2;
    ws_init_func(w2);
    unique_ptr<OperatorBase> op(CreateOperator(op_def, &w2));
    ASSERT_TRUE(op.get());
    {
      tc::CudaProfiler p;
      ASSERT_TRUE(op->Run());
    }
    OperatorDef def = op_def;
    // auto pOp = static_cast<TcOp<float,
    // caffe2::CUDAContext>*>(op.get());
    // The following three lines flush any changes
    //  done by calling functions on pOp (i.e. pOp->setIslGrad)
    //  such that this information is available to the operatordef
    //  and thus the gradient creation
    CHECK(false) << "NYI: C2 gradients are not supported atm, FIXME!";
    // caffe2::AddArgument("strategy", // pOp->serializeIsl()
    //                     , &def);
    // caffe2::AddArgument("grad_strategy", // pOp->serializeIslGrad()
    //                     , &def);
    RunGradient(w2, def);

    if (check) {
      for (auto out : op_def.input()) {
        TestHarness::CheckEqual(w1, w2, out + "_grad");
      }
    }
  }
};

} // namespace caffe2
