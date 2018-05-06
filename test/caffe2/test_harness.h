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

#include "cuda/test_harness.h"

namespace caffe2 {

// CPUBackend is always used and the source of truth for performing checks
struct CPUBackend {
  static constexpr auto Device = DeviceType::CPU;
  using Context = CPUContext;
  using Tensor = TensorCPU;
};

/// Make a context for the proper Backend type.
/// A DeviceOption may be passed (e.g. to set the random seed).
template <typename Caffe2Backend>
std::unique_ptr<typename Caffe2Backend::Context> makeContext(
    caffe2::DeviceOption opt = DeviceOption());

/// This function retrieves a Caffe2 tensor of the proper backend type
/// from a workspace. The lookup is done by the underlying Blob name in the
/// workspace.
/// The backend type ***must match*** the underlying Blob type because the
/// Blob.Get method is templated and performs correctness checks ar runtime.
/// This function is used for testing purposes, we do not worry about
/// const correctness for now.
template <typename Caffe2Backend>
caffe2::Tensor<typename Caffe2Backend::Context> getNamedTensor(
    caffe2::Workspace& ws,
    const std::string& name);

// helper functions to construct an ATen tensor from a caffe2 tensor
template <typename Caffe2TensorType>
at::Tensor makeATenTensor(
    const Caffe2TensorType& c2Tensor,
    at::Backend backend,
    at::ScalarType stype);

/// We need to provide a way to perform correctness checks on gradients
/// using existing Caffe2 operators.
///
/// The default reference implementation builder can be obtained by calling
/// MakeDefaultReferenceImplementationworks for Caffe2 operators whose
/// gradient reference implementation has been registered properly
/// (in the ReferenceImplementationRegistry). Such operators usually are
/// named TcOpCaffe2OpName (e.g. TcOpMatMul).
///
/// In the case of the generic TcOp, this is not possible because there is
/// no such thing as generic matching of a TC function to a Caffe2 operator
/// (at least not for now). Therefore we need to provide a way to construct
/// a reference implementation for generic TcOp instances.
///
/// This is the purpose of the ReferenceImplementationBuilder.
/// For properly registered TcOp, one can use the default
/// MakeDefaultReferenceImplementationBuilder()
using ReferenceImplementationBuilder =
    std::function<void(const OperatorDef& op_def, NetDef* net_def)>;

ReferenceImplementationBuilder MakeDefaultReferenceImplementationBuilder() {
  return [](const OperatorDef& op_def, NetDef* net_def) {
    caffe2::ReferenceImplementationRegistry::Append(net_def, op_def);
  };
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
      caffe2::DeviceType dtype = caffe2::CPUBackend::Device) {
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
        type, ins, outs, args, caffe2::CUDABackend::Device);
  }

  static DeviceOption getCUDADevice() {
    DeviceOption device;
    device.set_device_type(caffe2::CUDABackend::Device);
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

  template <typename Caffe2Backend, typename T>
  static void AddConstInput(
      caffe2::Workspace& ws,
      const std::vector<caffe2::TIndex>& shape,
      const T value,
      const std::string& name) {
    auto context = makeContext<Caffe2Backend>();
    auto* tensor =
        TestHarness::NewTensor<typename Caffe2Backend::Tensor>(ws, shape, name);
    caffe2::math::Set<T, typename Caffe2Backend::Context>(
        tensor->size(),
        value,
        tensor->template mutable_data<T>(),
        context.get());
    context->FinishDeviceComputation();
  }

  // May need copies because RNG on CPU and GPU do not produce the same
  // values when initialized with the same seed.
  template <
      typename Caffe2SourceBackend,
      typename Caffe2DestinationBackend,
      typename T>
  static void AddCopyOfTensor(
      caffe2::Workspace& ws,
      const std::string& name,
      const caffe2::Workspace& sourceWs,
      const std::string& sourceName) {
    auto sourceContext = makeContext<Caffe2SourceBackend>();
    auto destinationContext = makeContext<Caffe2DestinationBackend>();
    const auto& sourceTensor =
        sourceWs.GetBlob(sourceName)
            ->Get<typename Caffe2SourceBackend::Tensor>();
    auto* destinationTensor =
        TestHarness::NewTensor<typename Caffe2DestinationBackend::Tensor>(
            ws, sourceTensor.dims(), name);
    destinationTensor->CopyFrom(sourceTensor);
    sourceContext->FinishDeviceComputation();
    destinationContext->FinishDeviceComputation();
  }

  template <typename Caffe2Backend, typename T>
  static void AddDeterministicallyRandomInputWithRange(
      caffe2::Workspace& ws,
      const std::vector<caffe2::TIndex>& shape,
      const std::string& name,
      T min,
      T max) {
    std::lock_guard<std::mutex> lock{rng_mutex};
    DeviceOption option;
    option.set_random_seed(std::hash<std::string>()(name));
    auto context = makeContext<Caffe2Backend>(option);
    auto* tensor =
        TestHarness::NewTensor<typename Caffe2Backend::Tensor>(ws, shape, name);
    caffe2::math::RandUniform<T, typename Caffe2Backend::Context>(
        tensor->size(),
        min,
        max,
        tensor->template mutable_data<T>(),
        context.get());
    context->FinishDeviceComputation();
  }

  template <typename Caffe2Backend, typename T>
  static void AddDeterministicallyRandomInput(
      caffe2::Workspace& ws,
      const std::vector<caffe2::TIndex>& shape,
      const std::string& name) {
    // 0..2 seems like a nice range for weights
    AddDeterministicallyRandomInputWithRange<Caffe2Backend, T>(
        ws, shape, name, 0, 2);
  }

  template <typename T>
  static void CheckMatMulOfOnesOnCPU(caffe2::Blob* b, int m, int n, int k) {
    auto& C = b->Get<T>();
    auto cpuC = caffe2::Tensor<caffe2::CPUBackend::Context>(C);
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
    caffe2::Tensor<caffe2::CPUBackend::Context> Texpected(
        expected.GetBlob(name)->Get<caffe2::CUDABackend::Tensor>());
    caffe2::Tensor<caffe2::CPUBackend::Context> Tactual(
        actual.GetBlob(name)->Get<caffe2::CUDABackend::Tensor>());
    for (int i = 0; i < Texpected.size(); ++i) {
      LOG(INFO) << name << "[" << i << "] | E=" << Texpected.data<float>()[i]
                << " \tA=" << Tactual.data<float>()[i] << "\n";
    }
  }

  static void CheckEqual(
      const caffe2::Tensor<caffe2::CPUBackend::Context>& Texpected,
      const caffe2::Tensor<caffe2::CPUBackend::Context>& Tactual,
      float relativePrecision = 0.0,
      long offsetInExpected = 0,
      long offsetInActual = 0) {
    for (int i = 0; i < Texpected.size() - offsetInExpected; ++i) {
      if (relativePrecision == 0.0) {
        ASSERT_FLOAT_EQ(
            Texpected.data<float>()[i + offsetInExpected],
            Tactual.data<float>()[i + offsetInActual])
            << " for Tensor " << Texpected.DebugString() << " at position "
            << i;
      } else {
        // From glog's glog/src/glog/logging.h.in
        // #define CHECK_NEAR(val1, val2, margin)
        // CHECK_NEAR is actualy absolute!!!
        ASSERT_NEAR(
            Texpected.data<float>()[i + offsetInExpected],
            Tactual.data<float>()[i + offsetInActual],
            relativePrecision * Texpected.data<float>()[i + offsetInExpected])
            << " for Tensor " << Texpected.DebugString() << " at position "
            << i;
      }
    }
  }

  template <typename T = caffe2::CUDABackend::Tensor>
  static void CheckEqual(
      const caffe2::Workspace& expected,
      const caffe2::Workspace& actual,
      const std::string& name,
      float relativePrecision = 0.0,
      long offsetInExpected = 0,
      long offsetInActual = 0) {
    // Resolved dynamically
    caffe2::CPUBackend::Tensor Texpected(expected.GetBlob(name)->Get<T>());
    caffe2::CPUBackend::Tensor Tactual(actual.GetBlob(name)->Get<T>());
    CheckEqual(
        Texpected,
        Tactual,
        relativePrecision,
        offsetInExpected,
        offsetInActual);
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

    void RunAllAndCheck() {
      RunReference();
      Run();
      Check();
    }

    // XXX:stupid gtest macros return void because
    // google doesn't like exceptions
    void GetTcOp(TcOp<float, caffe2::CUDABackend::Context>** op) {
      *op = dynamic_cast<TcOp<float, caffe2::CUDABackend::Context>*>(
          op_test.get());
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

  /// This function runs forward and gradient for op_def (the actual operator
  /// we want to compare against a reference) and for the reference
  /// implementation.
  /// Then it compares named tensors from both the reference and actual
  /// workspace to check correctness.
  ///
  /// op_def is the OperatorDef corresponding to the operator we wish to check
  ///        for correctness
  /// ws_init_func is a function to initialize both the reference and actual
  ///        workspaces with
  /// params is a map containing constexpr values for operator specific
  ///        parameters (e.g. strides for convolutions)
  /// names_to_compare contains the names of the tensors that will be compared
  ///        after the gradient is run. Note tht Caffe2 seems to append the
  ///        _grad suffix to input tensors. For instance the gradient of
  ///        tensor I is I_grad. While unsatisfactory from a static robustness
  ///        perspective, it should be enough for testing
  /// make_reference_impl is a function that builds the reference
  ///        implementation to compare against (see the destription of the
  ///        type MakeDefaultReferenceImplementationBuilder above)
  template <typename Backend>
  static void BasicGradientCorrectnessTest(
      const OperatorDef& op_def,
      std::function<void(Workspace&)> ws_init_func,
      float relativePrecision = 0.0,
      const std::vector<std::string>& names_to_compare = {},
      std::map<string, int> params = {},
      ReferenceImplementationBuilder make_reference_impl =
          MakeDefaultReferenceImplementationBuilder());
};

} // namespace caffe2

#include "test_harness-inl.h"
