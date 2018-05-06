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

namespace caffe2 {

namespace detail {

std::mutex& RNGMutex() {
  static std::mutex rng_mutex;
  return rng_mutex;
}

template <typename T>
T* NewTensor(
    caffe2::Workspace& ws,
    std::vector<caffe2::TIndex> shape,
    const std::string& name) {
  caffe2::Blob* blob = ws.CreateBlob(name);
  auto* tensor = blob->GetMutable<T>();
  tensor->Resize(shape);
  return tensor;
}

template <typename Caffe2Backend, typename T>
void AddDeterministicallyRandomInputWithRange(
    caffe2::Workspace& ws,
    const std::vector<caffe2::TIndex>& shape,
    const std::string& name,
    T min,
    T max) {
  std::lock_guard<std::mutex> lock{detail::RNGMutex()};
  DeviceOption option;
  option.set_random_seed(std::hash<std::string>()(name));
  auto context = makeContext<Caffe2Backend>(option);
  auto* tensor =
      detail::NewTensor<typename Caffe2Backend::Tensor>(ws, shape, name);
  caffe2::math::RandUniform<T, typename Caffe2Backend::Context>(
      tensor->size(),
      min,
      max,
      tensor->template mutable_data<T>(),
      context.get());
  context->FinishDeviceComputation();
}

} // namespace detail

template <typename Caffe2Backend>
std::unique_ptr<typename Caffe2Backend::Context> makeContext(
    caffe2::DeviceOption opt) {
  opt.set_device_type(Caffe2Backend::Device);
  return std::unique_ptr<typename Caffe2Backend::Context>(
      new typename Caffe2Backend::Context(opt));
}

template <typename Caffe2Backend>
caffe2::Tensor<typename Caffe2Backend::Context> getNamedTensor(
    caffe2::Workspace& ws,
    const std::string& name) {
  // Resolved dynamically
  caffe2::Tensor<typename Caffe2Backend::Context> Texpected(
      ws.GetBlob(name)->Get<typename Caffe2Backend::Tensor>());
  return Texpected;
}

// helper functions to construct an ATen tensor from a caffe2 tensor
template <typename Caffe2TensorType>
at::Tensor makeATenTensor(
    const Caffe2TensorType& c2Tensor,
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

template <
    class IterableInputs = std::initializer_list<string>,
    class IterableOutputs = std::initializer_list<string>,
    class IterableArgs = std::initializer_list<Argument>>
OperatorDef Configure(
    std::string type,
    IterableInputs ins,
    IterableOutputs outs,
    IterableArgs args,
    caffe2::DeviceType dtype) {
  OperatorDef def = CreateOperatorDef(type, "", ins, outs, args);
  def.mutable_device_option()->set_device_type(dtype);
  return def;
}

template <
    class IterableInputs = std::initializer_list<string>,
    class IterableOutputs = std::initializer_list<string>,
    class IterableArgs = std::initializer_list<Argument>>
OperatorDef ConfigureCUDA(
    std::string type,
    IterableInputs ins,
    IterableOutputs outs,
    IterableArgs args) {
  return Configure(type, ins, outs, args, caffe2::CUDABackend::Device);
}

template <typename Caffe2Backend, typename T>
void AddConstInput(
    caffe2::Workspace& ws,
    const std::vector<caffe2::TIndex>& shape,
    const T value,
    const std::string& name) {
  auto context = makeContext<Caffe2Backend>();
  auto* tensor =
      detail::NewTensor<typename Caffe2Backend::Tensor>(ws, shape, name);
  caffe2::math::Set<T, typename Caffe2Backend::Context>(
      tensor->size(), value, tensor->template mutable_data<T>(), context.get());
  context->FinishDeviceComputation();
}

template <typename Caffe2Backend, typename T>
void AddDeterministicallyRandomInput(
    caffe2::Workspace& ws,
    const std::vector<caffe2::TIndex>& shape,
    const std::string& name) {
  // 0..2 seems like a nice range for weights
  detail::AddDeterministicallyRandomInputWithRange<Caffe2Backend, T>(
      ws, shape, name, 0, 2);
}

template <
    typename Caffe2SourceBackend,
    typename Caffe2DestinationBackend,
    typename T>
void AddCopyOfTensor(
    caffe2::Workspace& ws,
    const std::string& name,
    const caffe2::Workspace& sourceWs,
    const std::string& sourceName) {
  auto sourceContext = makeContext<Caffe2SourceBackend>();
  auto destinationContext = makeContext<Caffe2DestinationBackend>();
  const auto& sourceTensor =
      sourceWs.GetBlob(sourceName)->Get<typename Caffe2SourceBackend::Tensor>();
  auto* destinationTensor =
      detail::NewTensor<typename Caffe2DestinationBackend::Tensor>(
          ws, sourceTensor.dims(), name);
  destinationTensor->CopyFrom(sourceTensor);
  sourceContext->FinishDeviceComputation();
  destinationContext->FinishDeviceComputation();
}

template <typename T>
void CheckEqual(
    const caffe2::Workspace& expected,
    const caffe2::Workspace& tested,
    const std::string& name,
    float relativePrecision,
    long offsetInExpected,
    long offsetInTested) {
  // Resolved dynamically
  caffe2::CPUBackend::Tensor Texpected(expected.GetBlob(name)->Get<T>());
  caffe2::CPUBackend::Tensor Ttested(tested.GetBlob(name)->Get<T>());
  CheckEqual(
      Texpected, Ttested, relativePrecision, offsetInExpected, offsetInTested);
}

template <typename Backend>
void BasicGradientCorrectnessTest(
    const OperatorDef& op_def,
    std::function<void(Workspace&)> ws_init_func,
    float relativePrecision,
    const std::vector<std::string>& names_to_compare,
    std::map<string, int> params,
    ReferenceImplementationBuilder make_reference_impl) {
  // Reference implementation runs on a first workspace initialized with
  // random tensors, in a reproducible fashion
  Workspace w1;
  ws_init_func(w1);
  NetDef net_def;
  make_reference_impl(op_def, &net_def);
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

  // TC implementation runs on a second workspace initialized with
  // random tensors, in a reproducible fashion
  Workspace w2;
  ws_init_func(w2);
  unique_ptr<OperatorBase> op(CreateOperator(op_def, &w2));
  ASSERT_TRUE(op.get());
  {
    tc::CudaProfiler p;
    ASSERT_TRUE(op->Run());
  }
  OperatorDef def = op_def;
  RunGradient(w2, def);

  for (const auto& n : names_to_compare) {
    CheckEqual(
        CPUBackend::Tensor(getNamedTensor<Backend>(w1, n)),
        CPUBackend::Tensor(getNamedTensor<Backend>(w2, n)),
        relativePrecision);
  }
}

} // namespace caffe2
