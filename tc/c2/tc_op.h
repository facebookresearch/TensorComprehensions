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

#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "tc/core/compiler.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/tensor.h"

#include "tc/c2/context.h"
#include "tc/c2/dlpack_c2.h"

#include "caffe2/core/common.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, class Engine = DefaultEngine>
class TcOp : public Operator<Context> {
 public:
  TcOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        cudaMappingOptions_(tc::CudaMappingOptions::makeNaiveMappingOptions()),
        gradCudaMappingOptions_(
            tc::CudaMappingOptions::makeNaiveMappingOptions()) {
    is_backward_ = OperatorBase::GetSingleArgument<bool>("is_backward_", false);
    tc_ = OperatorBase::GetSingleArgument<std::string>(
        is_backward_ ? "tcGradDef" : "tcDef", "ERROR");
    tcName_ = OperatorBase::GetSingleArgument<std::string>(
        is_backward_ ? "tcGradName" : "tcName", "ERROR");
    compiled_ = false;
    checkSizes_ = OperatorBase::GetSingleArgument<bool>("checkSizes", false);
    ArgumentHelper args(operator_def);

    if (args.HasArgument("mappingOptions")) {
      cudaMappingOptions_ = tc::CudaMappingOptions(
          args.GetSingleArgument<std::string>("mappingOptions", "ERROR"));
    } else {
      setupNaiveCudaMappingOptions();
    }

    if (args.HasArgument("gradCudaMappingOptions")) {
      gradCudaMappingOptions_ =
          tc::CudaMappingOptions(args.GetSingleArgument<std::string>(
              "gradCudaMappingOptions", "ERROR"));
    } else {
      setupDefaultGradCudaMappingOptions();
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ~TcOp() override {}

 protected:
  /// Hook called when the mappingOptions are not provided in the Caffe2
  /// operator arguments. Does nothing by default, derived classes can
  /// reimplement this to customize stategies.
  virtual void setupNaiveCudaMappingOptions() {}

  /// Hook called when the gradCudaMappingOptions are not provided in the Caffe2
  /// operator arguments. Does nothing by default, derived classes can
  /// reimplement this to customize stategies.
  virtual void setupDefaultGradCudaMappingOptions() {}

  void prepareOutputs(const std::vector<tc::TensorInfo> tensorInfos) {
    for (size_t i = 0; i < tensorInfos.size(); ++i) {
      auto info = tensorInfos[i];
      Output(i)->Resize(info.shape);
      // Note: this mutable_data() call actually creates the data storage.
      Output(i)->template mutable_data<T>();
    }
  }

  virtual bool RunOnDevice() override {
    if (!compiled_) {
      // now, given the input tensors, convert them to dlpack tensors so that
      // we can call the compile command
      for (int idx = 0; idx < this->InputSize(); ++idx) {
        inputDLTensors_.emplace_back(
            dlpack::makeDLConstTensor(this->Input(idx)));
        rawInputDLTensors_.push_back(inputDLTensors_.back().get());
        inputVoidPtrs_.push_back(inputDLTensors_.back()->data);
      }

      auto parsedTcs = tc::detail::parse(tc_);
      CHECK_EQ(parsedTcs.count(tcName_), 1u)
          << "attempting to access undefined function " << tcName_;

      auto outTensorInfo = tc::detail::inferOutputTensorInfo(
          parsedTcs.at(tcName_), rawInputDLTensors_);
      prepareOutputs(outTensorInfo);

      // now create the outputDLTensors
      for (int i = 0; i < OutputSize(); ++i) {
        outputDLTensors_.emplace_back(dlpack::makeDLTensor(*Output(i)));
        rawOutputDLTensors_.push_back(outputDLTensors_.back().get());
        outputVoidPtrs_.push_back(outputDLTensors_[i]->data);
      }

      // compile
      executor_ = tc::compile<tc::CudaBackend>(
          parsedTcs.at(tcName_),
          rawInputDLTensors_,
          is_backward_ ? gradCudaMappingOptions_.toProtobufSerializedString()
                       : cudaMappingOptions_.toProtobufSerializedString());
      compiled_ = true;
    }

    // run
    if (!checkSizes_) {
      executor_->uncheckedRun(inputVoidPtrs_, outputVoidPtrs_);
    } else {
      executor_->run(rawInputDLTensors_, rawOutputDLTensors_);
    }
    return true;
  }

 protected:
  std::string tc_;
  std::string tcName_;
  bool compiled_;
  bool checkSizes_;
  bool is_backward_;
  tc::CudaMappingOptions cudaMappingOptions_;
  tc::CudaMappingOptions gradCudaMappingOptions_;
  // Owning DLTensor wrapping C2 tensors
  std::vector<tc::DLConstTensorUPtr> inputDLTensors_;
  std::vector<tc::DLTensorUPtr> outputDLTensors_;
  // Pointers into owning DLTensor wrapping C2 tensors
  std::vector<const DLConstTensor*> rawInputDLTensors_;
  std::vector<const DLTensor*> rawOutputDLTensors_;
  // Pointers into unchecked void*
  std::vector<const void*> inputVoidPtrs_;
  std::vector<void*> outputVoidPtrs_;

  std::unique_ptr<tc::CudaBackend::ExecutorType> executor_;
};

class GetTcOpGradient : public GradientMakerBase {
 public:
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper args(Def());
    vector<OperatorDef> grad_ops;

    std::vector<string> input_vec, output_vec;

    // First input: inputs to be used in TC Op Gradient
    for (int idx : args.GetRepeatedArgument<int>("inputs_used_by_gradient")) {
      input_vec.push_back(I(idx));
    }

    // Second input: outputs to be used in TC Op Gradient
    for (int idx : args.GetRepeatedArgument<int>("outputs_used_by_gradient")) {
      input_vec.push_back(O(idx));
    }

    // Third input: Gradient-of-outputs to be used in TC Op Gradient
    for (int idx :
         args.GetRepeatedArgument<int>("output_gradients_used_by_gradient")) {
      input_vec.push_back(GO(idx));
    }

    // Output: calculated output from TC Op Gradient
    for (int idx :
         args.GetRepeatedArgument<int>("inputs_to_compute_gradients_of")) {
      output_vec.push_back(GI(idx));
    }

    Argument grad_arg = MakeArgument<bool>("is_backward_", true);

    grad_ops.push_back(CreateOperatorDef(
        "TcOp", "", input_vec, output_vec, std::vector<Argument>{grad_arg}));
    return grad_ops;
  }
};
} // namespace caffe2
