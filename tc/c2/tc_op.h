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
        mapping_options_(tc::CudaMappingOptions::makeNaiveMappingOptions()),
        grad_mapping_options_(
            tc::CudaMappingOptions::makeNaiveMappingOptions()) {
    is_backward_ = OperatorBase::GetSingleArgument<bool>("is_backward", false);
    tc_ = OperatorBase::GetSingleArgument<std::string>(
        is_backward_ ? "tc_grad_def" : "tc_def", "ERROR");
    tc_name_ = OperatorBase::GetSingleArgument<std::string>(
        is_backward_ ? "tc_grad_name" : "tc_name", "ERROR");
    compiled_ = false;
    check_sizes_ = OperatorBase::GetSingleArgument<bool>("check_sizes", false);
    ArgumentHelper args(operator_def);

    if (args.HasArgument("mapping_options")) {
      mapping_options_ = tc::CudaMappingOptions(
          args.GetSingleArgument<std::string>("mapping_options", "ERROR"));
    } else {
      SetupNaiveMappingOptions();
    }
    if (args.HasArgument("grad_mapping_options")) {
      grad_mapping_options_ = tc::CudaMappingOptions(
          args.GetSingleArgument<std::string>("grad_mapping_options", "ERROR"));
    } else {
      SetupNaiveGradMappingOptions();
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ~TcOp() override {}

 protected:
  /// Hook called when the mapping_options are not provided in the Caffe2
  /// operator arguments. Does nothing by default, derived classes can
  /// reimplement this to customize stategies.
  virtual void SetupNaiveMappingOptions() {}

  /// Hook called when the grad_mapping_options are not provided in the Caffe2
  /// operator arguments. Does nothing by default, derived classes can
  /// reimplement this to customize stategies.
  virtual void SetupNaiveGradMappingOptions() {}

  void PrepareOutputs(const std::vector<tc::TensorInfo> tensorInfos) {
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
        input_dl_tensors_.emplace_back(
            dlpack::makeDLConstTensor(this->Input(idx)));
        raw_input_dl_tensors_.push_back(input_dl_tensors_.back().get());
        input_void_ptrs_.push_back(input_dl_tensors_.back()->data);
      }

      auto out_tensor_info =
          tc::inferOutputTensorInfo(tc_, tc_name_, raw_input_dl_tensors_);
      PrepareOutputs(out_tensor_info);

      // now create the output_dl_tensors
      for (int i = 0; i < OutputSize(); ++i) {
        output_dl_tensors_.emplace_back(dlpack::makeDLTensor(*Output(i)));
        raw_output_dl_tensors_.push_back(output_dl_tensors_.back().get());
        output_void_ptrs_.push_back(output_dl_tensors_[i]->data);
      }

      // compile
      executor_ = tc::compile<tc::CudaBackend>(
          tc_,
          tc_name_,
          raw_input_dl_tensors_,
          is_backward_ ? grad_mapping_options_ : mapping_options_);
      compiled_ = true;
    }

    // run
    if (!check_sizes_) {
      executor_->uncheckedRun(input_void_ptrs_, output_void_ptrs_);
    } else {
      executor_->run(raw_input_dl_tensors_, raw_output_dl_tensors_);
    }
    return true;
  }

 protected:
  std::string tc_;
  std::string tc_name_;
  bool compiled_;
  bool check_sizes_;
  bool is_backward_;
  tc::CudaMappingOptions mapping_options_;
  tc::CudaMappingOptions grad_mapping_options_;
  // Owning DLTensor wrapping C2 tensors
  std::vector<tc::DLConstTensorUPtr> input_dl_tensors_;
  std::vector<tc::DLTensorUPtr> output_dl_tensors_;
  // Pointers into owning DLTensor wrapping C2 tensors
  std::vector<const DLConstTensor*> raw_input_dl_tensors_;
  std::vector<const DLTensor*> raw_output_dl_tensors_;
  // Pointers into unchecked void*
  std::vector<const void*> input_void_ptrs_;
  std::vector<void*> output_void_ptrs_;

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

    Argument grad_arg = MakeArgument<bool>("is_backward", true);

    grad_ops.push_back(CreateOperatorDef(
        "TcOp", "", input_vec, output_vec, std::vector<Argument>{grad_arg}));
    return grad_ops;
  }
};
} // namespace caffe2
