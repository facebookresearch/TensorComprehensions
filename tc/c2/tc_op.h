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

#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/execution_engine.h"
#include "tc/core/utils/dlpack.h"

#include "tc/c2/context.h"
#include "tc/c2/dlpack_c2.h"

#include "caffe2/core/common.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, class Engine = DefaultEngine>
class TcOp : public Operator<Context> {
 public:
  TcOp(const OperatorDef& operator_def, Workspace* ws)
      : caffe2::Operator<Context>(operator_def, ws),
        tc_(OperatorBase::GetSingleArgument<std::string>("tcDef", "ERROR")),
        tcName_(
            OperatorBase::GetSingleArgument<std::string>("tcName", "ERROR")),
        cudaMappingOptions_(tc::CudaMappingOptions::makeNaiveMappingOptions()),
        gradCudaMappingOptions_(
            tc::CudaMappingOptions::makeNaiveMappingOptions()) {
    gradTc_ =
        OperatorBase::GetSingleArgument<std::string>("tcGradDef", "ERROR");
    gradTcName_ =
        OperatorBase::GetSingleArgument<std::string>("tcGradName", "ERROR");
    checkSizes_ = OperatorBase::GetSingleArgument<bool>("checkSizes", false);
    compiled_ = false;
    handle_ = 0;
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
    executionEngine_ = std::unique_ptr<tc::ExecutionEngine<tc::CudaTcExecutor>>(
        new tc::ExecutionEngine<tc::CudaTcExecutor>());
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

  void prepareOutputs(const std::vector<const DLTensor*> tensorInfo) {
    for (size_t i = 0; i < tensorInfo.size(); ++i) {
      auto info = tensorInfo[i];
      std::vector<int64_t> shape(info->shape, info->shape + info->ndim);
      Output(i)->Resize(shape);
      // Note: this mutable_data() call actually creates the data storage.
      Output(i)->template mutable_data<T>();
    }
  }

  virtual bool RunOnDevice() override {
    if (!compiled_) {
      // first, given the TC, define it in the executionEngine_
      executionEngine_->define(tc_);
      for (int idx = 0; idx < this->InputSize(); ++idx) {
        auto dims = this->Input(idx).dims();
        inTensorUPtrs_.emplace_back(
            dlpack::makeConstDLTensor(this->Input(idx), dims));
        inputDLTensors_.push_back(inTensorUPtrs_[idx].get());
        inputVoidPtrs_.push_back(inputDLTensors_[idx]->data);
      }
      auto outTensorInfo =
          executionEngine_->inferOutputTensorInfo(tcName_, inputDLTensors_);
      prepareOutputs(outTensorInfo);
      for (int idx = 0; idx < OutputSize(); ++idx) {
        outTensorUPtrs_.emplace_back(dlpack::makeDLTensor(Output(idx)));
        outputDLTensors_.push_back(outTensorUPtrs_[idx].get());
        outputVoidPtrs_.push_back(outputDLTensors_[idx]->data);
      }
      handle_ = executionEngine_->compile(
          tcName_,
          inputDLTensors_,
          cudaMappingOptions_.toProtobufSerializedString());
      compiled_ = true;
    }

    if (checkSizes_) {
      executionEngine_->run(handle_, inputDLTensors_, outputDLTensors_);
    } else {
      executionEngine_->uncheckedRun(handle_, inputVoidPtrs_, outputVoidPtrs_);
    }

    return true;
  }

 protected:
  std::string tc_;
  std::string gradTc_;
  std::string tcName_;
  std::string gradTcName_;
  bool checkSizes_;
  bool compiled_;
  size_t handle_;
  std::vector<const void*> inputVoidPtrs_;
  std::vector<void*> outputVoidPtrs_;
  std::vector<const DLTensor*> inputDLTensors_;
  std::vector<DLTensor*> outputDLTensors_;
  std::vector<::tc::dlutils::DLTensorUPtr> inTensorUPtrs_;
  std::vector<::tc::dlutils::DLTensorUPtr> outTensorUPtrs_;
  tc::CudaMappingOptions cudaMappingOptions_;
  tc::CudaMappingOptions gradCudaMappingOptions_;

 private:
  std::unique_ptr<tc::ExecutionEngine<tc::CudaTcExecutor>> executionEngine_;
};

class GetTcOpGradient : public GradientMakerBase {
 public:
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper args(Def());
    CHECK(false) << "NYI gradient";
    return {};
  }
};
} // namespace caffe2
