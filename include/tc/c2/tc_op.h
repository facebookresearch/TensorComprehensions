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
        mappingOptions_(tc::CudaMappingOptions::makeNaiveCudaMappingOptions()),
        gradCudaMappingOptions_(
            tc::CudaMappingOptions::makeNaiveCudaMappingOptions()) {
    gradTc_ =
        OperatorBase::GetSingleArgument<std::string>("tcGradDef", "ERROR");
    gradTcName_ =
        OperatorBase::GetSingleArgument<std::string>("tcGradName", "ERROR");
    profile_ = OperatorBase::GetSingleArgument<bool>("profile", false);
    ArgumentHelper args(operator_def);
    if (args.HasArgument("mappingOptions")) {
      mappingOptions_ = tc::CudaMappingOptions(
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
    for (int i = 0; i < tensorInfo.size(); ++i) {
      auto info = tensorInfo[i];
      std::vector<int64_t> shape(info->shape, info->shape + info->ndim);
      Output(i)->Resize(shape);
      // Note: this mutable_data() call actually creates the data storage.
      Output(i)->template mutable_data<T>();
    }
  }

  virtual bool RunOnDevice() override {
    // first, given the TC, define it in the executionEngine_
    executionEngine_->define(tc_);

    // now, given the input tensors, convert them to dlpack tensors so that
    // we can call the compile command
    std::vector<::tc::dlutils::DLTensorUPtr> inTensorUPtrs;
    std::vector<const DLTensor*> inputDLTensors;
    for (int idx = 0; idx < this->InputSize(); ++idx) {
      auto dims = this->Input(idx).dims();
      inTensorUPtrs.emplace_back(
          dlpack::makeConstDLTensor(this->Input(idx), dims));
      inputDLTensors.push_back(inTensorUPtrs.back().get());
    }

    auto outTensorInfo =
        executionEngine_->inferOutputTensorInfo(tcName_, inputDLTensors);
    prepareOutputs(outTensorInfo);

    // now create the outputDLTensors
    std::vector<::tc::dlutils::DLTensorUPtr> outTensorUPtrs;
    std::vector<DLTensor*> outputDLTensors;
    for (int i = 0; i < OutputSize(); ++i) {
      outTensorUPtrs.emplace_back(dlpack::makeDLTensor(Output(i)));
      outputDLTensors.push_back(outTensorUPtrs.back().get());
    }

    // compile and run
    auto handle = executionEngine_->compile(
        tcName_, inputDLTensors, mappingOptions_.toProtobufSerializedString());
    executionEngine_->run(handle, inputDLTensors, outputDLTensors, profile_);
    return true;
  }

 protected:
  std::string tc_;
  std::string gradTc_;
  std::string tcName_;
  std::string gradTcName_;
  bool profile_;
  tc::CudaMappingOptions mappingOptions_;
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
