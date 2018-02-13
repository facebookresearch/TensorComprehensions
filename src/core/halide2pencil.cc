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
#include "tc/core/halide2pencil.h"

#include <chrono>
#include <vector>

#include "tc/core/flags.h"
#include "tc/core/tc2halide.h"
#include "tc/core/utils/dlpack.h"

namespace tc {

using namespace dlutils;
using namespace Halide;
using namespace Halide::Internal;

namespace {

std::string C99TensorSignature(
    std::string name,
    const DLTensor* t,
    const std::unordered_set<std::string>& indirectAccesses,
    bool input = false) {
  std::stringstream ss;
  if (input) {
    ss << "const ";
  }
  // An extra level of indirection allows to pass full array sizes to Ppcg and
  // takes care of indirect memory accesses.
  ss << toString(t->dtype) << " " << name;
  if (indirectAccesses.count(name) > 0) {
    ss << "[]";
  }
  ss << "[" << t->shape[0] << "]";
  for (int ii = 0; ii < t->ndim - 1; ++ii) {
    CHECK_LT(0, t->strides[ii + 1]);
    CHECK_EQ(0, t->strides[ii] % t->strides[ii + 1])
        << "Strides non-comformable with C99 array in " << t;
    auto v = t->strides[ii] / t->strides[ii + 1];
    ss << "[" << v << "]";
  }
  return ss.str();
}

std::string C99TensorSignatures(
    const std::vector<const DLTensor*>& outs,
    const std::vector<const DLTensor*>& ins,
    const std::vector<std::string>& outputNames,
    const std::vector<std::string>& inputNames,
    const std::unordered_set<std::string>& indirectAccesses) {
  std::stringstream ss;
  for (size_t i = 0; i < outs.size(); ++i) {
    if (i > 0)
      ss << ", ";
    ss << C99TensorSignature(outputNames[i], outs[i], indirectAccesses);
  }
  for (size_t i = 0; i < ins.size(); ++i) {
    if (i > 0)
      ss << ", ";
    ss << C99TensorSignature(inputNames[i], ins[i], indirectAccesses, true);
  }
  return ss.str();
}

std::string CUDATensorSignatures(
    const std::vector<const DLTensor*>& outs,
    const std::vector<const DLTensor*>& ins,
    const std::vector<std::string>& outputNames,
    const std::vector<std::string>& inputNames) {
  std::stringstream ss;
  for (size_t i = 0; i < outs.size(); ++i) {
    if (i > 0)
      ss << ", ";
    ss << toString(outs[i]->dtype) << "* __restrict__ " << outputNames[i];
  }
  for (size_t i = 0; i < ins.size(); ++i) {
    if (i > 0)
      ss << ", ";
    ss << "const " << toString(ins[i]->dtype) << "* __restrict__ "
       << inputNames[i];
  }
  return ss.str();
}

} // namespace

// Different signatures are temporary, they should be removed in favor of
// a DeviceTensor-style API that will be compatible with whatever tensor
// library we end up using.
std::string C99Signature(
    const HalidePencilState& state,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<const DLTensor*>& inputs,
    const std::unordered_set<std::string>& indirectAccesses) {
  CHECK_EQ(outputs.size(), state.outputNames.size());
  CHECK_EQ(inputs.size(), state.inputNames.size());
  auto res = state.parameterSignature +
      C99TensorSignatures(
                 outputs,
                 inputs,
                 state.outputNames,
                 state.inputNames,
                 indirectAccesses);
  return res;
}

std::string CUDASignature(
    const HalidePencilState& state,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<const DLTensor*>& inputs) {
  auto res = state.parameterSignature +
      CUDATensorSignatures(
                 outputs, inputs, state.outputNames, state.inputNames);
  return res;
}

DLDataType fromHalideType(const Halide::Type& type) {
  // Hmm, these are suspiciously similar...
  DLDataType dtype;
  dtype.bits = type.bits();
  dtype.code = type.code();
  dtype.lanes = type.lanes();
  return dtype;
}

HalidePencilState toPencil(
    const tc2halide::HalideComponents& halide,
    const std::vector<const DLTensor*>& inputsDLT,
    bool scheduleSpecialize,
    const std::string& kernelName) {
  HalidePencilState pencilState;
  CHECK_EQ(halide.inputs.size(), inputsDLT.size())
      << "Mismatched HalideIR and DLTensor number of inputs";

  // fill the ParameterValMap
  for (size_t i = 0; i < inputsDLT.size(); i++) {
    const ImageParam& in = halide.inputs[i];
    const DLTensor* tensor = inputsDLT[i];
    // for error messages
    auto param_type = halide.getDef().params()[i].tensorType();
    for (int d = 0; d < in.dimensions(); d++) {
      Expr extent = in.parameter().extent_constraint(d);
      int64_t current_size = tensor->shape[d];
      auto dim_exp_tree = param_type.dims()[d];
      // dims can either be symbolic 'D' or a literal constant '4'
      if (const Variable* v = extent.as<Variable>()) {
        if (pencilState.pvm.count(v->name) > 0) {
          int64_t prev = pencilState.pvm[v->name];
          if (prev != current_size) {
            throw lang::ErrorReport(dim_exp_tree)
                << "Mismatched sizes for dimension " << v->name
                << " previous value is " << prev << " but found "
                << current_size << " here";
          }
        } else {
          pencilState.pvm[v->name] = current_size;
        }
      } else { // it was a constant
        const int64_t* c = as_const_int(extent);
        CHECK(c != NULL);
        if (*c != tensor->shape[d]) {
          throw lang::ErrorReport(dim_exp_tree)
              << "Constant dimension expected size " << *c << " but found "
              << tensor->shape[d];
        }
      }
    }
  }

  // Update actual parameters integer values
  pencilState.parameters.clear();
  for (auto& p : halide.params) {
    // halide.params contains all args to tc::def including tensors
    // only filter params coming from tensor shapes
    if (!p.second.is_buffer()) {
      pencilState.parameters.push_back(pencilState.pvm.at(p.first));
    }
  }

  // Update names: input, output, kernelName, parameterSignature,
  // kernelSpecializedName
  pencilState.inputNames.clear();
  for (auto& i : halide.inputs) {
    pencilState.inputNames.push_back(i.name());
  }
  pencilState.outputNames.clear();
  for (auto& o : halide.outputs) {
    pencilState.outputNames.push_back(o.name());
  }
  std::stringstream ss;
  for (auto& p : halide.params) {
    if (!p.second.is_buffer()) {
      ss << "int " << p.first << ", ";
    }
  }
  pencilState.parameterSignature = ss.str();
  std::stringstream ss2;
  pencilState.kernelName = kernelName;
  ss2 << pencilState.kernelName;
  for (int i : pencilState.parameters) {
    ss2 << "_" << i;
  }
  pencilState.kernelSpecializedName = ss2.str();

  // instantiate parameters with runtime values and build output DLpack metadata
  std::map<std::string, Expr> substitutions;
  for (auto p : pencilState.pvm) {
    substitutions[p.first] = p.second;
  }
  DLContext ctx{kDLGPU, 0};
  if (inputsDLT.size() > 0)
    ctx = inputsDLT[0]->ctx;
  std::vector<DLTensorUPtr> outputsDLT;
  for (size_t i = 0; i < halide.outputs.size(); ++i) {
    std::vector<long> sizes;
    auto& out = halide.outputs[i];
    auto tree = halide.getDef().returns()[i];
    for (int d = 0; d < out.dimensions(); d++) {
      Expr extent = out.parameter().extent_constraint(d);
      extent = simplify(substitute(substitutions, extent));
      const int64_t* c = as_const_int(extent);
      if (!c) {
        throw lang::ErrorReport(tree)
            << "Output tensor dimension " << d
            << " does not have a constant size, its extent is " << extent;
      }
      sizes.push_back(static_cast<int>(*c));
    }
    outputsDLT.emplace_back(
        makeDLTensorWithSizes(ctx, fromHalideType(out.type()), sizes));
  }
  std::vector<const DLTensor*> outputsRawPtr;
  for (auto& p : outputsDLT) {
    outputsRawPtr.push_back(p.get());
  }

  Stmt stmt = halide.stmt;
  if (scheduleSpecialize) {
    stmt = substitute(substitutions, stmt);
    // Constant-fold following substitution
    stmt = simplify(stmt);
  }

  // extract indirectAccesses from the HalideContextPencil. Their signature is
  // different
  std::unordered_set<std::string> indirectAccesses;
  /*
    TODO
  for (auto i : HalideContextPencil::IndirectAccesses(vops)) {
    indirectAccesses.insert(i->name);
  }
  */

  auto body = tc::halide2Pencil(stmt);

  pencilState.outputsDLT = std::move(outputsDLT);
  pencilState.signatureSourcePair = SignatureSourcePair{
      C99Signature(pencilState, outputsRawPtr, inputsDLT, indirectAccesses),
      CUDASignature(pencilState, outputsRawPtr, inputsDLT),
      body};

  return pencilState;
}

std::string KernelSpecializedName(const HalidePencilState& state) {
  return state.kernelSpecializedName;
}

std::string GetKernelName(const HalidePencilState& state) {
  return state.kernelName;
}

void SetKernelName(HalidePencilState& state, const std::string& name) {
  state.kernelName = name;
}

std::vector<int> GetKernelParameters(const HalidePencilState& state) {
  return state.parameters;
}

std::map<std::string, int> GetParameterValMap(const HalidePencilState& state) {
  return state.pvm;
}

std::unordered_map<std::string, long> GetParameterValues(
    const HalidePencilState& state) {
  std::unordered_map<std::string, long> m;
  for (const auto& p : state.pvm) {
    m[p.first] = p.second;
  }
  return m;
}

std::string halide2Pencil(const Stmt& stmt) {
  // build Pencil string from Halide stmt
  std::ostringstream ss;
  class PencilPrinter : public IRPrinter {
    using IRPrinter::visit;

    void visit(const Call* op) override {
      if (op->is_intrinsic(tc2halide::kReductionInit) ||
          op->is_intrinsic(tc2halide::kReductionUpdate)) {
        op->args[0].accept(this);
      } else if (
          op->call_type == Call::Halide || op->call_type == Call::Image) {
        // TODO: worry about indirect access...
        stream << op->name;
        for (auto& a : op->args) {
          stream << "[";
          a.accept(this);
          stream << "]";
        }
      } else {
        IRPrinter::visit(op);
      }
    }

    void visit(const Provide* op) override {
      do_indent();
      stream << op->name;
      for (auto& a : op->args) {
        stream << "[";
        a.accept(this);
        stream << "]";
      }
      stream << " = ";
      CHECK_EQ(1, op->values.size())
          << "Cannot generate PENCIL for provide with != 1 values";
      op->values[0].accept(this);
      stream << ";\n";
    }

    void visit(const For* op) override {
      do_indent();
      Expr max = simplify(op->min + op->extent);
      stream << "for (int " << op->name << " = " << op->min << "; " << op->name
             << " < " << max << "; " << op->name << "++) {\n";
      indent += 2;
      op->body.accept(this);
      indent -= 2;
      do_indent();
      stream << "}\n";
    }

   public:
    PencilPrinter(std::ostream& os) : IRPrinter(os) {}
  } printer(ss);

  stmt.accept(&printer);
  LOG_IF(INFO, FLAGS_debug_halide) << stmt;
  LOG_IF(INFO, FLAGS_debug_halide) << ss.str();
  return ss.str();
}

} // namespace tc
