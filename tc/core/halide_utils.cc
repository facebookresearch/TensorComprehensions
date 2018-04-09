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
#include "tc/core/halide_utils.h"

#include <map>
#include <vector>

#include "tc/core/flags.h"
#include "tc/core/tc2halide.h"
#include "tc/core/tensor.h"

namespace tc {

using namespace Halide;
using namespace Halide::Internal;

DLDataType fromHalideType(const Halide::Type& type) {
  // Hmm, these are suspiciously similar...
  DLDataType dtype;
  dtype.bits = type.bits();
  dtype.code = type.code();
  dtype.lanes = type.lanes();
  return dtype;
}

std::map<std::string, int> computeParamValueMap(
    const tc2halide::HalideComponents& halide,
    const std::vector<const DLConstTensor*>& inputsDLT) {
  std::map<std::string, int> pvm;
  if (halide.inputs.size() != inputsDLT.size()) {
    throw lang::ErrorReport(halide.getDef())
        << "expected " << halide.inputs.size() << " inputs but got "
        << inputsDLT.size();
  }

  // fill the ParameterValMap
  for (size_t i = 0; i < inputsDLT.size(); i++) {
    const ImageParam& in = halide.inputs[i];
    auto tensor = inputsDLT[i];
    // for error messages
    auto paramType = halide.getDef().params()[i].tensorType();
    for (int d = 0; d < in.dimensions(); d++) {
      Expr extent = in.parameter().extent_constraint(d);
      int64_t currentSize = tensor->shape[d];
      auto dimExpTree = paramType.dims()[d];
      // dims can either be symbolic 'D' or a literal constant '4'
      if (const Variable* v = extent.as<Variable>()) {
        if (pvm.count(v->name) > 0) {
          int64_t prev = pvm[v->name];
          if (prev != currentSize) {
            throw lang::ErrorReport(dimExpTree)
                << "Mismatched sizes for dimension " << v->name
                << " previous value is " << prev << " but found " << currentSize
                << " here";
          }
        } else {
          pvm[v->name] = currentSize;
        }
      } else { // it was a constant
        const int64_t* c = as_const_int(extent);
        CHECK(c != NULL);
        if (*c != tensor->shape[d]) {
          throw lang::ErrorReport(dimExpTree)
              << "Constant dimension expected size " << *c << " but found "
              << tensor->shape[d];
        }
      }
    }
  }
  return pvm;
}

std::vector<TensorInfo> inferOutputTensorInfo(
    const tc2halide::HalideComponents& halide,
    const std::vector<const DLConstTensor*>& inputsDLT) {
  auto pvm = computeParamValueMap(halide, inputsDLT);

  // instantiate parameters with runtime values and build output DLpack metadata
  std::map<std::string, Expr> substitutions;
  for (auto p : pvm) {
    substitutions[p.first] = p.second;
  }
  std::vector<TensorInfo> outputTensorInfos;
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
    outputTensorInfos.emplace_back(TensorInfo(
        fromHalideType(out.type()), 0, sizes, makeStridesFromSizes(sizes)));
  }

  return outputTensorInfos;
}

std::string halideCodegenC(const Stmt& stmt) {
  // build C string from Halide stmt
  std::ostringstream ss;
  class HalideCPrinter : public IRPrinter {
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
      CHECK_EQ(1u, op->values.size())
          << "Cannot generate C for provide with != 1 values";
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
    HalideCPrinter(std::ostream& os) : IRPrinter(os) {}
  } printer(ss);

  stmt.accept(&printer);
  LOG_IF(INFO, FLAGS_debug_halide) << stmt;
  LOG_IF(INFO, FLAGS_debug_halide) << ss.str();
  return ss.str();
}

} // namespace tc
