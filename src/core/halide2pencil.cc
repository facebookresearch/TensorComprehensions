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
    const std::vector<const DLTensor*>& inputsDLT) {
  HalidePencilState pencilState;
  std::map<std::string, int> pvm;
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
        if (pvm.count(v->name) > 0) {
          int64_t prev = pvm[v->name];
          if (prev != current_size) {
            throw lang::ErrorReport(dim_exp_tree)
                << "Mismatched sizes for dimension " << v->name
                << " previous value is " << prev << " but found "
                << current_size << " here";
          }
        } else {
          pvm[v->name] = current_size;
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

  // instantiate parameters with runtime values and build output DLpack metadata
  std::map<std::string, Expr> substitutions;
  for (auto p : pvm) {
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

  pencilState.outputsDLT = std::move(outputsDLT);

  return pencilState;
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
