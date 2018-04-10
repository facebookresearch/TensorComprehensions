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
#include "caffe2/core/context_gpu.h"

#include "tc/c2/2fcrelu_op.h"
#include "tc/c2/2lut_op.h"
#include "tc/c2/3fcrelu_op.h"
#include "tc/c2/4fcrelu_op.h"
#include "tc/c2/convolution_op.h"
#include "tc/c2/copy_op.h"
#include "tc/c2/dper_lut_concat_op.h"
#include "tc/c2/fcrelu_op.h"
#include "tc/c2/group_convolution_op.h"
#include "tc/c2/lut_op.h"
#include "tc/c2/matmul_op.h"
#include "tc/c2/tc_op.h"

#include "tc/c2/operator_meta.h"

namespace caffe2 {

// TODO: generate optimized CPU code too if needed
// REGISTER_CPU_OPERATOR(TcMatMulOp, TcMatMulOp<float, CPUContext>);

REGISTER_CUDA_OPERATOR(TcMatMulOp, TcMatMulOp<float, CUDAContext>);
REGISTER_GRADIENT(TcMatMulOp, GetTcOpGradient);
OPERATOR_SCHEMA(TcMatMulOp)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(TcMatMulOp<float, CUDAContext>::description)
    .Arg("trans_a", "Pass 1 to transpose I before multiplication")
    .Arg("trans_b", "Pass 1 to transpose W before multiplication");
TC_REFERENCE_IMPLEMENTATION(
    TcMatMulOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "MatMul",
          "",
          {op_def.input(0), op_def.input(1)},
          {"O"},
          op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(TcFCReluOp, TcFCReluOp<float, CUDAContext>);
OPERATOR_SCHEMA(TcFCReluOp)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(TcFCReluOp<float, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    TcFCReluOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {op_def.input(0), op_def.input(1), op_def.input(2)},
          {"O1_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O1_r"}, {op_def.output(0)}, op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(Tc2FCReluOp, Tc2FCReluOp<float, CUDAContext>);
OPERATOR_SCHEMA(Tc2FCReluOp)
    .NumInputs(5)
    .NumOutputs(2)
    .SetDoc(Tc2FCReluOp<float, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    Tc2FCReluOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {op_def.input(0), op_def.input(1), op_def.input(2)},
          {"O1_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O1_r"}, {op_def.output(0)}, op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {op_def.output(0), op_def.input(3), op_def.input(4)},
          {"O2_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O2_r"}, {op_def.output(1)}, op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(Tc3FCReluOp, Tc3FCReluOp<float, CUDAContext>);
OPERATOR_SCHEMA(Tc3FCReluOp)
    .NumInputs(7)
    .NumOutputs(3)
    .SetDoc(Tc3FCReluOp<float, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    Tc3FCReluOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {op_def.input(0), op_def.input(1), op_def.input(2)},
          {"O1_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O1_r"}, {op_def.output(0)}, op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {"O1", op_def.input(3), op_def.input(4)},
          {"O2_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O2_r"}, {op_def.output(1)}, op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {"O2", op_def.input(5), op_def.input(6)},
          {"O3_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O3_r"}, {op_def.output(2)}, op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(Tc4FCReluOp, Tc4FCReluOp<float, CUDAContext>);
OPERATOR_SCHEMA(Tc4FCReluOp)
    .NumInputs(9)
    .NumOutputs(4)
    .SetDoc(Tc4FCReluOp<float, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    Tc4FCReluOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {op_def.input(0), op_def.input(1), op_def.input(2)},
          {"O1_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O1_r"}, {op_def.output(0)}, op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {"O1", op_def.input(3), op_def.input(4)},
          {"O2_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O2_r"}, {op_def.output(1)}, op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {"O2", op_def.input(5), op_def.input(6)},
          {"O3_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O3_r"}, {op_def.output(2)}, op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {"O3", op_def.input(7), op_def.input(8)},
          {"O4_r"},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Relu", "", {"O4_r"}, {op_def.output(3)}, op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(TcCopyOp, TcCopyOp<float, CUDAContext>);
REGISTER_GRADIENT(TcCopyOp, GetTcOpGradient);
OPERATOR_SCHEMA(TcCopyOp).NumInputs(1).NumOutputs(1).SetDoc(
    TcCopyOp<float, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    TcCopyOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Copy",
          "",
          {op_def.input(0)},
          {op_def.output(0)},
          op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(TcConvolutionOp, TcConvolutionOp<float, CUDAContext>);
REGISTER_GRADIENT(TcConvolutionOp, GetTcOpGradient);
OPERATOR_SCHEMA(TcConvolutionOp)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(TcConvolutionOp<float, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    TcConvolutionOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Conv",
          "",
          {op_def.input(0), op_def.input(1), op_def.input(2)},
          {op_def.output(0)},
          op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(
    TcGroupConvolutionOp,
    TcGroupConvolutionOp<float, CUDAContext>);
REGISTER_GRADIENT(TcGroupConvolutionOp, GetTcOpGradient);
OPERATOR_SCHEMA(TcGroupConvolutionOp).NumInputs(3).NumOutputs(1).SetDoc(R"DOC(
    O(b, g, o, x, y)  = 0
    O(b, g, o, x, y) += I(b, g, i, x + kx, y + ky) * W(g, i, o, kx, ky)
  )DOC");

////////////////////////////////////////////////////////////////////////////////
// TcLUTOp
////////////////////////////////////////////////////////////////////////////////
REGISTER_CUDA_OPERATOR(TcLUTOp, TcLUTOp<float, int, CUDAContext>);
OPERATOR_SCHEMA(TcLUTOp).NumInputs(2).NumOutputs(1).SetDoc(
    TcLUTOp<float, int, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    TcLUTOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      // For now we don't have proper ops in Caffe2 to do sparse length sum
      // on square matrix with missing indices. Thus trying to simulate it
      // by producing fake __lengths tensor in the test.
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FlattenToVec",
          "",
          {op_def.input(1)},
          {"__indices1"}, // HACK
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Cast",
          "",
          {"__indices1"},
          {"__indices1_casted"}, // HACK
          {MakeArgument<std::string>("to", "int64")},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum",
          "",
          {op_def.input(0), "__indices1_casted", "__lengths"}, // HACK
          {op_def.output(0)},
          op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(Tc2LUTOp, Tc2LUTOp<float, int, CUDAContext>);
OPERATOR_SCHEMA(Tc2LUTOp).NumInputs(4).NumOutputs(2).SetDoc(
    Tc2LUTOp<float, int, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    Tc2LUTOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      // For now we don't have proper ops in Caffe2 to do sparse length sum
      // on square matrix with missing indices. Thus trying to simulate it
      // by producing fake __lengths tensor in the test.
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FlattenToVec",
          "",
          {op_def.input(1)},
          {"__indices1"}, // HACK
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Cast",
          "",
          {"__indices1"},
          {"__indices1_casted"}, // HACK
          {MakeArgument<std::string>("to", "int64")},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum",
          "",
          {op_def.input(0), "__indices1_casted", "__lengths1"}, // HACK
          {op_def.output(0)},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FlattenToVec",
          "",
          {op_def.input(3)},
          {"__indices2"}, // HACK
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Cast",
          "",
          {"__indices2"},
          {"__indices2_casted"}, // HACK
          {MakeArgument<std::string>("to", "int64")},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum",
          "",
          {op_def.input(2), "__indices2_casted", "__lengths2"}, // HACK
          {op_def.output(1)},
          op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(
    TcDperLutConcatOp,
    TcDperLutConcatOp<float, CUDAContext>);
OPERATOR_SCHEMA(TcDperLutConcatOp)
    .NumInputs(7)
    .NumOutputs(3)
    .SetDoc(TcDperLutConcatOp<float, CUDAContext>::description);
TC_REFERENCE_IMPLEMENTATION(
    TcDperLutConcatOp,
    [](NetDef* net_def, const OperatorDef& op_def) {
      // For now we don't have proper ops in Caffe2 to do sparse length sum
      // on square matrix with missing indices. Thus trying to simulate it
      // by producing fake __lengths tensor in the test.
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FC",
          "",
          {op_def.input(0), op_def.input(3), op_def.input(4)},
          {op_def.output(0)},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FlattenToVec",
          "",
          {op_def.input(1)},
          {"__indices1"}, // HACK
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Cast",
          "",
          {"__indices1"},
          {"__indices1_casted"}, // HACK
          {MakeArgument<std::string>("to", "int64")},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum",
          "",
          {op_def.input(5), "__indices1_casted", "__lengths1"}, // HACK
          {op_def.output(1)},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "FlattenToVec",
          "",
          {op_def.input(2)},
          {"__indices2"}, // HACK
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "Cast",
          "",
          {"__indices2"},
          {"__indices2_casted"}, // HACK
          {MakeArgument<std::string>("to", "int64")},
          op_def.device_option()));
      net_def->add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum",
          "",
          {op_def.input(6), "__indices2_casted", "__lengths2"}, // HACK
          {op_def.output(2)},
          op_def.device_option()));
    });

REGISTER_CUDA_OPERATOR(TcOp, TcOp<float, CUDAContext>);
REGISTER_GRADIENT(TcOp, GetTcOpGradient);
OPERATOR_SCHEMA(TcOp).SetDoc(R"DOC(Generic Op using CudaExecutionEngine)DOC");
}; // namespace caffe2
