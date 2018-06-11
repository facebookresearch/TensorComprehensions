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
#include "MLP_model.h"

#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include "tc/aten/aten_compiler.h"
#include "tc/core/check.h"
#include "tc/core/cuda/cuda_mapping_options.h"

#include "../test/caffe2/cuda/test_harness.h"
#include "../test/caffe2/test_harness.h"
#include "../test/test_harness_aten_cuda.h"
#include "benchmark_fixture.h"

#include "tc/c2/context.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/flags.h"

using namespace caffe2;

DEFINE_uint32(B, 128, "Batch size");

// LUT part of the model
DEFINE_uint32(D, 64, "LUT embedding size (in floats)");
DEFINE_uint32(L1, 50, "LUT1 indices per batch");
DEFINE_uint32(E1, 1e5, "LUT1 rows (number of embeddings)");
DEFINE_uint32(L2, 50, "LUT2 indices per batch");
DEFINE_uint32(E2, 1e5, "LUT2 rows (number of embeddings)");
DEFINE_uint32(WX, 1000, "W rows");
DEFINE_uint32(WY, 1024, "W cols");

// MLP part of the model
DEFINE_uint32(M, 2000, "I_w == W1_w");
DEFINE_uint32(N, 128, "W1_h == W2_w");
DEFINE_uint32(O, 64, "W2_h == W3_w");
DEFINE_uint32(P, 32, "W3_h == W4_w");
DEFINE_uint32(Q, 2, "W4_h");

// ATen functions defined here:
//   third-party-install/share/ATen/Declarations.yaml
//   third-party-install/include/ATen/Functions.h
//   third-party-install/include/ATen/TensorMethods.h

//  paper writes the model in this pseudo-code form
// def _2LUT(
//   float(E1, D) LUT1, int32(B, L1) I1,
//   float(E2, D) LUT2, int32(B, L2) I2) -> (O1, O2)
// {
//     O1(b, d) +=! LUT1(I1(b, r_l1), d)
//     O2(b, d) +=! LUT2(I2(b, r_l2), d)
// }
// def _3FCRELU(
//   float(B,M) I, float(O,N) W2, float(O) B2,
//   float(P,O) W3, float(P) B3, float(Q,P) W4,
//   float(Q) B4) -> (O1, O2, O3, O4)
// {
//     O2(b, o)  = B2(o)
//     O2(b, o) += O1(b, n) * W2(o, n)
//     O2(b, o)  = fmax(O2(b, o), 0)
//     O3(b, p)  = B3(p)
//     O3(b, p) += O2(b, o) * W3(p, o)
//     O3(b, p)  = fmax(O3(b, p), 0)
//     O4(b, q)  = B4(q)
//     O4(b, q) += O3(b, p) * W4(q, p)
//     O4(b, q)  = fmax(O4(b, q), 0)
// }
// def prod_model(float(E1, D) LUT1, int32(B, L1) I1,
//                float(E2, D) LUT2, int32(B, L2) I2,
//                float(B, WX) I3, float(WY,WX) W,
//                float(N,M) W1, float(N) B1,
//                float(O,N) W2, float(O) B2,
//                float(P,O) W3, float(P) B3,
//                float(Q,P) W4, float(Q) B4)
// -> (C1, C2, C3, I, O1, O2, O3, O4)
// {
//       (C1, C2) = _2LUT(LUT1, I1, LUT2, I2)
//     C3(b, wy) +=! I3(b, r_wx) * W(wy, r_wx)
//        I(b, m) = Concat(C1, C2, C3) // not in TC atm
//       O1(b, n) = B1(n)
//      O1(b, n) +=! I(b, m) * W1(n, m)
//       O1(b, n) = fmax(O1(b, n), 0)
//   (O2, O3, O4) =
//       _3FCRELU(I, W1, B1, W2, B2, W3, B3, W4, B4)
//     # O4 goes out to binary classifier, omitted here
// }

class ProductionModel : public Benchmark {
 protected:
  uint32_t B, M, N, O, P, Q;
  uint32_t D, L1, L2, E1, E2;
  uint32_t WX, WY;

 public:
  void InitAll(
      uint32_t b,
      uint32_t m,
      uint32_t n,
      uint32_t o,
      uint32_t p,
      uint32_t q,
      uint32_t d,
      uint32_t l1,
      uint32_t l2,
      uint32_t e1,
      uint32_t e2,
      uint32_t wx,
      uint32_t wy) {
    B = b;
    M = m;
    N = n;
    O = o;
    P = p;
    Q = q;
    D = d;
    L1 = l1;
    L2 = l2;
    E1 = e1;
    E2 = e2;
    WX = wx;
    WY = wy;
  }
  void InitBDL1E1(uint32_t b, uint32_t d, uint32_t l1, uint32_t e1) {
    B = b;
    D = d;
    L1 = l1;
    E1 = e1;
  }
  void InitBDL1E1L2E2(
      uint32_t b,
      uint32_t d,
      uint32_t l1,
      uint32_t e1,
      uint32_t l2,
      uint32_t e2) {
    B = b;
    D = d;
    L1 = l1;
    L2 = l2;
    E1 = e1;
    E2 = e2;
  }
  void InitBWXWY(uint32_t b, uint32_t wx, uint32_t wy) {
    B = b;
    WX = wx;
    WY = wy;
  }
  void InitBMN(uint32_t b, uint32_t m, uint32_t n) {
    B = b;
    M = m;
    N = n;
  }
  void InitBNOPQ(uint32_t b, uint32_t n, uint32_t o, uint32_t p, uint32_t q) {
    B = b;
    N = n;
    O = o;
    P = p;
    Q = q;
  }

  void run1LUT(const tc::CudaMappingOptions& options);
  void runCaffe21LUT();
  void runATen1LUT();

  void run2LUT(const tc::CudaMappingOptions& options);
  void runCaffe22LUT();
  void runATen2LUT();

  void runC3(const tc::CudaMappingOptions& options);
  void runCaffe2C3();
  void runATenC3();

  void runMLP1(const tc::CudaMappingOptions& options);
  void runCaffe2MLP1();
  void runATenMLP1();

  void runMLP3(const tc::CudaMappingOptions& options);
  void runCaffe2MLP3();
  void runATenMLP3();
};

void ProductionModel::run1LUT(const tc::CudaMappingOptions& options) {
  // This test uses an c2 OpTester because we need to run the C2 reference
  // implementation for TcLUTOp.
  auto ws_init_func = [=](Workspace& w) {
    AddDeterministicallyRandomInput<caffe2::CUDABackend, float>(
        w, {E1, D}, "LUT");

    detail::AddDeterministicallyRandomInputWithRange<caffe2::CUDABackend, int>(
        w, {B, L1}, "I", 0, E1 - 1);
    AddConstInput<caffe2::CUDABackend, int>(w, {B}, L1, "__lengths");
  };
  OperatorDef op_def =
      MakeOperatorDef<caffe2::CUDABackend>("TcLUTOp", {"LUT", "I"}, {"O"});
  std::unique_ptr<OpTester> reference(new OpTester(op_def));
  reference->InitializeReference(ws_init_func);
  reference->RunReference();

  auto& reference_workspace = reference->w_ref;
  auto expected_blob = caffe2::TensorCPU(
      GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "O"));

  {
    // Piggy-back on the C2 CUDA tensors
    auto in_lut_blob =
        GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "LUT");
    auto in_idx_blob =
        GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "I");
    at::Tensor LUT1 =
        MakeAtenTensor(in_lut_blob, at::Backend::CUDA, at::ScalarType::Float);
    at::Tensor IDX1 =
        MakeAtenTensor(in_idx_blob, at::Backend::CUDA, at::ScalarType::Int);

    auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                         const std::vector<at::Tensor>& outputs) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      double prec = 3e-7;
      std::cout << "Checking expected output relative precision @" << prec;
      at::Tensor tO =
          MakeAtenTensor(expected_blob, at::Backend::CUDA, at::kFloat)
              .resize_({B, D});
      checkRtol(outputs[0].sub(tO), inputs, L1, prec);
      return true;
    };

    std::vector<at::Tensor> inputs = {LUT1, IDX1};
    std::string tc = R"(
def _1LUT(float(E1, D) LUT1, int32(B, L1) I1) -> (O1) {
    O1(b, d) +=! LUT1(I1(b, r_l1), d)
}
    )";

    std::string suffix = std::string("_B_") + std::to_string(FLAGS_B) +
        std::string("_D_") + std::to_string(FLAGS_D) + std::string("_L1_") +
        std::to_string(FLAGS_L1) + std::string("_E1_") +
        std::to_string(FLAGS_E1);
    std::vector<tc::CudaMappingOptions> bestOptions{options};
    if (FLAGS_autotune) {
      bestOptions = autotune(
          FLAGS_save_tuner_proto_prefix + std::string("/1LUT_cache") + suffix,
          FLAGS_save_tuner_proto_prefix + std::string("/1LUT_best") + suffix,
          tc,
          "_1LUT",
          inputs,
          options,
          check_fun);
      TC_CHECK_GE(bestOptions.size(), 1u);
    }
    Check(tc, "_1LUT", options, inputs, check_fun);
  }
}

void ProductionModel::runCaffe21LUT() {
  auto ws_init_func = [=](Workspace& w) {
    AddDeterministicallyRandomInput<caffe2::CUDABackend, float>(
        w, {E1, D}, "LUT");
    detail::AddDeterministicallyRandomInputWithRange<caffe2::CUDABackend, int>(
        w, {B, L1}, "I", 0, E1 - 1);
    AddConstInput<caffe2::CUDABackend, int>(w, {B}, L1, "__lengths");
  };
  OperatorDef op_def =
      MakeOperatorDef<caffe2::CUDABackend>("TcLUTOp", {"LUT", "I"}, {"O"});
  std::unique_ptr<OpTester> reference(new OpTester(op_def));
  reference->InitializeReference(ws_init_func);

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

void ProductionModel::runATen1LUT() {
  std::cout << "No ATen1LUTReference available\n";
}

void ProductionModel::run2LUT(const tc::CudaMappingOptions& options) {
  TC_CHECK_LT(0, E1);
  TC_CHECK_LT(0, E2);
  auto ws_init_func = [=](Workspace& w) {
    AddDeterministicallyRandomInput<caffe2::CUDABackend, float>(
        w, {E1, D}, "LUT1");

    detail::AddDeterministicallyRandomInputWithRange<caffe2::CUDABackend, int>(
        w, {B, L1}, "IDX1", 0, E1 - 1);
    AddDeterministicallyRandomInput<caffe2::CUDABackend, float>(
        w, {E2, D}, "LUT2");

    detail::AddDeterministicallyRandomInputWithRange<caffe2::CUDABackend, int>(
        w, {B, L2}, "IDX2", 0, E2 - 1);
    AddConstInput<caffe2::CUDABackend, int>(w, {B}, L1, "__lengths1");
    AddConstInput<caffe2::CUDABackend, int>(w, {B}, L2, "__lengths2");
  };
  OperatorDef op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "Tc2LUTOp", {"LUT1", "IDX1", "LUT2", "IDX2"}, {"O1", "O2"});
  std::unique_ptr<OpTester> reference(new OpTester(op_def));
  reference->InitializeReference(ws_init_func);
  reference->RunReference();

  auto& reference_workspace = reference->w_ref;
  std::vector<caffe2::Tensor<caffe2::CPUContext>> expected_output;
  expected_output.emplace_back(caffe2::TensorCPU(
      GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "O1")));
  expected_output.emplace_back(caffe2::TensorCPU(
      GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "O2")));

  {
    auto in_lut1_blob =
        GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "LUT1");
    auto in_idx1_blob =
        GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "IDX1");
    at::Tensor LUT1 =
        MakeAtenTensor(in_lut1_blob, at::Backend::CUDA, at::ScalarType::Float);
    at::Tensor IDX1 =
        MakeAtenTensor(in_idx1_blob, at::Backend::CUDA, at::ScalarType::Int);

    auto in_lut2_blob =
        GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "LUT2");
    auto in_idx2_blob =
        GetNamedTensor<caffe2::CUDABackend>(reference_workspace, "IDX2");
    at::Tensor LUT2 =
        MakeAtenTensor(in_lut2_blob, at::Backend::CUDA, at::ScalarType::Float);
    at::Tensor IDX2 =
        MakeAtenTensor(in_idx2_blob, at::Backend::CUDA, at::ScalarType::Int);

    auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                         const std::vector<at::Tensor>& outputs) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      double prec = 3e-7;
      std::cout << "Checking expected output relative precision @" << prec;
      at::Tensor tO =
          MakeAtenTensor(expected_output[0], at::Backend::CUDA, at::kFloat)
              .resize_({B, D});
      checkRtol(outputs[0].sub(tO), inputs, L1, prec);
      {
        at::Tensor tO =
            MakeAtenTensor(expected_output[1], at::Backend::CUDA, at::kFloat)
                .resize_({B, D});
        checkRtol(outputs[1].sub(tO), inputs, L2, prec);
      }
      return true;
    };

    std::vector<at::Tensor> inputs = {LUT1, IDX1, LUT2, IDX2};
    std::string tc = R"(
def _2LUT(float(E1, D) LUT1, int32(B, L1) I1, float(E2, D) LUT2, int32(B, L2) I2) -> (O1, O2) {
    O1(b, d) +=! LUT1(I1(b, r_l1), d)
    O2(b, d) +=! LUT2(I2(b, r_l2), d)
}
    )";

    std::string suffix = std::string("_B_") + std::to_string(FLAGS_B) +
        std::string("_D_") + std::to_string(FLAGS_D) + std::string("_L1_") +
        std::to_string(FLAGS_L1) + std::string("_E1_") +
        std::to_string(FLAGS_E1) + std::string("_L2_") +
        std::to_string(FLAGS_L2) + std::string("_E2_") +
        std::to_string(FLAGS_E2);
    std::vector<tc::CudaMappingOptions> bestOptions{options};
    if (FLAGS_autotune) {
      bestOptions = autotune(
          FLAGS_save_tuner_proto_prefix + std::string("/2LUT_cache") + suffix,
          FLAGS_save_tuner_proto_prefix + std::string("/2LUT_best") + suffix,
          tc,
          "_2LUT",
          inputs,
          options,
          check_fun);
      TC_CHECK_GE(bestOptions.size(), 1u);
    }
    Check(tc, "_2LUT", bestOptions[0], inputs, check_fun);
  }
}

void ProductionModel::runCaffe22LUT() {
  auto ws_init_func = [=](Workspace& w) {
    AddDeterministicallyRandomInput<caffe2::CUDABackend, float>(
        w, {E1, D}, "LUT1");

    detail::AddDeterministicallyRandomInputWithRange<caffe2::CUDABackend, int>(
        w, {B, L1}, "IDX1", 0, E1 - 1);
    AddDeterministicallyRandomInput<caffe2::CUDABackend, float>(
        w, {E2, D}, "LUT2");

    detail::AddDeterministicallyRandomInputWithRange<caffe2::CUDABackend, int>(
        w, {B, L2}, "IDX2", 0, E2 - 1);
    AddConstInput<caffe2::CUDABackend, int>(w, {B}, L1, "__lengths1");
    AddConstInput<caffe2::CUDABackend, int>(w, {B}, L2, "__lengths2");
  };
  OperatorDef op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "Tc2LUTOp", {"LUT1", "IDX1", "LUT2", "IDX2"}, {"O1", "O2"});
  std::unique_ptr<OpTester> reference(new OpTester(op_def));
  reference->InitializeReference(ws_init_func);

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

void ProductionModel::runATen2LUT() {
  std::cout << "No ATen2LUTReference available\n";
}

void ProductionModel::runC3(const tc::CudaMappingOptions& options) {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, WX});
  at::Tensor W = at::CUDA(at::kFloat).rand({WY, WX});

  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 1e-6;
    std::cout << "Checking expected output relative precision @" << prec;
    auto C3 = I.mm(W.t());
    at::Tensor diff = outputs[0].sub(C3);
    checkRtol(diff, inputs, WY + 1, prec);
    return true;
  };

  std::vector<at::Tensor> inputs = {I, W};
  std::string tc = R"TC(
def _C3(float(B,WX) I, float(WY, WX) W) -> (C3) {
    C3(b, wy) +=! I(b, r_wx) * W(wy, r_wx)
}
)TC";

  std::string suffix = std::string("_B_") + std::to_string(FLAGS_B) +
      std::string("_WX_") + std::to_string(FLAGS_WX) + std::string("_WY_") +
      std::to_string(FLAGS_WY);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        FLAGS_save_tuner_proto_prefix + std::string("/_C3_cache") + suffix,
        FLAGS_save_tuner_proto_prefix + std::string("/_C3_best") + suffix,
        tc,
        "_C3",
        inputs,
        options,
        check_fun);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }
  Check(tc, "_C3", bestOptions[0], inputs, check_fun);
}

void ProductionModel::runCaffe2C3() {
  auto ws_init_func = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {B, WX}, "I");
    AddInput(w, {WY, WX}, "W");
  };
  OperatorDef op_def =
      MakeOperatorDef<caffe2::CUDABackend>("TcMatMulOp", {"I", "W"}, {"O"});
  std::unique_ptr<OpTester> reference(new OpTester(op_def));
  reference->InitializeReference(ws_init_func, {{"trans_b", 1}});
  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

void ProductionModel::runATenC3() {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, WX});
  at::Tensor W = at::CUDA(at::kFloat).rand({WY, WX});
  Reference(
      [&]() { return I.mm(W.t()); },
      [&](at::Tensor& res) { mm_out(res, I, W.t()); });
}

void ProductionModel::runMLP1(const tc::CudaMappingOptions& options) {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, M});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({M, N});
  at::Tensor B1 = at::CUDA(at::kFloat).rand({N});

  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 1e-6;
    std::cout << "Checking expected output relative precision @" << prec;
    auto O1 = I.mm(W1).add(B1).clamp_min(0);
    at::Tensor diff = outputs[0].sub(O1);
    checkRtol(diff, inputs, M + 1, prec);
    return true;
  };

  std::vector<at::Tensor> inputs = {I, W1, B1};
  std::string tc = R"TC(
def mlp1(float(B,M) I, float(M, N) W1, float(N) B1) -> (O1) {
    O1(b, n) +=! I(b, r_m) * W1(r_m, n)
    O1(b, n)  = O1(b,   n) + B1(n)
    O1(b, n)  = fmax(O1(b, n), 0)
}
)TC";

  std::string suffix = std::string("_B_") + std::to_string(FLAGS_B) +
      std::string("_M_") + std::to_string(FLAGS_M) + std::string("_N_") +
      std::to_string(FLAGS_N);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        FLAGS_save_tuner_proto_prefix + std::string("/mlp1_cache") + suffix,
        FLAGS_save_tuner_proto_prefix + std::string("/mlp1_best") + suffix,
        tc,
        "mlp1",
        inputs,
        options,
        check_fun);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }
  Check(tc, "mlp1", bestOptions[0], inputs, check_fun);
}

void ProductionModel::runCaffe2MLP1() {
  auto ws_init_func = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {B, M}, "I");
    AddInput(w, {N, M}, "W1");
    AddInput(w, {N}, "B1");
  };
  OperatorDef op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcFCReluOp", {"I", "W1", "B1"}, {"O1"});
  std::unique_ptr<OpTester> reference(new OpTester(op_def));
  reference->InitializeReference(ws_init_func);

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

void ProductionModel::runATenMLP1() {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, M});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({M, N});
  at::Tensor B1 = at::CUDA(at::kFloat).rand({N});

  Reference(
      [&]() { return I.mm(W1).add(B1).clamp_min(0); },
      [&](at::Tensor& res) { mm_out(res, I, W1).add(B1).clamp_min(0); });
}

void ProductionModel::runMLP3(const tc::CudaMappingOptions& options) {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, N});
  at::Tensor W2 = at::CUDA(at::kFloat).rand({O, N});
  at::Tensor B2 = at::CUDA(at::kFloat).rand({O});
  at::Tensor W3 = at::CUDA(at::kFloat).rand({P, O});
  at::Tensor B3 = at::CUDA(at::kFloat).rand({P});
  at::Tensor W4 = at::CUDA(at::kFloat).rand({Q, P});
  at::Tensor B4 = at::CUDA(at::kFloat).rand({Q});

  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 3e-7;
    std::cout << "Checking expected output relative precision @" << prec;
    auto O2 = I.mm(W2.t()).add(B2).clamp_min(0);
    checkRtol(outputs[0].sub(O2), inputs, N + 1, prec);
    auto O3 = O2.mm(W3.t()).add(B3).clamp_min(0);
    checkRtol(outputs[1].sub(O3), inputs, N * O + 2, prec);
    auto O4 = O3.mm(W4.t()).add(B4).clamp_min(0);
    checkRtol(outputs[2].sub(O4), inputs, N * O * P + 3, prec);
    return true;
  };

  std::vector<at::Tensor> inputs = {I, W2, B2, W3, B3, W4, B4};
  std::string tc = R"TC(
def mlp3(float(B,N) I, float(O,N) W2, float(O) B2, float(P,O) W3, float(P) B3,
         float(Q,P) W4, float(Q) B4) -> (O2, O3, O4) {
    O2(b, o) +=!  I(b, n) * W2(o, n)
    O2(b, o)  =  O2(b, o) + B2(o)
    O2(b, o)  = fmax(O2(b, o), 0)
    O3(b, p) +=! O2(b, o) * W3(p, o)
    O3(b, p)  =  O3(b, p) + B3(p)
    O3(b, p)  = fmax(O3(b, p), 0)
    O4(b, q) +=! O3(b, p) * W4(q, p)
    O4(b, q)  =  O4(b, q) + B4(q)
    O4(b, q)  = fmax(O4(b, q), 0)
}
)TC";

  std::string suffix = std::string("_B_") + std::to_string(FLAGS_B) +
      std::string("_M_") + std::to_string(FLAGS_M) + std::string("_N_") +
      std::to_string(FLAGS_N);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        FLAGS_save_tuner_proto_prefix + std::string("/mlp3_cache") + suffix,
        FLAGS_save_tuner_proto_prefix + std::string("/mlp3_best") + suffix,
        tc,
        "mlp3",
        inputs,
        options,
        check_fun);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }
  Check(tc, "mlp3", bestOptions[0], inputs, check_fun);
}

void ProductionModel::runCaffe2MLP3() {
  auto AddInput = AddConstInput<caffe2::CUDABackend, float>;
  auto ws_init_func = [&](Workspace& w) {
    AddInput(w, vector<TIndex>{B, N}, 1., "I");
    AddInput(w, vector<TIndex>{O, N}, 1., "W1");
    AddInput(w, vector<TIndex>{O}, 1., "B1");
    AddInput(w, vector<TIndex>{P, O}, 1., "W2");
    AddInput(w, vector<TIndex>{P}, 1., "B2");
    AddInput(w, vector<TIndex>{Q, P}, 1., "W3");
    AddInput(w, vector<TIndex>{Q}, 1., "B3");
  };
  OperatorDef op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "Tc3FCReluOp",
      {"I", "W1", "B1", "W2", "B2", "W3", "B3"},
      {"O1", "O2", "O3"});
  std::unique_ptr<OpTester> reference(new OpTester(op_def));
  reference->InitializeReference(ws_init_func);

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

void ProductionModel::runATenMLP3() {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, N});
  at::Tensor W2 = at::CUDA(at::kFloat).rand({O, N});
  at::Tensor B2 = at::CUDA(at::kFloat).rand({O});
  at::Tensor W3 = at::CUDA(at::kFloat).rand({P, O});
  at::Tensor B3 = at::CUDA(at::kFloat).rand({P});
  at::Tensor W4 = at::CUDA(at::kFloat).rand({Q, P});
  at::Tensor B4 = at::CUDA(at::kFloat).rand({Q});

  Reference(
      [&]() {
        auto O2 = I.mm(W2.t()).add(B2).clamp_min(0);
        auto O3 = O2.mm(W3.t()).add(B3).clamp_min(0);
        auto O4 = O3.mm(W4.t()).add(B4).clamp_min(0);
        return std::vector<at::Tensor>{O2, O3, O4};
      },
      [&](std::vector<at::Tensor>& res) {
        auto& O2 = res[0];
        auto& O3 = res[1];
        auto& O4 = res[2];
        mm_out(O2, I, W2.t()).add(B2).clamp_min(0);
        mm_out(O3, O2, W3.t()).add(B3).clamp_min(0);
        mm_out(O4, O3, W4.t()).add(B4).clamp_min(0);
      });
}

///////////////////////////////////////////////////////////////////////////////
// Start tests
///////////////////////////////////////////////////////////////////////////////
/// 1LUT
// Generic
TEST_F(ProductionModel, 1LUT) {
  InitAll(
      FLAGS_B,
      FLAGS_M,
      FLAGS_N,
      FLAGS_O,
      FLAGS_P,
      FLAGS_Q,
      FLAGS_D,
      FLAGS_L1,
      FLAGS_L2,
      FLAGS_E1,
      FLAGS_E2,
      FLAGS_WX,
      FLAGS_WY);
  run1LUT(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100 TC
TEST_F(ProductionModel, 1LUT_P100_autotuned_B_128_D_64_L1_50_E1_10000000) {
  InitBDL1E1(128, 64, 50, 10000000);
  run1LUT(tc::options_1LUT_P100_autotuned_B_128_D_64_L1_50_E1_10000000);
}

// P100 Caffe2
TEST_F(ProductionModel, 1LUT_Caffe2_P100_B_128_D_64_L1_50_E1_10000000) {
  InitBDL1E1(128, 64, 50, 10000000);
  runCaffe21LUT();
}

// P100 ATen
TEST_F(ProductionModel, 1LUT_ATen_P100_B_128_D_64_L1_50_E1_10000000) {
  InitBDL1E1(128, 64, 50, 10000000);
  runATen1LUT();
}

// V100 TC
TEST_F(ProductionModel, 1LUT_V100_autotuned_B_128_D_64_L1_50_E1_10000000) {
  InitBDL1E1(128, 64, 50, 10000000);
  run1LUT(tc::options_1LUT_V100_autotuned_B_128_D_64_L1_50_E1_10000000);
}

// V100 Caffe2
TEST_F(ProductionModel, 1LUT_Caffe2_V100_B_128_D_64_L1_50_E1_10000000) {
  InitBDL1E1(128, 64, 50, 10000000);
  runCaffe21LUT();
}

// V100 ATen
TEST_F(ProductionModel, 1LUT_ATen_V100_B_128_D_64_L1_50_E1_10000000) {
  InitBDL1E1(128, 64, 50, 10000000);
  runATen1LUT();
}

/// 2LUT
// Generic
TEST_F(ProductionModel, 2LUT) {
  InitAll(
      FLAGS_B,
      FLAGS_M,
      FLAGS_N,
      FLAGS_O,
      FLAGS_P,
      FLAGS_Q,
      FLAGS_D,
      FLAGS_L1,
      FLAGS_L2,
      FLAGS_E1,
      FLAGS_E2,
      FLAGS_WX,
      FLAGS_WY);
  run2LUT(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100 TC
TEST_F(
    ProductionModel,
    2LUT_P100_autotuned_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000) {
  InitBDL1E1L2E2(128, 64, 50, 10000000, 50, 10000000);
  run2LUT(
      tc::options_2LUT_P100_autotuned_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000);
}

// P100 Caffe2
TEST_F(
    ProductionModel,
    2LUT_Caffe2_P100_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000) {
  InitBDL1E1L2E2(128, 64, 50, 10000000, 50, 10000000);
  runCaffe22LUT();
}

// P100 ATen
TEST_F(
    ProductionModel,
    2LUT_ATen_P100_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000) {
  InitBDL1E1L2E2(128, 64, 50, 10000000, 50, 10000000);
  runATen2LUT();
}

// V100 TC
TEST_F(
    ProductionModel,
    2LUT_V100_autotuned_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000) {
  InitBDL1E1L2E2(128, 64, 50, 10000000, 50, 10000000);
  run2LUT(
      tc::options_2LUT_V100_autotuned_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000);
}

// V100 Caffe2
TEST_F(
    ProductionModel,
    2LUT_Caffe2_V100_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000) {
  InitBDL1E1L2E2(128, 64, 50, 10000000, 50, 10000000);
  runCaffe22LUT();
}

// V100 ATen
TEST_F(
    ProductionModel,
    2LUT_ATen_V100_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000) {
  InitBDL1E1L2E2(128, 64, 50, 10000000, 50, 10000000);
  runATen2LUT();
}

/// C3
// Generic
TEST_F(ProductionModel, C3) {
  InitAll(
      FLAGS_B,
      FLAGS_M,
      FLAGS_N,
      FLAGS_O,
      FLAGS_P,
      FLAGS_Q,
      FLAGS_D,
      FLAGS_L1,
      FLAGS_L2,
      FLAGS_E1,
      FLAGS_E2,
      FLAGS_WX,
      FLAGS_WY);
  runC3(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100 TC
TEST_F(ProductionModel, C3_P100_autotuned_B_128_WX_1000_WY_1024) {
  InitBWXWY(128, 1000, 1024);
  runC3(tc::options_C3_P100_autotuned_B_128_WX_1000_WY_1024);
}

// P100 Caffe2
TEST_F(ProductionModel, C3_Caffe2_P100_B_128_WX_1000_WY_1024) {
  InitBWXWY(128, 1000, 1024);
  runCaffe2C3();
}

// P100 ATen
TEST_F(ProductionModel, C3_ATen_P100_B_128_WX_1000_WY_1024) {
  InitBWXWY(128, 1000, 1024);
  runATenC3();
}

// V100 TC
TEST_F(ProductionModel, C3_V100_autotuned_B_128_WX_1000_WY_1024) {
  InitBWXWY(128, 1000, 1024);
  runC3(tc::options_C3_V100_autotuned_B_128_WX_1000_WY_1024);
}

// V100 Caffe2
TEST_F(ProductionModel, C3_Caffe2_V100_B_128_WX_1000_WY_1024) {
  InitBWXWY(128, 1000, 1024);
  runCaffe2C3();
}

// V100 ATen
TEST_F(ProductionModel, C3_ATen_V100_B_128_WX_1000_WY_1024) {
  InitBWXWY(128, 1000, 1024);
  runATenC3();
}

/// MLP1
// Generic
TEST_F(ProductionModel, MLP1) {
  InitAll(
      FLAGS_B,
      FLAGS_M,
      FLAGS_N,
      FLAGS_O,
      FLAGS_P,
      FLAGS_Q,
      FLAGS_D,
      FLAGS_L1,
      FLAGS_L2,
      FLAGS_E1,
      FLAGS_E2,
      FLAGS_WX,
      FLAGS_WY);
  runMLP1(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100 TC
TEST_F(ProductionModel, MLP1_P100_autotuned_B_128_M_2000_N_128) {
  InitBMN(128, 2000, 128);
  runMLP1(tc::options_MLP1_P100_autotuned_B_128_M_2000_N_128);
}

// P100 Caffe2
TEST_F(ProductionModel, MLP1_Caffe2_P100_B_128_M_2000_N_128) {
  InitBMN(128, 2000, 128);
  runCaffe2MLP1();
}

// P100 ATen
TEST_F(ProductionModel, MLP1_ATen_P100_B_128_M_2000_N_128) {
  InitBMN(128, 2000, 128);
  runATenMLP1();
}

// V100 TC
TEST_F(ProductionModel, MLP1_V100_autotuned_B_128_M_2000_N_128) {
  InitBMN(128, 2000, 128);
  runMLP1(tc::options_MLP1_V100_autotuned_B_128_M_2000_N_128);
}

// V100 Caffe2
TEST_F(ProductionModel, MLP1_Caffe2_V100_B_128_M_2000_N_128) {
  InitBMN(128, 2000, 128);
  runCaffe2MLP1();
}

// V100 ATen
TEST_F(ProductionModel, MLP1_ATen_V100_B_128_M_2000_N_128) {
  InitBMN(128, 2000, 128);
  runATenMLP1();
}

/// MLP3
// Generic
TEST_F(ProductionModel, MLP3) {
  InitAll(
      FLAGS_B,
      FLAGS_M,
      FLAGS_N,
      FLAGS_O,
      FLAGS_P,
      FLAGS_Q,
      FLAGS_D,
      FLAGS_L1,
      FLAGS_L2,
      FLAGS_E1,
      FLAGS_E2,
      FLAGS_WX,
      FLAGS_WY);
  runMLP3(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100 TC
TEST_F(ProductionModel, MLP3_P100_autotuned_B_128_N_128_O_64_P_32_Q_2) {
  InitBNOPQ(128, 128, 64, 32, 2);
  runMLP3(tc::options_MLP3_P100_autotuned_B_128_N_128_O_64_P_32_Q_2);
}

// P100 Caffe2
TEST_F(ProductionModel, MLP3_Caffe2_P100_B_128_N_128_O_64_P_32_Q_2) {
  InitBNOPQ(128, 128, 64, 32, 2);
  runCaffe2MLP3();
}

// P100 ATen
TEST_F(ProductionModel, MLP3_ATen_P100_B_128_N_128_O_64_P_32_Q_2) {
  InitBNOPQ(128, 128, 64, 32, 2);
  runATenMLP3();
}

// V100 TC
TEST_F(ProductionModel, MLP3_V100_autotuned_B_128_N_128_O_64_P_32_Q_2) {
  InitBNOPQ(128, 128, 64, 32, 2);
  runMLP3(tc::options_MLP3_V100_autotuned_B_128_N_128_O_64_P_32_Q_2);
}

// V100 Caffe2
TEST_F(ProductionModel, MLP3_Caffe2_V100_B_128_N_128_O_64_P_32_Q_2) {
  InitBNOPQ(128, 128, 64, 32, 2);
  runCaffe2MLP3();
}

// V100 ATen
TEST_F(ProductionModel, MLP3_ATen_V100_B_128_N_128_O_64_P_32_Q_2) {
  InitBNOPQ(128, 128, 64, 32, 2);
  runATenMLP3();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
