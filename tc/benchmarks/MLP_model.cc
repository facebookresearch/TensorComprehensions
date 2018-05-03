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
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"

#include "../test/test_harness.h"
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
 public:
  void run1LUT(
      uint32_t B,
      uint32_t D,
      uint32_t L1,
      uint32_t E1,
      const tc::CudaMappingOptions& options,
      bool useFlags = false);
  void run2LUT(
      uint32_t B,
      uint32_t D,
      uint32_t L1,
      uint32_t L2,
      uint32_t E1,
      uint32_t E2,
      const tc::CudaMappingOptions& options,
      bool useFlags = false);
  void runC3(
      uint32_t B,
      uint32_t WX,
      uint32_t WY,
      const tc::CudaMappingOptions& options,
      bool useFlags = false);
  void runMLP1(
      uint32_t B,
      uint32_t N,
      uint32_t M,
      const tc::CudaMappingOptions& options,
      bool useFlags = false);
  void runMLP3(
      uint32_t B,
      uint32_t N,
      uint32_t O,
      uint32_t P,
      uint32_t Q,
      const tc::CudaMappingOptions& options,
      bool useFlags = false);
};

void ProductionModel::run1LUT(
    uint32_t B,
    uint32_t D,
    uint32_t L1,
    uint32_t E1,
    const tc::CudaMappingOptions& options,
    bool useFlags) {
  CHECK_LT(0, E1);

  // This test uses an c2 OpTester because we need to run the C2 reference
  // implementation for TcLUTOp.
  auto ws_init_func = [=](Workspace& w) {
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {E1, D}, "LUT");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {B, L1}, "I", 0, E1 - 1);
    TestHarness::AddConstInput<int, CUDAContext>(w, {B}, L1, "__lengths");
  };
  OperatorDef op_def =
      TestHarness::ConfigureCUDA("TcLUTOp", {"LUT", "I"}, {"O"});
  std::unique_ptr<TestHarness::OpTester> reference(
      new TestHarness::OpTester(op_def));
  reference->InitializeReference(ws_init_func);
  reference->RunReference();
  auto expectedBlob = reference->getReferenceHostBlob("O");

  {
    // Piggy-back on the C2 CUDA tensors
    auto inLutBlob = reference->getReferenceDeviceBlob("LUT");
    auto inIdxBlob = reference->getReferenceDeviceBlob("I");
    at::Tensor LUT1 = makeATenTensor<caffe2::CUDAContext>(
        inLutBlob, at::Backend::CUDA, at::ScalarType::Float);
    at::Tensor IDX1 = makeATenTensor<caffe2::CUDAContext>(
        inIdxBlob, at::Backend::CUDA, at::ScalarType::Int);

    auto checkFun = [&](const std::vector<at::Tensor>& inputs,
                        const std::vector<at::Tensor>& outputs) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      double prec = 3e-7;
      std::cout << "Checking expected output relative precision @" << prec;
      at::Tensor tO =
          makeATenTensor(expectedBlob, at::Backend::CUDA, at::kFloat)
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
    if (useFlags && FLAGS_validate_proto) {
      validateProto(
          FLAGS_save_tuner_proto_prefix + std::string("/1LUT_cache") + suffix,
          tc,
          "_1LUT",
          inputs,
          checkFun);
    } else {
      std::vector<at::Tensor> outputs;
      Check(tc, "_1LUT", options, inputs, outputs, checkFun);
      if (useFlags) {
        autotune(
            FLAGS_save_tuner_proto_prefix + std::string("/1LUT_cache") + suffix,
            FLAGS_save_tuner_proto_prefix + std::string("/1LUT_best") + suffix,
            tc,
            "_1LUT",
            inputs,
            options,
            {options},
            checkFun);
      }
    }
  }
}

void ProductionModel::run2LUT(
    uint32_t B,
    uint32_t D,
    uint32_t L1,
    uint32_t L2,
    uint32_t E1,
    uint32_t E2,
    const tc::CudaMappingOptions& options,
    bool useFlags) {
  CHECK_LT(0, E1);
  CHECK_LT(0, E2);

  auto ws_init_func = [=](Workspace& w) {
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {E1, D}, "LUT1");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {B, L1}, "IDX1", 0, E1 - 1);
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {E2, D}, "LUT2");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {B, L2}, "IDX2", 0, E2 - 1);
    TestHarness::AddConstInput<int, CUDAContext>(w, {B}, L1, "__lengths1");
    TestHarness::AddConstInput<int, CUDAContext>(w, {B}, L2, "__lengths2");
  };
  OperatorDef op_def = TestHarness::ConfigureCUDA(
      "Tc2LUTOp", {"LUT1", "IDX1", "LUT2", "IDX2"}, {"O1", "O2"});
  std::unique_ptr<TestHarness::OpTester> reference(
      new TestHarness::OpTester(op_def));
  reference->InitializeReference(ws_init_func);
  reference->RunReference();
  std::vector<caffe2::Tensor<caffe2::CPUContext>> expectedOutput;
  expectedOutput.push_back(reference->getReferenceHostBlob("O1"));
  expectedOutput.push_back(reference->getReferenceHostBlob("O2"));

  {
    auto inLut1Blob = reference->getReferenceDeviceBlob("LUT1");
    auto inIdx1Blob = reference->getReferenceDeviceBlob("IDX1");
    at::Tensor LUT1 = makeATenTensor<caffe2::CUDAContext>(
        inLut1Blob, at::Backend::CUDA, at::ScalarType::Float);
    at::Tensor IDX1 = makeATenTensor<caffe2::CUDAContext>(
        inIdx1Blob, at::Backend::CUDA, at::ScalarType::Int);

    auto inLut2Blob = reference->getReferenceDeviceBlob("LUT2");
    auto inIdx2Blob = reference->getReferenceDeviceBlob("IDX2");
    at::Tensor LUT2 = makeATenTensor<caffe2::CUDAContext>(
        inLut2Blob, at::Backend::CUDA, at::ScalarType::Float);
    at::Tensor IDX2 = makeATenTensor<caffe2::CUDAContext>(
        inIdx2Blob, at::Backend::CUDA, at::ScalarType::Int);

    auto checkFun = [&](const std::vector<at::Tensor>& inputs,
                        const std::vector<at::Tensor>& outputs) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      double prec = 3e-7;
      std::cout << "Checking expected output relative precision @" << prec;
      at::Tensor tO =
          makeATenTensor(expectedOutput[0], at::Backend::CUDA, at::kFloat)
              .resize_({B, D});
      checkRtol(outputs[0].sub(tO), inputs, L1, prec);
      {
        at::Tensor tO =
            makeATenTensor(expectedOutput[1], at::Backend::CUDA, at::kFloat)
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
    if (useFlags && FLAGS_validate_proto) {
      validateProto(
          FLAGS_save_tuner_proto_prefix + std::string("/2LUT_cache") + suffix,
          tc,
          "_2LUT",
          inputs,
          checkFun);
    } else {
      std::vector<at::Tensor> outputs;
      Check(tc, "_2LUT", options, inputs, outputs, checkFun);
      if (useFlags) {
        autotune(
            FLAGS_save_tuner_proto_prefix + std::string("/2LUT_cache") + suffix,
            FLAGS_save_tuner_proto_prefix + std::string("/2LUT_best") + suffix,
            tc,
            "_2LUT",
            inputs,
            options,
            {options},
            checkFun);
      }
    }
  }
}

void ProductionModel::runC3(
    uint32_t B,
    uint32_t WX,
    uint32_t WY,
    const tc::CudaMappingOptions& options,
    bool useFlags) {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, WX});
  at::Tensor W = at::CUDA(at::kFloat).rand({WY, WX});

  auto checkFun = [&](const std::vector<at::Tensor>& inputs,
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
  if (useFlags && FLAGS_validate_proto) {
    validateProto(
        FLAGS_save_tuner_proto_prefix + std::string("/_C3_cache") + suffix,
        tc,
        "_C3",
        inputs,
        checkFun);
  } else {
    std::vector<at::Tensor> outputs;
    Check(tc, "_C3", options, inputs, outputs, checkFun);
    if (useFlags) {
      autotune(
          FLAGS_save_tuner_proto_prefix + std::string("/_C3_cache") + suffix,
          FLAGS_save_tuner_proto_prefix + std::string("/_C3_best") + suffix,
          tc,
          "_C3",
          inputs,
          options,
          {options},
          checkFun);
    }
  }
}

void ProductionModel::runMLP1(
    uint32_t B,
    uint32_t N,
    uint32_t M,
    const tc::CudaMappingOptions& options,
    bool useFlags) {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, M});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({M, N});
  at::Tensor B1 = at::CUDA(at::kFloat).rand({N});

  auto checkFun = [&](const std::vector<at::Tensor>& inputs,
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
  if (useFlags && FLAGS_validate_proto) {
    validateProto(
        FLAGS_save_tuner_proto_prefix + std::string("/mlp1_cache") + suffix,
        tc,
        "mlp1",
        inputs,
        checkFun);
  } else {
    std::vector<at::Tensor> outputs;
    Check(tc, "mlp1", options, inputs, outputs, checkFun);
    if (useFlags) {
      autotune(
          FLAGS_save_tuner_proto_prefix + std::string("/mlp1_cache") + suffix,
          FLAGS_save_tuner_proto_prefix + std::string("/mlp1_best") + suffix,
          tc,
          "mlp1",
          inputs,
          options,
          {options},
          checkFun);
    }
  }
}

void ProductionModel::runMLP3(
    uint32_t B,
    uint32_t N,
    uint32_t O,
    uint32_t P,
    uint32_t Q,
    const tc::CudaMappingOptions& options,
    bool useFlags) {
  at::Tensor I = at::CUDA(at::kFloat).rand({B, N});
  at::Tensor W2 = at::CUDA(at::kFloat).rand({O, N});
  at::Tensor B2 = at::CUDA(at::kFloat).rand({O});
  at::Tensor W3 = at::CUDA(at::kFloat).rand({P, O});
  at::Tensor B3 = at::CUDA(at::kFloat).rand({P});
  at::Tensor W4 = at::CUDA(at::kFloat).rand({Q, P});
  at::Tensor B4 = at::CUDA(at::kFloat).rand({Q});

  auto checkFun = [&](const std::vector<at::Tensor>& inputs,
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
def mlp3(float(B,N) I, float(O,N) W2, float(O) B2, float(P,O) W3, float(P) B3, float(Q,P) W4, float(Q) B4) -> (O2, O3, O4) {
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
  if (useFlags && FLAGS_validate_proto) {
    validateProto(
        FLAGS_save_tuner_proto_prefix + std::string("/mlp3_cache") + suffix,
        tc,
        "mlp3",
        inputs,
        checkFun);
  } else {
    std::vector<at::Tensor> outputs;

    Check(tc, "mlp3", options, inputs, outputs, checkFun);
    if (useFlags) {
      autotune(
          FLAGS_save_tuner_proto_prefix + std::string("/mlp3_cache") + suffix,
          FLAGS_save_tuner_proto_prefix + std::string("/mlp3_best") + suffix,
          tc,
          "mlp3",
          inputs,
          options,
          {options},
          checkFun);
    }
  }
}

TEST_F(ProductionModel, 1LUT) {
  auto B = FLAGS_B;
  auto D = FLAGS_D;
  auto L1 = FLAGS_L1;
  auto E1 = FLAGS_E1;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .tile(1, 32)
                     .mapToThreads({1, 32})
                     .mapToBlocks({128, 128})
                     .unroll(256);
  run1LUT(B, D, L1, E1, options, true);
}

TEST_F(ProductionModel, 1LUT_P100_autotuned_B_128_D_64_L1_50_E1_10000000) {
  uint32_t B = 128;
  uint32_t D = 64;
  uint32_t L1 = 50;
  uint32_t E1 = 10000000;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .fixParametersBeforeScheduling(true)
          .tile(1)
          .tileImperfectlyNested(false)
          .mapToBlocks(524288)
          .mapToThreads(153)
          .unroll(8);
  run1LUT(B, D, L1, E1, options);
}

TEST_F(ProductionModel, 1LUT_P100_autotuned_B_16_D_64_L1_50_E1_10000000) {
  uint32_t B = 16;
  uint32_t D = 64;
  uint32_t L1 = 50;
  uint32_t E1 = 10000000;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .fixParametersBeforeScheduling(false)
          .tile(1, 32)
          .tileImperfectlyNested(false)
          .mapToBlocks(128, 39, 156250)
          .mapToThreads(5, 32)
          .unroll(1);
  run1LUT(B, D, L1, E1, options);
}

TEST_F(ProductionModel, C21LUTReference) {
  int vB, vD, vL, vE;
  vB = FLAGS_B;
  vD = FLAGS_D;
  vL = FLAGS_L1;
  vE = FLAGS_E1;

  auto ws_init_func = [=](Workspace& w) {
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {vE, vD}, "LUT");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {vB, vL}, "I", 0, vE - 1);
    TestHarness::AddConstInput<int, CUDAContext>(w, {vB}, vL, "__lengths");
  };
  OperatorDef op_def =
      TestHarness::ConfigureCUDA("TcLUTOp", {"LUT", "I"}, {"O"});
  std::unique_ptr<TestHarness::OpTester> reference(
      new TestHarness::OpTester(op_def));
  reference->InitializeReference(ws_init_func);

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

TEST_F(ProductionModel, ATen1LUTReference) {
  std::cout << "No ATen1LUTReference available\n";
}

TEST_F(ProductionModel, 2LUT) {
  auto B = FLAGS_B;
  auto D = FLAGS_D;
  auto L1 = FLAGS_L1;
  auto L2 = FLAGS_L2;
  auto E1 = FLAGS_E1;
  auto E2 = FLAGS_E2;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .tile(1, 32)
                     .mapToThreads({1, 32})
                     .mapToBlocks({128, 128})
                     .unroll(256);
  run2LUT(B, D, L1, L2, E1, E2, options, true);
}

TEST_F(
    ProductionModel,
    2LUT_P100_autotuned_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000) {
  uint32_t B = 128;
  uint32_t D = 64;
  uint32_t L1 = 50;
  uint32_t E1 = 10000000;
  uint32_t L2 = 50;
  uint32_t E2 = 10000000;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .fixParametersBeforeScheduling(false)
          .tile(1, 256, 1250000)
          .tileImperfectlyNested(false)
          .mapToBlocks(5000000)
          .mapToThreads(306)
          .unroll(64);
  run2LUT(B, D, L1, L2, E1, E2, options);
}

TEST_F(
    ProductionModel,
    2LUT_P100_autotuned_B_16_D_64_L1_50_E1_10000000_L2_50_E2_10000000) {
  uint32_t B = 16;
  uint32_t D = 64;
  uint32_t L1 = 50;
  uint32_t E1 = 10000000;
  uint32_t L2 = 50;
  uint32_t E2 = 10000000;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .fixParametersBeforeScheduling(false)
          .tile(1, 64)
          .tileImperfectlyNested(false)
          .mapToBlocks(156250, 156250, 3)
          .mapToThreads(5, 32)
          .unroll(16);
  run2LUT(B, D, L1, L2, E1, E2, options);
}

TEST_F(ProductionModel, C22LUTReference) {
  int vB, vD, vL1, vE1, vL2, vE2;
  vB = FLAGS_B;
  vD = FLAGS_D;
  vL1 = FLAGS_L1;
  vE1 = FLAGS_E1;
  vL2 = FLAGS_L2;
  vE2 = FLAGS_E2;

  auto ws_init_func = [=](Workspace& w) {
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {vE1, vD}, "LUT1");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {vB, vL1}, "IDX1", 0, vE1 - 1);
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {vE2, vD}, "LUT2");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {vB, vL2}, "IDX2", 0, vE2 - 1);
    TestHarness::AddConstInput<int, CUDAContext>(w, {vB}, vL1, "__lengths1");
    TestHarness::AddConstInput<int, CUDAContext>(w, {vB}, vL2, "__lengths2");
  };
  OperatorDef op_def = TestHarness::ConfigureCUDA(
      "Tc2LUTOp", {"LUT1", "IDX1", "LUT2", "IDX2"}, {"O1", "O2"});
  std::unique_ptr<TestHarness::OpTester> reference(
      new TestHarness::OpTester(op_def));
  reference->InitializeReference(ws_init_func);

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

TEST_F(ProductionModel, ATen2LUTReference) {
  std::cout << "No ATen2LUTReference available\n";
}

TEST_F(ProductionModel, C3) {
  auto B = FLAGS_B;
  auto WX = FLAGS_WX;
  auto WY = FLAGS_WY;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .fixParametersBeforeScheduling(true)
                     .tile(32, 32, 32)
                     .mapToThreads({4, 32})
                     .mapToBlocks({128, 128})
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unroll(128);

  runC3(B, WX, WY, options, true);
}

TEST_F(ProductionModel, C3_P100_autotuned_B_128_WX_1000_WY_1024) {
  uint32_t B = 128;
  uint32_t WX = 1000;
  uint32_t WY = 1024;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(1024, 8, 125)
                     .mapToThreads(4, 32, 1)
                     .mapToBlocks(128, 128, 250)
                     .unroll(128)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(true);
  runC3(B, WX, WY, options);
}

TEST_F(ProductionModel, C3_P100_autotuned_B_16_WX_1000_WY_1024) {
  uint32_t B = 16;
  uint32_t WX = 1000;
  uint32_t WY = 1024;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(1024, 8, 125)
                     .mapToThreads(4, 32, 1)
                     .mapToBlocks(128, 128, 250)
                     .unroll(128)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(true);
  runC3(B, WX, WY, options);
}

TEST_F(ProductionModel, ATenC3Reference) {
  auto B = FLAGS_B;
  auto WX = FLAGS_WX;
  auto WY = FLAGS_WY;
  at::Tensor I = at::CUDA(at::kFloat).rand({B, WX});
  at::Tensor W = at::CUDA(at::kFloat).rand({WY, WX});

  Reference(
      [&]() { return I.mm(W.t()); },
      [&](at::Tensor& res) { mm_out(res, I, W.t()); });
}

TEST_F(ProductionModel, C2C3Reference) {
  auto B = FLAGS_B;
  auto WX = FLAGS_WX;
  auto WY = FLAGS_WY;

  auto ws_init_func = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {B, WX}, "I");
    AddInput(w, {WY, WX}, "W");
  };
  OperatorDef op_def =
      TestHarness::ConfigureCUDA("TcMatMulOp", {"I", "W"}, {"O"});
  std::unique_ptr<TestHarness::OpTester> reference(
      new TestHarness::OpTester(op_def));
  reference->InitializeReference(ws_init_func, {{"trans_b", 1}});

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

TEST_F(ProductionModel, MLP1) {
  auto B = FLAGS_B;
  auto N = FLAGS_N;
  auto M = FLAGS_M;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .fixParametersBeforeScheduling(true)
                     .tile(16, 16, 128)
                     .mapToThreads({16, 16})
                     .mapToBlocks({32, 32})
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unroll(1);
  runMLP1(B, N, M, options, true);
}

TEST_F(ProductionModel, MLP1_P100_autotuned_B_128_M_2000_N_128) {
  uint32_t B = 128;
  uint32_t M = 2000;
  uint32_t N = 128;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(4, 250)
          .mapToThreads(64, 8)
          .mapToBlocks(2000, 16)
          .unroll(32)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(true)
          .unrollCopyShared(false)
          .matchLibraryCalls(true);
  runMLP1(B, N, M, options);
}

TEST_F(ProductionModel, MLP1_P100_autotuned_B_16_M_2000_N_128) {
  uint32_t B = 16;
  uint32_t M = 2000;
  uint32_t N = 128;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(4, 250)
          .mapToThreads(64, 8)
          .mapToBlocks(2000, 16)
          .unroll(32)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(true)
          .unrollCopyShared(false)
          .matchLibraryCalls(true);
  runMLP1(B, N, M, options);
}

TEST_F(ProductionModel, ATenMLP1Reference) {
  auto B = FLAGS_B;
  auto N = FLAGS_N;
  auto M = FLAGS_M;
  at::Tensor I = at::CUDA(at::kFloat).rand({B, M});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({M, N});
  at::Tensor B1 = at::CUDA(at::kFloat).rand({N});

  Reference(
      [&]() { return I.mm(W1).add(B1).clamp_min(0); },
      [&](at::Tensor& res) { mm_out(res, I, W1).add(B1).clamp_min(0); });
}

TEST_F(ProductionModel, C2MLP1Reference) {
  auto B = FLAGS_B;
  auto M = FLAGS_M;
  auto N = FLAGS_N;
  auto ws_init_func = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {B, M}, "I");
    AddInput(w, {N, M}, "W1");
    AddInput(w, {N}, "B1");
  };
  OperatorDef op_def =
      TestHarness::ConfigureCUDA("TcFCReluOp", {"I", "W1", "B1"}, {"O1"});
  std::unique_ptr<TestHarness::OpTester> reference(
      new TestHarness::OpTester(op_def));
  reference->InitializeReference(ws_init_func);

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

TEST_F(ProductionModel, MLP3) {
  auto B = FLAGS_B;
  auto N = FLAGS_N;
  auto O = FLAGS_O;
  auto P = FLAGS_P;
  auto Q = FLAGS_Q;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .fixParametersBeforeScheduling(true)
                     .tile(16, 16, 128)
                     .mapToThreads({16, 16})
                     .mapToBlocks({32, 32})
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unroll(1);
  runMLP3(B, N, O, P, Q, options, true);
}

TEST_F(ProductionModel, MLP3_P100_autotuned_B_128_N_128_O_64_P_32_Q_2) {
  auto B = 128;
  auto N = 128;
  auto O = 64;
  auto P = 32;
  auto Q = 2;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(
                         tc::FusionStrategy::Preserve3Coincident)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(4, 8)
                     .mapToThreads(128, 4)
                     .mapToBlocks(128)
                     .unroll(2)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(false)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(false);
  runMLP3(B, N, O, P, Q, options);
}

TEST_F(ProductionModel, MLP3_P100_autotuned_B_16_M_2000_N_128_Q_2) {
  uint32_t B = 16;
  auto N = 128;
  auto O = 64;
  auto P = 32;
  auto Q = 2;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(
                         tc::FusionStrategy::Preserve3Coincident)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(4, 8)
                     .mapToThreads(128, 4)
                     .mapToBlocks(128)
                     .unroll(2)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(false)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(false);
  runMLP3(B, N, O, P, Q, options);
}

TEST_F(ProductionModel, ATenMLP3Reference) {
  auto B = FLAGS_B;
  auto N = FLAGS_N;
  auto O = FLAGS_O;
  auto P = FLAGS_P;
  auto Q = FLAGS_Q;
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

TEST_F(ProductionModel, C2MLP3Reference) {
  auto B = FLAGS_B;
  auto N = FLAGS_N;
  auto O = FLAGS_O;
  auto P = FLAGS_P;
  auto Q = FLAGS_Q;
  auto AddConstInput = TestHarness::AddConstInput<float, CUDAContext>;
  auto ws_init_func = [&](Workspace& w) {
    AddConstInput(w, vector<TIndex>{B, N}, 1., "I");
    AddConstInput(w, vector<TIndex>{O, N}, 1., "W1");
    AddConstInput(w, vector<TIndex>{O}, 1., "B1");
    AddConstInput(w, vector<TIndex>{P, O}, 1., "W2");
    AddConstInput(w, vector<TIndex>{P}, 1., "B2");
    AddConstInput(w, vector<TIndex>{Q, P}, 1., "W3");
    AddConstInput(w, vector<TIndex>{Q}, 1., "B3");
  };
  OperatorDef op_def = TestHarness::ConfigureCUDA(
      "Tc3FCReluOp",
      {"I", "W1", "B1", "W2", "B2", "W3", "B3"},
      {"O1", "O2", "O3"});
  std::unique_ptr<TestHarness::OpTester> reference(
      new TestHarness::OpTester(op_def));
  reference->InitializeReference(ws_init_func);

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
