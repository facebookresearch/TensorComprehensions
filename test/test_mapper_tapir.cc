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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>

#include "tc/aten/aten.h"
#include "tc/core/cpu/cpu_tc_executor.h"
#include "tc/core/mapping_options.h"
#include "tc/core/polyhedral/codegen_llvm.h"
#include "tc/core/polyhedral/llvm_jit.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/scope_guard.h"

#include "test_harness_aten.h"

using namespace std;

using namespace tc;
using namespace tc::polyhedral;
using namespace tc::polyhedral::detail;

TEST(TapirCodegen, BasicParallel) {
  string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
  C(n, m) = A(n, m) + B(n, m)
}
)TC";
  auto N = 40;
  auto M = 24;

  auto ctx = isl::with_exceptions::globalIslCtx();
  auto scop = polyhedral::Scop::makeScop(ctx, tc);
  auto context = scop->makeContext(
      std::unordered_map<std::string, int>{{"N", N}, {"M", M}});
  scop = Scop::makeSpecializedScop(*scop, context);
  SchedulerOptionsProto sop;
  SchedulerOptionsView sov(sop);
  scop = Scop::makeScheduled(*scop, sov);
  Jit jit;
  auto mod = jit.codegenScop("kernel_anon", *scop);
  auto fn = mod->getFunction("kernel_anon");

  std::set<string> calledFunctions;
  for (llvm::inst_iterator I = llvm::inst_begin(fn), E = llvm::inst_end(fn);
       I != E;
       ++I) {
    if (llvm::CallInst* c = llvm::dyn_cast<llvm::CallInst>(&*I)) {
      if (auto called = c->getCalledFunction()) {
        calledFunctions.insert(called->getName());
      }
    }
  }

  ASSERT_NE(0u, calledFunctions.count("__cilkrts_get_tls_worker"));
  ASSERT_NE(0u, calledFunctions.count("__cilkrts_bind_thread_1"));
  ASSERT_NE(0u, calledFunctions.count("llvm.stacksave"));
  ASSERT_NE(0u, calledFunctions.count("__cilkrts_sync"));

  auto fptr =
      (void (*)(float*, float*, float*))jit.getSymbolAddress("kernel_anon");

  at::Tensor A = at::CPU(at::kFloat).rand({N, M});
  at::Tensor B = at::CPU(at::kFloat).rand({N, M});
  at::Tensor C = at::CPU(at::kFloat).rand({N, M});
  at::Tensor Cc = A + B;
  fptr(A.data<float>(), B.data<float>(), C.data<float>());

  checkRtol(Cc - C, {A, B}, N * M);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  initialize_llvm();
  return RUN_ALL_TESTS();
}
