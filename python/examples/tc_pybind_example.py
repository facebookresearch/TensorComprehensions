# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import time
import torch

################################################################################
# The purpose of these examples is to demonstrate the usage of the python
# bindings to build a simple, low-overhead, python abstraction.
# We demonstrate the bindings by building a series of examples leading to a
# MultiTcFunction abstraction for PyTorch autograd.
################################################################################

################################################################################
# 0. Initializations
################################################################################
from tensor_comprehensions.tclib import MappingOptions

# Define a timing function to print some results
def time_tc(iters, prepend, runFun, tc_entry_point, inputs):
    timesCPU = []
    timesCPUAndGPU = []
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.clock()
        outputs = runFun(tc_entry_point, inputs)
        timesCPU.append(time.clock() - start)
        torch.cuda.synchronize()
        timesCPUAndGPU.append(time.clock() - start)
    print("#################################################################")
    timesCPU = sorted(timesCPU)
    print("{} Total CPU time to launch kernel: min {}us, p50 {}us, p90 {}us, max {}us".format(
        prepend,
        int(timesCPU[0] * 1e6),
        int(timesCPU[int(len(timesCPU) // 2)] * 1e6),
        int(timesCPU[int((len(timesCPU) * 9) // 10)] * 1e6),
        int(timesCPU[len(timesCPU) - 1] * 1e6),
    ))
    timesCPUAndGPU = sorted(timesCPUAndGPU)
    print("{} Total CPU launch + GPU kernel time: min {}us, p50 {}us, p90 {}us, max {}us".format(
        prepend,
        int(timesCPUAndGPU[0] * 1e6),
        int(timesCPUAndGPU[int(len(timesCPUAndGPU) // 2)] * 1e6),
        int(timesCPUAndGPU[int((len(timesCPUAndGPU) * 9) // 10)] * 1e6),
        int(timesCPUAndGPU[len(timesCPUAndGPU) - 1] * 1e6),
    ))

# Define a TC string for matmul and some input torch cuda tensors
mm = """
def matmul(float(M,N) A, float(N,K) B) -> (C) {
    C(m, k) +=! A(m, r_n) * B(r_n, k)
}
def matmul_grad(float(M,N) A, float(N,K) B, float(M,K) d_O) -> (d_A, d_B) {
    d_A(m, n) +=! d_O(  m, r_k) * B(  n, r_k)
    d_B(n, k) +=! d_O(r_m,   k) * A(r_m,   n)
}
"""
mat1, mat2 = torch.randn(300, 400).cuda(), torch.randn(400, 500).cuda()

################################################################################
# 1. Use the simple high-overhead compile/run C++ API
#    If one can keep state in their layer or wishes to experiment with TC,
#    this is a simple entry point.
#    If state cannot be kept, be aware that this API has a non-trivial overhead
#    when outputs sizes need to be inferred and outputs allocated.
#    Compilation itself has a prohibitive cost and needs to be memoized either
#    by holding on to the executor or by using the low-overhead abstraction, see
#    below
################################################################################
from tensor_comprehensions.tclib import compile

executor = compile(mm, "matmul", (mat1, mat2), MappingOptions('naive'))
outputs = executor.run((mat1, mat2), ())
outputs = executor.unchecked_run((mat1, mat2), tuple(outputs))
time_tc(100,
        "simple API\t",
        lambda name, ins: executor.unchecked_run(ins, tuple(outputs)),
        "matmul",
        (mat1, mat2))
time_tc(100,
        "simple API (with allocation overhead)\t",
        lambda name, ins: executor.unchecked_run(ins, ()),
        "matmul",
        (mat1, mat2))

################################################################################
# 2. Use the C++ API to build a low-overhead compilation cache and time it
################################################################################
from tensor_comprehensions.tclib import CompilationCache

compilation_cache = CompilationCache(mm)
# Compilation returns an allocated tuple of outputs with the proper shapes.
# Allocation overhead is negligible compared to compilation overhead.
compilation_cache.compile("matmul", (mat1, mat2), MappingOptions('naive'))
# Run once without timing
compilation_cache.unchecked_run("matmul", (mat1, mat2), ())
# unchecked_run on  tensors
time_tc(100,
        "raw unchecked_run naive options\t",
        lambda name, ins: compilation_cache.unchecked_run(name, ins, ()),
        "matmul",
        (mat1, mat2))

################################################################################
# 3. Short tuning run saving to file then load the best option to create a
#    compilation cache
################################################################################
from tensor_comprehensions.tclib import Tuner
from tensor_comprehensions.tclib import MappingOptionsCache
from tensor_comprehensions.tclib import TunerConfig

import uuid
unique_filename = "/tmp/" + str(uuid.uuid4())
print("Tune with cache @", unique_filename)
print("Note that if you pass a fixed filename, you can reinforce an " +
      "existing tuning state")

tuner = Tuner(mm, unique_filename)
top1  = tuner.tune(
    "matmul",
    (mat1, mat2),
    MappingOptions('naive'),
    TunerConfig(threads = 8, pop_size = 25, generations = 3, devices = "0"))
cache = MappingOptionsCache(unique_filename)
top10 = cache.load(mm, "matmul", (mat1, mat2), 10)
assert top1.__str__() == top10[0].__str__()

# Compile and run with the new options
compilation_cache.compile("matmul", (mat1, mat2), top1)
time_tc(100,
        "raw unchecked_run tuned options\t",
        lambda name, ins: compilation_cache.unchecked_run(name, ins, ()),
        "matmul",
        (mat1, mat2))

################################################################################
# 4. Simple TC builder
################################################################################
from tensor_comprehensions.detail import TcBuilder

# TcBuilder exposes a compileOrTune function that can be used independently,
# just use it to benchmark
tcb = TcBuilder(
    tc = mm,
    tuner_cache_file = "/tmp/some_cache_file_we_reuse_for_perf_reinforcement",
    tuner_config = TunerConfig(threads = 8,
                               pop_size = 25,
                               generations = 3,
                               devices = "0"))
tcb.compileOrTune(entry_point = "matmul", inputs = (mat1, mat2))
time_tc(100,
        "TcBuilder unchecked_run\t",
        lambda name, ins: tcb.compilation_cache.unchecked_run(name, ins, ()),
        "matmul",
        (mat1, mat2))

################################################################################
# 5. Simple torch.autograd.Function backed by TcBuilder
################################################################################
from tensor_comprehensions.detail import TcFunction

tcb = TcBuilder(
    tc = mm,
    forward_entry_point = "matmul",
    forward_force_reinforcement_tuning = False,
    backward_entry_point = "matmul_grad",
    backward_force_reinforcement_tuning = False,
    check_output_shapes = False,
    tuner_cache_file = "/tmp/some_cache_file_we_reuse_for_perf_reinforcement",
    tuner_config = TunerConfig(threads = 8,
                               pop_size = 25,
                               generations = 3,
                               devices = "0"),
)

time_tc(100,
        "TcFunction forward unchecked_run\t",
        lambda name, ins: TcFunction.apply(tcb, *ins),
        "matmul",
        (mat1, mat2))

# This is the PyTorch way of triggering backward:
#   forward input tensors require gradients,
#   then call backward (either with a grad sized tensor or
#      pass through a loss and compute the backward of a Variable)
mat1.requires_grad = True
mat2.requires_grad = True
outputs = TcFunction.apply(tcb, mat1, mat2)

# For example purposes, use retain_graph
# retain_graph = True prevents freeing the buffers when performing backward
# see e.g. https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
grad_sized_tensor = outputs[0].clone()
outputs[0].backward(grad_sized_tensor, retain_graph = True)

# Flip to true to measure the overhead of getting into backward
# Measured at ~150us on matmul_grad
time_tc(100,
        "TcFunction backward unchecked_run\t",
        lambda name, ins: outputs[0].backward(grad_sized_tensor, retain_graph = True),
        "matmul",
        (mat1, mat2))

# For completeness, also compute the backward of a Variable
# This path will trigger reallocations / copies at t.contiguous() time
v = outputs[0].sum()
v.backward(retain_graph = True)

################################################################################
# 6. Multi-TC builder
################################################################################
from tensor_comprehensions.detail import MultiTcBuilder

# TcBuilder exposes a compileOrTune function that can be used independently,
# just use it to benchmark
tcb = MultiTcBuilder(
    tc = mm,
    tuner_cache_file = "/tmp/some_cache_file_we_reuse_for_perf_reinforcement",
    tuner_config = TunerConfig(threads = 8,
                               pop_size = 25,
                               generations = 3,
                               devices = "0"))

tcb.compileOrTune(entry_point = "matmul", inputs = (mat1, mat2))
time_tc(100,
        "MultiTcBuilder unchecked_run\t",
        lambda name, ins: tcb.compilation_cache.unchecked_run(name, ins, ()),
        "matmul",
        (mat1, mat2))

################################################################################
# 7. Multi-TC torch.autograd.Function backed by MultiTcBuilder
################################################################################
from tensor_comprehensions.detail import MultiTcFunction

tcb = MultiTcBuilder(
    tc = mm,
    # Note the extra commas inside the tuple arguments.
    # They force python to pass arguments by tuples instead of single value
    forward_entry_points = ("matmul", ),
    forward_input_indices = ((0, 1), ),
    forward_force_reinforcement_tunings = (False, ),
    backward_entry_points = ("matmul_grad", ),
    backward_input_indices = ((0, 1, 2), ),
    backward_force_reinforcement_tunings = (False, ),
    check_output_shapes = False,
    tuner_cache_file = "/tmp/some_cache_file_we_reuse_for_perf_reinforcement",
    tuner_config = TunerConfig(threads = 8,
                               pop_size = 25,
                               generations = 3,
                               devices = "0"),
)

time_tc(100,
        "MultiTcFunction forward unchecked_run\t",
        lambda name, ins: MultiTcFunction.apply(tcb, *ins),
        "matmul",
        (mat1, mat2))

# This is the PyTorch way of triggering backward:
#   forward input tensors require gradients,
#   then call backward (either with a grad sized tensor or
#      pass through a loss and compute the backward of a Variable)
mat1.requires_grad = True
mat2.requires_grad = True
outputs = MultiTcFunction.apply(tcb, mat1, mat2)

# For example purposes, use retain_graph
# retain_graph = True prevents freeing the buffers when performing backward
# see e.g. https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
grad_sized_tensor = outputs[0].clone()
outputs[0].backward(grad_sized_tensor, retain_graph = True)

# Flip to true to measure the overhead of getting into backward
# Measured at ~150us on matmul_grad
time_tc(100,
        "MultiTcFunction backward unchecked_run\t",
        lambda name, ins: outputs[0].backward(grad_sized_tensor, retain_graph = True),
        "matmul",
        (mat1, mat2))

# For completeness, also compute the backward of a Variable
# This path will trigger reallocations / copies at t.contiguous() time
v = outputs[0].sum()
v.backward(retain_graph = True)
