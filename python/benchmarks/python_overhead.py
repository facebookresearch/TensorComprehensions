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
import tempfile
import time

import torch

from utils import time_tc

from tensor_comprehensions.tclib import MappingOptions
from tensor_comprehensions.tclib import Tuner
from tensor_comprehensions.tclib import MappingOptionsCache
from tensor_comprehensions.tclib import TunerConfig
from tensor_comprehensions.tclib import compile
from tensor_comprehensions.tclib import CompilationCache

import tensor_comprehensions as tc

################################################################################
# The purpose of these benchmarks is to measure the overhead of our python
# abstraction.
################################################################################

################################################################################
# 0. Initializations
################################################################################
# Define a TC string for matmul and some input torch cuda tensors
mm = """
def matmul(float(M,N) A, float(N,K) B) -> (C) {
    C(m, k) +=! A(m, r_n) * B(r_n, k)
}
def matmul_agrad(float(N,K) B, float(M,K) d_C) -> (d_A) {
    d_A(m, n) +=! d_C(  m, r_k) * B(  n, r_k)
}
def matmul_bgrad(float(M,N) A, float(M,K) d_C) -> (d_B) {
    d_B(n, k) +=! d_C(r_m,   k) * A(r_m,   n)
}
"""
A, B = (
    torch.randn(300, 400, device='cuda', requires_grad=True),
    torch.randn(400, 500, device='cuda', requires_grad=True))

compilation_cache = CompilationCache(mm)

tuner_config = TunerConfig().threads(8).pop_size(25).generations(3).devices("0")

################################################################################
# 1. Use the simple high-overhead compile/run C++ API
#    If one can keep state in their layer or wishes to experiment with TC,
#    this is a simple entry point.
#    If state cannot be kept, be aware that this API has a non-trivial overhead
#    when outputs sizes need to be inferred and outputs allocated.
#    Compilation itself has a prohibitive cost and needs to be memoized either
#    by holding on to the executor or by using the low-overhead abstraction, see
#    below.
################################################################################
executor = compile(mm, "matmul", (A, B), MappingOptions('naive'))
C = executor.run((A, B))

time_tc(100,
        "simple API (in place)\t",
        lambda name, ins: executor.unchecked_run(ins, (C,)),
        "matmul",
        (A, B))

time_tc(100,
        "simple API (with allocation overhead)\t",
        lambda name, ins: executor.unchecked_run(ins),
        "matmul",
        (A, B))

################################################################################
# 2. Use the C++ API to build a low-overhead compilation cache and time it
################################################################################
# Compilation returns an allocated tuple of outputs with the proper shapes.
# Allocation overhead is negligible compared to compilation overhead.
compilation_cache.compile("matmul", (A, B), MappingOptions('naive'))

# Run once without timing
compilation_cache.unchecked_run("matmul", (A, B))

# unchecked_run on tensors
time_tc(100,
        "raw unchecked_run naive options\t",
        lambda name, ins: compilation_cache.unchecked_run(name, ins),
        "matmul",
        (A, B))

################################################################################
# 3. Short tuning run saving to file then load the best option to create a
#    compilation cache
################################################################################
with tempfile.NamedTemporaryFile() as cache_file:
    tuner = Tuner(mm, cache_file.name)
    top1  = tuner.tune(
        "matmul",
        (A, B),
        MappingOptions('naive'),
        tuner_config)
    cache = MappingOptionsCache(cache_file.name)
    top10 = cache.load(mm, "matmul", (A, B), 10)
    assert top1.__str__() == top10[0].__str__()

    # Compile and run with the new options
    compilation_cache.compile("matmul", (A, B), top1)
    time_tc(100,
            "raw unchecked_run tuned options\t",
            lambda name, ins: compilation_cache.unchecked_run(name, ins),
            "matmul",
            (A, B))

################################################################################
# 4. Simple torch.autograd.Function
################################################################################
T = tc.define(
    mm,
    tc.make_autotuned_options_factory(starting_options='naive',
                                      tuner_config=tuner_config))

def backward(A, B, d_C):
    d_A = T.matmul_agrad(B, d_C, unchecked=True)
    d_B = T.matmul_bgrad(A, d_C, unchecked=True)
    return d_A, d_B

matmul_function = tc.make_autograd(
    lambda A, B: T.matmul(A, B, unchecked=True),
    backward
)

# Run once to trigger automatic tuning and compilation then time
# For example purposes, use retain_graph
# retain_graph = True prevents freeing the buffers when performing backward
# see e.g. https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
C = matmul_function(A, B)
grad_sized_tensor = C.clone()
C.backward(grad_sized_tensor, retain_graph = True)

time_tc(100,
        "TC forward unchecked_run\t",
        lambda name, ins: T.matmul(*ins, unchecked=True),
        "matmul",
        (A, B))

time_tc(100,
        "TC backward unchecked_run\t",
        lambda name, ins: backward(*ins),
        "matmul",
        (A, B, grad_sized_tensor))

time_tc(100,
        "Autograd forward unchecked_run\t",
        lambda name, ins: matmul_function(*ins),
        "matmul",
        (A, B))

time_tc(100,
        "Autograd backward unchecked_run\t",
        lambda name, ins: C.backward(*ins, retain_graph = True),
        "matmul",
        (grad_sized_tensor, ))
