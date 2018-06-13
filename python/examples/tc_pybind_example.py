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

dump_backward_overhead = False

################################################################################
# The purpose of these examples is to demonstrate the usage of the python
# bindings to build a simple, low-overhead, python abstraction.
# We demonstrate the bnidings by building a series of examples leading to a
# MultiTcFunction abstraction for PyTorch autograd.
################################################################################

################################################################################
# 0. Initializations
################################################################################
from tensor_comprehensions.tclib import MappingOptions

# Define a timing function to print some results
def time_tc(iters, prepend, runFun, tc_name, inputs):
    global dump_backward_overhead

    timesCPU = []
    timesCPUAndGPU = []
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.clock()
        if dump_backward_overhead:
            dump_backward_overhead = time.clock()
        outputs = runFun(tc_name, inputs)
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

executor = compile(mm, "matmul", (mat1, mat2), MappingOptions())
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
compilation_cache.compile("matmul", (mat1, mat2), MappingOptions())
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
    MappingOptions(),
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
class TcBuilder():
    def __init__(self,
                 tc = "",
                 forward_name = "", forward_force_reinforcement_tuning = False,
                 backward_name = "", backward_force_reinforcement_tuning = False,
                 check_output_shapes = True,
                 tuner_cache_file = "",
                 tuner_config = TunerConfig(),
                 debug = False):
        if debug:
            assert isinstance(tc, str), type(tc)
            assert isinstance(forward_name, str), type(forward_name)
            assert isinstance(forward_force_reinforcement_tuning, bool), type(forward_force_reinforcement_tuning)
            assert isinstance(backward_name, str), type(backward_name)
            assert isinstance(backward_force_reinforcement_tuning, bool), type(backward_force_reinforcement_tuning)
            assert isinstance(check_output_shapes, bool), type(tuner_cache_file)
            assert isinstance(tuner_cache_file, str), type(tuner_cache_file)
            assert isinstance(tuner_config, TunerConfig), type(tuner_config)

        self.tc = tc
        self.forward_name = forward_name
        self.forward_force_reinforcement_tuning = forward_force_reinforcement_tuning
        self.backward_name = backward_name
        self.backward_force_reinforcement_tuning = backward_force_reinforcement_tuning
        self.check_output_shapes = check_output_shapes
        self.tuner_cache_file = tuner_cache_file
        self.tuner_config = tuner_config
        self.debug = debug
        self.compilation_cache = CompilationCache(self.tc)

    def compileOrTune(self, name = "", force_reinforcement_tuning = False, inputs = ()):
        if self.debug:
            print("On Tc: {}\ncompile def {}, force_reinforcement_tuning {}, inputs: {}".format(
                self.tc, name, force_reinforcement_tuning, "".join("{}/{}, ".format(
                    t.size().__str__(), t.stride().__str__()) for t in inputs)))

        if not self.compilation_cache.is_compiled(name, inputs):
            cache = MappingOptionsCache(self.tuner_cache_file)
            mapping_options = None
            base_options_list = cache.load(self.tc, name, inputs, 1)
            if len(base_options_list) > 0 and not force_reinforcement_tuning:
                mapping_options = base_options_list[0]
                if self.debug:
                    print("Found best options in {}:\n{}".format(
                        self.tuner_cache_file, mapping_options))
            else:
                if self.debug:
                    print("########################################################"
                          "########################################################")
                    print("force_reinforcement_tuning = {} was specified, {} options loaded from "
                          "{}".format(
                              force_reinforcement_tuning, len(base_options_list), self.tuner_cache_file))
                    print("Starting a tuning run (abort it with Ctrl+C when "
                          "performance is satisfactory.\nYou can always reinforce "
                          "the results later by passing a proper tuner cache file "
                    "and specifying force_reinforcement_tuning=True)")
                    print("########################################################"
                          "########################################################")

                if len(base_options_list) == 0:
                    mapping_options = MappingOptions()
                else:
                    mapping_options = base_options_list[0]

                tuner = Tuner(self.tc, self.tuner_cache_file)
                mapping_options = tuner.tune(name, inputs, mapping_options, self.tuner_config)

            self.compilation_cache.compile(name, inputs, mapping_options)

# TcBuilder exposes a compileOrTune function that can be used independently,
# just use it to benchmark
tcb = TcBuilder(
    tc = mm,
    tuner_cache_file = "/tmp/some_cache_file_we_reuse_for_perf_reinforcement",
    tuner_config = TunerConfig(threads = 8,
                               pop_size = 25,
                               generations = 3,
                               devices = "0"))

tcb.compileOrTune(name = "matmul", inputs = (mat1, mat2))
time_tc(100,
        "TcBuilder unchecked_run\t",
        lambda name, ins: tcb.compilation_cache.unchecked_run(name, ins, ()),
        "matmul",
        (mat1, mat2))

################################################################################
# 5. Simple torch.autograd.Function backed by TcBuilder
################################################################################
class TcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tc_builder, *inputs):
        ctx.tc_builder = tc_builder
        ctx.save_for_backward(*inputs)
        if tc_builder.debug:
            assert isinstance(tc_builder, TcBuilder), type(tc_builder)
            assert isinstance(inputs, tuple), type(inputs)

        tc_builder.compileOrTune(name = tc_builder.forward_name,
                                 force_reinforcement_tuning = tc_builder.forward_force_reinforcement_tuning,
                                 inputs = inputs)
        eval_fun = (
            tc_builder.compilation_cache.run
            if tc_builder.check_output_shapes
            else tc_builder.compilation_cache.unchecked_run)
        outputs = eval_fun(tc_builder.forward_name, inputs, ())

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *gradients):
        # Getting into backward measured at ~150us on matmul_grad
        if dump_backward_overhead:
            print("IN BACKWARD:", time.clock())
        tc_builder = ctx.tc_builder
        if tc_builder.debug:
            assert isinstance(tc_builder, TcBuilder), type(tc_builder)
            assert isinstance(gradients, tuple), type(gradients)
        # The `contiguous` calls are needed because depending on the
        # operation that follows in the forward pass, we may receive
        # gradients with stride 0 during backward
        # In particular, this occurs because expand / sum operations are dual
        # of each other and expand has traditionally been implemented by just
        # setting strides to 0.
        # There are multiple potential ways to address this in TC:
        #   1. (currently) punt and call contiguous which will reallocate
        #      and copy. This is inefficient but the occurrence is supposedly
        #      rare and the implementation is trivial;
        #   2. detect non-comformable strides and crash, the user can then
        #      implement a fused version of the operator and displace the
        #      problem. This may or may not be enough;
        #   3. allow non-Fortran subarray style shapes if they are readonly and
        #      use a DeviceTensor-style abstraction which puts actual strides
        #      behind the operator[]
        #
        # Measured the following @25us overhead when doing nothing on matmul_grad
        inputs = tuple(ctx.saved_variables) + tuple(
            t.contiguous() for t in gradients)
        tc_builder.compileOrTune(name = tc_builder.backward_name,
                                 force_reinforcement_tuning = tc_builder.backward_force_reinforcement_tuning,
                                 inputs = inputs)
        eval_fun = (
            tc_builder.compilation_cache.run
            if tc_builder.check_output_shapes
            else tc_builder.compilation_cache.unchecked_run)
        outputs = eval_fun(tc_builder.backward_name, inputs, ())
         # PyTorch convention: need an extra None return for the ctx
        return (None, *tuple(outputs))

tcb = TcBuilder(
    tc = mm,
    forward_name = "matmul",
    forward_force_reinforcement_tuning = False,
    backward_name = "matmul_grad",
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
dump_backward_overhead = False
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
class MultiTcBuilder():
    def __init__(self,
                 tc = "",
                 forward_names = (), forward_input_indices = (()), forward_force_reinforcement_tunings = (),
                 backward_names = (), backward_input_indices = (()), backward_force_reinforcement_tunings = (),
                 check_output_shapes = True,
                 tuner_cache_file = "",
                 tuner_config = TunerConfig(),
                 debug = False):
        if debug:
            assert isinstance(tc, str), type(tc)
            assert isinstance(forward_names, tuple), type(forward_names)
            assert isinstance(forward_input_indices, tuple), type(forward_input_indices)
            assert isinstance(forward_force_reinforcement_tunings, tuple), type(forward_force_reinforcement_tunings)
            assert isinstance(backward_names, tuple), type(backward_names)
            assert isinstance(backward_input_indices, tuple), type(backward_input_indices)
            assert isinstance(backward_force_reinforcement_tunings, tuple), type(backward_force_reinforcement_tunings)
            assert isinstance(check_output_shapes, bool), type(tuner_cache_file)
            assert isinstance(tuner_cache_file, str), type(tuner_cache_file)
            assert isinstance(tuner_config, TunerConfig), type(tuner_config)

        self.tc = tc
        self.forward_names = forward_names
        self.forward_input_indices = forward_input_indices
        self.forward_force_reinforcement_tunings = forward_force_reinforcement_tunings
        self.backward_names = backward_names
        self.backward_input_indices = backward_input_indices
        self.backward_force_reinforcement_tunings = backward_force_reinforcement_tunings
        self.check_output_shapes = check_output_shapes
        self.tuner_cache_file = tuner_cache_file
        self.tuner_config = tuner_config
        self.debug = debug
        self.compilation_cache = CompilationCache(self.tc)

    def compileOrTune(self, name = "", force_reinforcement_tuning = False, inputs = ()):
        if self.debug:
            print("On Tc: {}\ncompile def {}, force_reinforcement_tuning {}, inputs: {}".format(
                self.tc, name, force_reinforcement_tuning, "".join("{}/{}, ".format(
                    t.size().__str__(), t.stride().__str__()) for t in inputs)))

        if not self.compilation_cache.is_compiled(name, inputs):
            cache = MappingOptionsCache(self.tuner_cache_file)
            mapping_options = None
            base_options_list = cache.load(self.tc, name, inputs, 1)
            if len(base_options_list) > 0 and not force_reinforcement_tuning:
                mapping_options = base_options_list[0]
                if self.debug:
                    print("Found best options in {}:\n{}".format(
                        self.tuner_cache_file, mapping_options))
            else:
                if self.debug:
                    print("########################################################"
                          "########################################################")
                    print("force_reinforcement_tuning = {} was specified, {} options loaded from "
                          "{}".format(
                              force_reinforcement_tuning, len(base_options_list), self.tuner_cache_file))
                    print("Starting a tuning run (abort it with Ctrl+C when "
                          "performance is satisfactory.\nYou can always reinforce "
                          "the results later by passing a proper tuner cache file "
                    "and specifying force_reinforcement_tuning=True)")
                    print("########################################################"
                          "########################################################")

                if len(base_options_list) == 0:
                    mapping_options = MappingOptions()
                else:
                    mapping_options = base_options_list[0]

                tuner = Tuner(self.tc, self.tuner_cache_file)
                mapping_options = tuner.tune(name, inputs, mapping_options, self.tuner_config)

            self.compilation_cache.compile(name, inputs, mapping_options)

# TcBuilder exposes a compileOrTune function that can be used independently,
# just use it to benchmark
tcb = MultiTcBuilder(
    tc = mm,
    tuner_cache_file = "/tmp/some_cache_file_we_reuse_for_perf_reinforcement",
    tuner_config = TunerConfig(threads = 8,
                               pop_size = 25,
                               generations = 3,
                               devices = "0"))

tcb.compileOrTune(name = "matmul", inputs = (mat1, mat2))
time_tc(100,
        "MultiTcBuilder unchecked_run\t",
        lambda name, ins: tcb.compilation_cache.unchecked_run(name, ins, ()),
        "matmul",
        (mat1, mat2))

################################################################################
# 7. Multi-TC torch.autograd.Function backed by MultiTcBuilder
################################################################################
class MultiTcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tc_builder, *inputs):
        ctx.tc_builder = tc_builder
        ctx.save_for_backward(*inputs)
        if tc_builder.debug:
            assert isinstance(tc_builder, MultiTcBuilder), type(tc_builder)
            assert isinstance(inputs, tuple), type(inputs)
        all_outputs = []
        for d in zip(tc_builder.forward_names,
                     tc_builder.forward_force_reinforcement_tunings,
                     tc_builder.forward_input_indices):
            name = d[0]
            force_reinforcement_tuning = d[1]
            inputs_for_this_tc_def = tuple(inputs[idx] for idx in d[2])
            tc_builder.compileOrTune(name = name,
                                     force_reinforcement_tuning = force_reinforcement_tuning,
                                     inputs = inputs_for_this_tc_def)
            eval_fun = (
                tc_builder.compilation_cache.run
                if tc_builder.check_output_shapes
                else tc_builder.compilation_cache.unchecked_run)
            outputs = eval_fun(name, inputs_for_this_tc_def, ())

            all_outputs = all_outputs + outputs
        return tuple(all_outputs)

    @staticmethod
    def backward(ctx, *gradients):
        # Getting into backward measured at ~150us on matmul_grad
        if dump_backward_overhead:
            print("Cost to enter backward:", int((time.clock() - dump_backward_overhead) * 1e6))
        tc_builder = ctx.tc_builder
        if tc_builder.debug:
            assert isinstance(tc_builder, MultiTcBuilder), type(tc_builder)
            assert isinstance(gradients, tuple), type(gradients)
        # The `contiguous` calls are needed because depending on the
        # operation that follows in the forward pass, we may receive
        # gradients with stride 0 during backward
        # In particular, this occurs because expand / sum operations are dual
        # of each other and expand has traditionally been implemented by just
        # setting strides to 0.
        # There are multiple potential ways to address this in TC:
        #   1. (currently) punt and call contiguous which will reallocate
        #      and copy. This is inefficient but the occurrence is supposedly
        #      rare and the implementation is trivial;
        #   2. detect non-comformable strides and crash, the user can then
        #      implement a fused version of the operator and displace the
        #      problem. This may or may not be enough;
        #   3. allow non-Fortran subarray style shapes if they are readonly and
        #      use a DeviceTensor-style abstraction which puts actual strides
        #      behind the operator[]
        #
        # Measured the following @25us overhead when doing nothing on matmul_grad
        inputs = tuple(ctx.saved_variables) + tuple(
            t.contiguous() for t in gradients)
        # PyTorch convention: need an extra None return for the ctx
        all_outputs = [None]
        for d in zip(tc_builder.backward_names,
                     tc_builder.backward_force_reinforcement_tunings,
                     tc_builder.backward_input_indices):
            name = d[0]
            force_reinforcement_tuning = d[1]
            inputs_for_this_tc_def = tuple(inputs[idx] for idx in d[2])
            # Measured the above @8us overhead on matmul_grad
            tc_builder.compileOrTune(name = name,
                                     force_reinforcement_tuning = force_reinforcement_tuning,
                                     inputs = inputs_for_this_tc_def)
            # Measured the above @18us overhead on matmul_grad
            eval_fun = (
                tc_builder.compilation_cache.run
                if tc_builder.check_output_shapes
                else tc_builder.compilation_cache.unchecked_run)
            outputs = eval_fun(name, inputs_for_this_tc_def, ())
            all_outputs = all_outputs + outputs
        return tuple(all_outputs)

tcb = MultiTcBuilder(
    tc = mm,
    # Note the extra commas inside the tuple arguments.
    # They force python to pass arguments by tuples instead of single value
    forward_names = ("matmul", ),
    forward_input_indices = ((0, 1), ),
    forward_force_reinforcement_tunings = (False, ),
    backward_names = ("matmul_grad", ),
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
dump_backward_overhead = False
time_tc(100,
        "MultiTcFunction backward unchecked_run\t",
        lambda name, ins: outputs[0].backward(grad_sized_tensor, retain_graph = True),
        "matmul",
        (mat1, mat2))

# For completeness, also compute the backward of a Variable
# This path will trigger reallocations / copies at t.contiguous() time
v = outputs[0].sum()
v.backward(retain_graph = True)
