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

# Importing pytorch before trying to dlopen tclib is currently required
# because of:
#   https://github.com/pytorch/pytorch/issues/6097
# This probably requires a patch on the pytorch side to remove the dependency
import torch

from tensor_comprehensions.tclib import logtostderr
from tensor_comprehensions.tclib import debug_lang
from tensor_comprehensions.tclib import debug_halide
from tensor_comprehensions.tclib import debug_tc_mapper
from tensor_comprehensions.tclib import debug_tuner
from tensor_comprehensions.tclib import dump_cuda

from tensor_comprehensions.tclib import CompilationCache
from tensor_comprehensions.tclib import MappingOptions
from tensor_comprehensions.tclib import MappingOptionsCache
from tensor_comprehensions.tclib import Tuner
from tensor_comprehensions.tclib import TunerConfig
from tensor_comprehensions.tclib import compile

from tensor_comprehensions.detail import TcBuilder, MultiTcBuilder, TcFunction, MultiTcFunction

def assert_almost_equal(diff, inputs, operations=1, precision=1e-7):
    max_value = 0.0
    for inp in inputs:
        max_value = max(float(inp.abs().max()), max_value)
    max_diff = float(diff.abs().max())
    assert max_diff <= operations * precision * max_value, (
        "error at relative precision: {}, #operations: {}, max_value: {}, max_diff: {}".format(
            precision, operations, max_value, max_diff)
    )

class TC(object):
    def __init__(self, tc, entry_point, fallback):
        self.tc = tc
        self.entry_point = entry_point
        self.builder = TcBuilder(tc, entry_point)
        self.fallback_mapping_options = fallback

    def __call__(self, *inputs, **kwargs):
        r"""Runs the defined TC function on given inputs.
        The TC must have been compiled for the specific input sizes.

        Args:
            *inputs (required):
                PyTorch Tensors or Variables that TC should
                execute on. The inputs should be passed in the order they
                are also passed in the definition of TC function.

        Kwargs:
            *outputs:
                List of tensors that will be written by the TC. This is used
                in particular to write in place. If output tensors are omitted,
                new outputs will be allocated and returned at each call.

            unchecked (bool):
                Run in low-latency unchecked mode where failsafe size and stride
                checks are omitted.
        """

        inputs = tuple(inputs) if len(inputs) > 1 else tuple(inputs, )
        outputs = ()
        if "outputs" in kwargs and kwargs["outputs"] is not None:
            outputs = tuple(kwargs["outputs"])

        if not self.builder.compilation_cache.is_compiled(
                self.entry_point, inputs):
            assert isinstance(self.fallback_mapping_options, MappingOptions), (
                "TC was not compiled for inputs of sizes:\n\t{}\n".format(
                    "".join("{}, ".format(i.size().__str__()) for i in inputs)).join(
                """
                   Additionally, TC was defined without explicit fallback
                   MappingOptions. You should construct the TC with an explicit
                   MappingOptions object or call compile (or tune) explicitly
                   on your TC before trying to run it.
                """))
            self.compile(self.fallback_mapping_options, *inputs)

        if "unchecked" in kwargs and kwargs["unchecked"]:
            return self.builder.compilation_cache.unchecked_run(
                self.entry_point, inputs, ())

        return self.builder.compilation_cache.run(
            self.entry_point, inputs, ())

    def compile(self, options, *inputs):
        inputs = tuple(inputs) if len(inputs) > 1 else tuple(inputs, )
        self.builder.compilation_cache.compile(
            self.entry_point, inputs, options)

    def tune(self, *inputs, **kwargs):
        starting_options = (
            kwargs["starting_options"]
            if "starting_options" in kwargs else None)
        tuner_config = (
            kwargs["tuner_config"]
            if "tuner_config" in kwargs and kwargs["tuner_config"] is not None
            else self.builder.tuner_config)
        load_cache_filename = (
            kwargs["cache_filename"]
            if "cache_filename" in kwargs and
            isinstance(kwargs["cache_filename"], str)
            else "")
        store = (
            True
            if "store_to_cache" in kwargs and kwargs["store_to_cache"]
            else False)
        store_cache_filename = (
            kwargs["cache_filename"]
            if "cache_filename" in kwargs and
            isinstance(kwargs["cache_filename"], str) and store
            else "")

        inputs = tuple(inputs) if len(inputs) > 1 else tuple(inputs, )
        cache = MappingOptionsCache(load_cache_filename)
        base_options = cache.load(self.tc, self.entry_point, inputs, 1)
        base_options = (
            base_options[0]
            if len(base_options) > 0 else starting_options)
        assert base_options is not None, """
        Tuner could not find MappingOptions to start from.
        Either pass a starting_options parameters or a MappingOptionsCache filename to load from.
        """
        tuner = Tuner(self.tc, store_cache_filename)
        best_options = tuner.tune(
            self.entry_point,
            inputs,
            base_options,
            tuner_config
        )

        # Compile best options to set the current for the current (entry point, inputs)
        self.builder.compilation_cache.compile(
            self.entry_point, inputs, best_options)

        return best_options

def define(tc, entry_point, fallback=None):
    return TC(tc, entry_point, fallback)

class TCWithFunction(object):
    def __init__(self, builder):
        self.builder = builder

    def __call__(self, *inputs):
        return MultiTcFunction.apply(self.builder, *inputs)

def define_with_autograd(
        tc,
        forward_entry_points,
        backward_entry_points,
        cache_filename,
        forward_input_indices=(()),
        backward_input_indices=(()),
        forward_force_reinforcement_tunings=(),
        backward_force_reinforcement_tunings=(),
        check_output_shapes=True,
        tuner_config=TunerConfig(),
        debug=False):
    return TCWithFunction(MultiTcBuilder(
        tc,
        forward_entry_points, forward_input_indices, forward_force_reinforcement_tunings,
        backward_entry_points, backward_input_indices, backward_force_reinforcement_tunings,
        check_output_shapes,
        cache_filename,
        tuner_config,
        debug))

__all__ = [
    # Debugging functions, pass True to activate
    'logtostderr',
    'debug_lang',
    'debug_halide',
    'debug_tc_mapper',
    'debug_tuner',
    'dump_cuda',
    # Classes exposed by the tclib
    'CompilationCache',
    'MappingOptions',
    'MappingOptionsCache',
    'Tuner',
    'TunerConfig',
    # Functions exposed by the tclib
    'compile',
    # Python-side functionality
    'define',
    'define_with_autograd',
    'assert_almost_equal',
    'TC',
    'TcBuilder',
    'MultiTcBuilder',
    'TcFunction',
    'MultiTcFunction',
]
