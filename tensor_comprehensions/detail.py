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

from tensor_comprehensions.tclib import CompilationCache
from tensor_comprehensions.tclib import MappingOptions
from tensor_comprehensions.tclib import MappingOptionsCache
from tensor_comprehensions.tclib import Tuner
from tensor_comprehensions.tclib import TunerConfig
from tensor_comprehensions.tclib import compile

class TcBuilder():
    def __init__(
            self,
            tc="",
            forward_entry_point="", forward_force_reinforcement_tuning=False,
            backward_entry_point="", backward_force_reinforcement_tuning=False,
            check_output_shapes=True,
            tuner_cache_file="",
            tuner_config=TunerConfig(),
            debug=False):
        assert isinstance(tc, str), type(tc)
        assert isinstance(forward_entry_point, str), type(forward_entry_point)
        assert isinstance(forward_force_reinforcement_tuning, bool), type(forward_force_reinforcement_tuning)
        assert isinstance(backward_entry_point, str), type(backward_entry_point)
        assert isinstance(backward_force_reinforcement_tuning, bool), type(backward_force_reinforcement_tuning)
        assert isinstance(check_output_shapes, bool), type(tuner_cache_file)
        assert isinstance(tuner_cache_file, str), type(tuner_cache_file)
        assert isinstance(tuner_config, TunerConfig), type(tuner_config)

        self.tc = tc
        self.forward_entry_point = forward_entry_point
        self.forward_force_reinforcement_tuning = forward_force_reinforcement_tuning
        self.backward_entry_point = backward_entry_point
        self.backward_force_reinforcement_tuning = backward_force_reinforcement_tuning
        self.check_output_shapes = check_output_shapes
        self.tuner_cache_file = tuner_cache_file
        self.tuner_config = tuner_config
        self.debug = debug
        self.compilation_cache = CompilationCache(self.tc)

    def compileOrTune(self, entry_point="", force_reinforcement_tuning=False, inputs=()):
        if self.debug:
            print("On Tc: {}\ncompile def {}, force_reinforcement_tuning {}, inputs: {}".format(
                self.tc, entry_point, force_reinforcement_tuning, "".join("{}/{}, ".format(
                    t.size().__str__(), t.stride().__str__()) for t in inputs)))

        if not self.compilation_cache.is_compiled(entry_point, inputs):
            cache = MappingOptionsCache(self.tuner_cache_file)
            mapping_options = None
            base_options_list = cache.load(self.tc, entry_point, inputs, 1)
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
                    mapping_options = MappingOptions('naive')
                else:
                    mapping_options = base_options_list[0]

                tuner = Tuner(self.tc, self.tuner_cache_file)
                mapping_options = tuner.tune(
                    entry_point, inputs, mapping_options, self.tuner_config)

            self.compilation_cache.compile(entry_point, inputs, mapping_options)

class TcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tc_builder, *inputs):
        ctx.tc_builder = tc_builder
        ctx.save_for_backward(*inputs)
        if tc_builder.debug:
            assert isinstance(tc_builder, TcBuilder), type(tc_builder)
            assert isinstance(inputs, tuple), type(inputs)

        tc_builder.compileOrTune(
            entry_point=tc_builder.forward_entry_point,
            force_reinforcement_tuning=tc_builder.forward_force_reinforcement_tuning,
            inputs=inputs)
        eval_fun = (
            tc_builder.compilation_cache.run
            if tc_builder.check_output_shapes
            else tc_builder.compilation_cache.unchecked_run)
        outputs = eval_fun(tc_builder.forward_entry_point, inputs, ())

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *gradients):
        # Getting into backward measured at ~150us
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
        # Measured the following @25us overhead when contiguous does not copy data
        inputs = tuple(ctx.saved_tensors) + tuple(
            t.contiguous() for t in gradients)
        tc_builder.compileOrTune(
            entry_point=tc_builder.backward_entry_point,
            force_reinforcement_tuning=tc_builder.backward_force_reinforcement_tuning,
            inputs=inputs)
        eval_fun = (
            tc_builder.compilation_cache.run
            if tc_builder.check_output_shapes
            else tc_builder.compilation_cache.unchecked_run)
        outputs = eval_fun(tc_builder.backward_entry_point, inputs, ())
         # PyTorch convention: need an extra None return for the ctx
        return (None, *tuple(outputs))

class MultiTcBuilder():
    def __init__(self,
                 tc="",
                 forward_entry_points=(), forward_input_indices=(()), forward_force_reinforcement_tunings=(),
                 backward_entry_points=(), backward_input_indices=(()), backward_force_reinforcement_tunings=(),
                 check_output_shapes=True,
                 tuner_cache_file="",
                 tuner_config=TunerConfig(),
                 debug=False):
        assert isinstance(tc, str), type(tc)
        assert isinstance(forward_entry_points, tuple), type(forward_entry_points)
        assert isinstance(forward_input_indices, tuple), type(forward_input_indices)
        assert isinstance(forward_force_reinforcement_tunings, tuple), type(forward_force_reinforcement_tunings)
        assert isinstance(backward_entry_points, tuple), type(backward_entry_points)
        assert isinstance(backward_input_indices, tuple), type(backward_input_indices)
        assert isinstance(backward_force_reinforcement_tunings, tuple), type(backward_force_reinforcement_tunings)
        assert isinstance(check_output_shapes, bool), type(tuner_cache_file)
        assert isinstance(tuner_cache_file, str), type(tuner_cache_file)
        assert isinstance(tuner_config, TunerConfig), type(tuner_config)

        self.tc = tc
        self.forward_entry_points = forward_entry_points
        self.forward_input_indices = forward_input_indices
        self.forward_force_reinforcement_tunings = forward_force_reinforcement_tunings
        self.backward_entry_points = backward_entry_points
        self.backward_input_indices = backward_input_indices
        self.backward_force_reinforcement_tunings = backward_force_reinforcement_tunings
        self.check_output_shapes = check_output_shapes
        self.tuner_cache_file = tuner_cache_file
        self.tuner_config = tuner_config
        self.debug = debug
        self.compilation_cache = CompilationCache(self.tc)

        assert len(self.forward_input_indices) <= len(self.forward_entry_points)
        assert (len(self.forward_force_reinforcement_tunings) <=
                len(self.forward_entry_points))
        assert (len(self.backward_input_indices) <=
                len(self.backward_entry_points))
        assert (len(self.backward_force_reinforcement_tunings) <=
                len(self.backward_entry_points))

    def compileOrTune(self, entry_point="", force_reinforcement_tuning=False, inputs=()):
        if self.debug:
            print("On Tc: {}\ncompile def {}, force_reinforcement_tuning {}, inputs: {}".format(
                self.tc, entry_point, force_reinforcement_tuning, "".join("{}/{}, ".format(
                    t.size().__str__(), t.stride().__str__()) for t in inputs)))

        if not self.compilation_cache.is_compiled(entry_point, inputs):
            cache = MappingOptionsCache(self.tuner_cache_file)
            mapping_options = None
            base_options_list = cache.load(self.tc, entry_point, inputs, 1)
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
                    mapping_options = MappingOptions('naive')
                else:
                    mapping_options = base_options_list[0]

                tuner = Tuner(self.tc, self.tuner_cache_file)
                mapping_options = tuner.tune(
                    entry_point, inputs, mapping_options, self.tuner_config)

            self.compilation_cache.compile(entry_point, inputs, mapping_options)

class MultiTcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tc_builder, *inputs):
        ctx.tc_builder = tc_builder
        ctx.save_for_backward(*inputs)
        if tc_builder.debug:
            assert isinstance(tc_builder, MultiTcBuilder), type(tc_builder)
            assert isinstance(inputs, tuple), type(inputs)

        # "Pad" force_reinforcement_tunings and input_indices
        # with None to signal default behavior.
        # When None is specified for the reinforcement_tunings, we do not tune
        # When None is specified for the input_indices, we use all inputs
        forward_entry_points = tc_builder.forward_entry_points
        forward_force_reinforcement_tunings = (
            tc_builder.forward_force_reinforcement_tunings +
            (len(forward_entry_points) -
             len(tc_builder.forward_force_reinforcement_tunings)) * (None, )
        )
        forward_input_indices =  (
            tc_builder.forward_input_indices +
            (len(forward_entry_points) -
             len(tc_builder.forward_input_indices)) * (None, )
        )

        all_outputs = []
        for d in zip(forward_entry_points,
                     forward_force_reinforcement_tunings,
                     forward_input_indices):
            # Select default behaviors for each entry point
            entry_point = d[0]
            force_reinforcement_tuning = d[1] if d[1] is not None else False
            inputs_for_this_tc_def = (
                tuple(inputs[idx] for idx in d[2]) if d[2] is not None else
                tuple(inputs)
            )

            tc_builder.compileOrTune(
                entry_point = entry_point,
                force_reinforcement_tuning = force_reinforcement_tuning,
                inputs = inputs_for_this_tc_def)
            eval_fun = (
                tc_builder.compilation_cache.run
                if tc_builder.check_output_shapes
                else tc_builder.compilation_cache.unchecked_run)
            outputs = eval_fun(entry_point, inputs_for_this_tc_def, ())

            # Augment inputs with outputs for case of multiple TCs
            inputs = inputs + tuple(outputs)
            all_outputs = all_outputs + outputs

        return tuple(all_outputs)

    @staticmethod
    def backward(ctx, *gradients):
        # Getting into backward measured at ~150us
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
        # Measured the following @25us overhead when contiguous does not copy data
        inputs = tuple(ctx.saved_tensors) + tuple(
            t.contiguous() for t in gradients)

        # "Pad" force_reinforcement_tunings and input_indices
        # with None to signal default behavior.
        # When None is specified for the reinforcement_tunings, we do not tune
        # When None is specified for the input_indices, we use all inputs
        backward_entry_points = tc_builder.backward_entry_points
        backward_force_reinforcement_tunings = (
            tc_builder.backward_force_reinforcement_tunings +
            (len(backward_entry_points) -
             len(tc_builder.backward_force_reinforcement_tunings)) * (None, )
        )
        backward_input_indices =  (
            tc_builder.backward_input_indices +
            (len(backward_entry_points) -
             len(tc_builder.backward_input_indices)) * (None, )
        )

        # PyTorch convention: need an extra None return for the ctx
        all_outputs = [None]
        for d in zip(backward_entry_points,
                     backward_force_reinforcement_tunings,
                     backward_input_indices):
            # Select default behaviors for each entry point
            entry_point = d[0]
            force_reinforcement_tuning = d[1] if d[1] is not None else False
            inputs_for_this_tc_def = (
                tuple(inputs[idx] for idx in d[2]) if d[2] is not None else
                tuple(inputs)
            )

            # Measured the above @8us overhead
            tc_builder.compileOrTune(
                entry_point = entry_point,
                force_reinforcement_tuning = force_reinforcement_tuning,
                inputs = inputs_for_this_tc_def)
            # Measured the above @18us overhead
            eval_fun = (
                tc_builder.compilation_cache.run
                if tc_builder.check_output_shapes
                else tc_builder.compilation_cache.unchecked_run)
            outputs = eval_fun(entry_point, inputs_for_this_tc_def, ())

            # Augment inputs with outputs for case of multiple TCs
            inputs = inputs + tuple(outputs)
            all_outputs = all_outputs + outputs

        return tuple(all_outputs)
