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

import os, sys, pdb, uuid, logging, subprocess

import torch
from torch.autograd import Variable

from tensor_comprehensions.tc import ATenCompilationUnit
from tensor_comprehensions.tc import set_logtostderr, set_debug_lang, set_debug_halide, set_debug_tc_mapper, set_debug_cuda, set_debug_tuner, set_dump_cuda
from tensor_comprehensions.torch_tc.tc_function import TCFunction, unpack_variables, get_tensors, make_contiguous
from tensor_comprehensions.autotuner import ATenAutotuner
from tensor_comprehensions.mapping_options import Options

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


# these are quick options for finishing autotuning
autotuner_settings = {
    "threads": 32, "generations": 1, "pop_size": 2,
}

# TC prunes autotuning for kernels which require < 256 threads. So to tune small
# size kernels, we set the min kernel threads to 1
small_sizes_autotuner_settings = {
    "threads": 32, "generations": 1, "pop_size": 2, "tuner_min_launch_total_threads": 1,
}

###############################################################################
# Set global debugging flags
###############################################################################
class SetDebugFlags(object):
    def __init__(self, **kwargs):
        self.set_gflags(**kwargs)

    def set_gflags(
        self, debug_lang=False, debug_halide=False, debug_tc_mapper=False,
        debug_cuda=False, debug_tuner=False, dump_cuda=False, **kwargs
    ):
        set_logtostderr(True)
        set_debug_lang(debug_lang)
        set_debug_halide(debug_halide)
        set_debug_tc_mapper(debug_tc_mapper)
        set_debug_cuda(debug_cuda)
        set_debug_tuner(debug_tuner)
        set_dump_cuda(dump_cuda)


###############################################################################
# Some helper functions
###############################################################################
def check_cache_file_exists(cache_file):
    # for autotuning, we save two files: .cuda and .options, we will check that
    # these two files exists for the validity of cache
    if os.path.exists(cache_file + ".options") and os.path.exists(cache_file + ".cuda"):
        return True
    return False


def get_options_from_cache_file(name, *inputs, **kwargs):
    options = None
    if "cache" in kwargs and kwargs["cache"] and isinstance(kwargs["cache"], str):
        cache_file = kwargs["cache"]
        assert "type" in kwargs, "layer type not specified: forward/backward"
        if "training" in kwargs and kwargs["training"]:
            if (kwargs["type"] == "backward"):
                cache_file = cache_file + "_backward"
        if "tuner" in kwargs and check_cache_file_exists(cache_file):
            tuner = kwargs["tuner"]
            loaded_options = tuner.load(cache_file, name, list(inputs), 1)
            if len(loaded_options) > 0:
                options = loaded_options[0]
        elif check_cache_file_exists(cache_file):
            tuner = TcAutotuner(kwargs["tc_lang"])
            options = tuner.load(cache_file, name, list(inputs))
    return options


# get the options from kwargs or construct the naive options
# argument type="forward" or "backward"
def get_options_from_kwargs(name, *inputs, **kwargs):
    # now the options can be a tuple (if training) or it will be just options
    # (only forward)
    options = None
    if "options" in kwargs and kwargs["options"] is not None:
        options = kwargs["options"]
        assert "type" in kwargs, "layer type not specified: forward/backward"
        if isinstance(options, list) and len(options) == 2:
            options = options[0] if kwargs["type"] == "forward" else options[1]
        elif "training" in kwargs and kwargs["training"] and kwargs["type"] == "backward":
            logger.warning('Same mapping options will be used to run backward layer, please pass backward mapping options for better performance.')
    elif "cache" in kwargs and kwargs["cache"]:
        options = get_options_from_cache_file(name, *inputs, **kwargs)
    elif "options_cache" in kwargs and kwargs["options_cache"]:
        options_cache = kwargs["options_cache"]
        assert kwargs["type"] is not None, "layer type not specified: forward/backward"
        options = options_cache[kwargs["type"]]
        logger.info("Tuned kernel options found, using those options")

    if options is None:
        options = Options("naive")
        logger.warning("No mapping options passed, 'naive' type mapping options will be used and will likely have bad performance. See help(your_layer.__call__) for setting mapping options.")
    if not isinstance(options, Options):
        options = Options(options)
    return options


def get_tc_hash_key(name, *inputs):
    sizes_key = "_".join(["_".join(map(str, list(inp.size()))) for inp in inputs])
    hash_key = "{}_{}".format(name, sizes_key)
    return hash_key


def get_tc_names_from_kwargs(**kwargs):
    backward, backward_name = False, None
    if "training" in kwargs and kwargs["training"]:
        backward = True
        assert "backward" in kwargs and kwargs["backward"] is not None, \
            "You forgot to specify the name of backward TC. Training requires backward layer TC as well."
        backward_name = kwargs["backward"]
    assert "name" in kwargs and kwargs["name"] is not None, \
        "You forgot to specify which TC to run, please pass the name in define()"
    name = kwargs["name"]
    return name, backward_name


def validate_input(*inputs):
    # at the moment, TC can only take tensors as the input, we validate that
    # the inputs are all tensors
    for inp in inputs:
        assert torch.is_tensor(inp) or isinstance(inp, Variable), \
            "Incorrect input type: One of the inputs is not a tensor / Variable"


# Autotuner helper function
def validate_autotuner_input(*inputs):
    # for autotuning, we accept tensors, Variable, tuple as inputs
    for inp in inputs:
        assert torch.is_tensor(inp) or isinstance(inp, Variable) or isinstance(inp, tuple), \
            "Incorrect input type: One of the inputs is not a tensor/Variable/tuple"


# Autotuner helper function
def get_options_from_kwargs_and_tuner_cache(name, cache_file, options_cache, *inputs, **kwargs):
    options = None
    if "options" in kwargs and kwargs["options"] is not None:
        options = kwargs["options"]
        assert "type" in kwargs, "tuning layer type not specified: forward/backward"
        # if we pass separate options for forward/backward, we use them otherwise
        # use the same options
        if isinstance(options, list) and len(options) == 2:
            options = options[0] if kwargs["type"] == "forward" else options[1]
    elif cache_file and isinstance(kwargs["cache"], str):
        options = get_options_from_cache_file(name, *inputs, **kwargs)
    elif options_cache and kwargs["type"] in options_cache and options_cache[kwargs["type"]] is not None:
        options = options_cache[kwargs["type"]]
        logger.info("Kernel was previously tuned, seeding the current tuning with those mapping options")

    if options is None:
        options = Options("naive")
        logger.warning("Using 'naive' type mapping options for autotuning. See help(your_layer.autotune) for how to set mapping options.")
    if not isinstance(options, Options):
        options = Options(options)
    return options

###############################################################################
# TC autotuner class - ATen
###############################################################################
class TcAutotuner(object):
    def __init__(self, tc_lang, **kwargs):
        # tuner_cache will look like:
        # hash_key -> {"forward": options1, "backward": options2}
        self.tuner_cache = {}
        self.kwargs = kwargs
        self.tc_lang = tc_lang
        self.autotuner = ATenAutotuner(tc_lang)
        self.set_autotuner_parameters(**kwargs)

    def set_autotuner_parameters(
        self, pop_size=20, crossover_rate=80, mutation_rate=7, generations=10,
        number_elites=1, threads=8, gpus="0", restore_from_proto=False,
        restore_number=10, log_generations=False, save_best_candidates_count=10,
        tuner_min_launch_total_threads=64, **kwargs
    ):
        self.autotuner.pop_size(pop_size)
        self.autotuner.crossover_rate(crossover_rate)
        self.autotuner.mutation_rate(mutation_rate)
        self.autotuner.generations(generations)
        self.autotuner.number_elites(number_elites)
        self.autotuner.threads(threads)
        self.autotuner.gpus(gpus)
        self.autotuner.restore_from_proto(restore_from_proto)
        self.autotuner.restore_number(restore_number)
        self.autotuner.log_generations(log_generations)
        self.autotuner.save_best_candidates_count(save_best_candidates_count)
        self.autotuner.tuner_min_launch_total_threads(tuner_min_launch_total_threads)

    # We need to pass the inputs so that we can load the correct options from
    # the cache that correspond to the inputs sizes. This is useful when the
    # cache may contain multiple kernels and multiple sizes for each kernel
    def load(self, filename, tc_name, inputs, num_candidates=1):
        best_options = self.autotuner.load(filename, tc_name, inputs, num_candidates)
        if num_candidates == 1:
            return best_options[0]
        return best_options

    # if the cache_file is not "" then the tuning results would be saved to file
    def tune_and_store(self, tc_name, inputs, mapping_options, cache_file=""):
        options = mapping_options
        if not isinstance(options, Options):
            options = Options(options)
        try:
            best_options = self.autotuner.tune(cache_file, tc_name, inputs, options, [options])
            return best_options
        except Exception as e:
            logger.error('Raised exception: {}'.format(e))
            return options

    def autotune(self, *inputs, **kwargs):
        input_tensors = get_tensors(list(inputs))
        kwargs.update(self.kwargs)
        name, backward_name = get_tc_names_from_kwargs(**kwargs)
        kwargs.pop("name", None)
        backward = True if backward_name is not None else False
        hash_key = get_tc_hash_key(name, *input_tensors)
        # lookup for the options in the cache. Whenever we make the call to
        # autotune, tuning must happen. But if the kernel has been tuned earlier
        # then we can use previous options to seed the tuning.
        if hash_key in self.tuner_cache:
            options_cache = self.tuner_cache[hash_key]
        else:
            options_cache = {}

        # we give priority to the options user might have passed via file, or
        # Options object.
        cache_file = ""
        if "cache" in kwargs and kwargs["cache"]:
            if isinstance(kwargs["cache"], bool):
                hash_key = get_tc_hash_key(name, *input_tensors)
                cache_file = "/tmp/{}_{}".format(hash_key, str(uuid.uuid4()))
            elif isinstance(kwargs["cache"], str):
                cache_file = kwargs["cache"]
            logger.info('Autotuning cache will be saved to: {}.cuda/options'.format(cache_file))
        else:
            logger.warning("Autotuning results won't be cached. 'cache' option is not set")

        # we will first run the autotuning on the forward layer, the inputs are given
        # for that, we will tune those
        kwargs["type"] = "forward"
        # we pass this tuner object so we can load from file without having to
        # create special object
        kwargs["tuner"] = self.autotuner
        options = get_options_from_kwargs_and_tuner_cache(name, cache_file, options_cache, *input_tensors, **kwargs)
        forward_best_options = self.tune_and_store(
            name, input_tensors, mapping_options=options, cache_file=cache_file
        )
        # update the cache with the options
        options_cache["forward"] = forward_best_options
        if not backward:
            self.tuner_cache[hash_key] = options_cache
            return forward_best_options

        # now, we have to tune the backward layer, for that, we need to run
        # the forward layer first, get its output,
        logger.info('Autotuning the backward layer now')
        cu = TcCompilationUnit()
        cu.define(self.tc_lang)

        if "options" in kwargs:
            orig_options = kwargs["options"]
            kwargs["options"] = forward_best_options
            outputs = cu.compile_and_run(name, input_tensors, **kwargs)
            kwargs["options"] = orig_options
        else:
            outputs = cu.compile_and_run(name, input_tensors, options=forward_best_options, **kwargs)
        # now that we have the outputs of the forward pass, we have the inputs
        # for the backward layer and we can now tune the backward layer
        reorder_function = kwargs["reorder_function"] if "reorder_function" in kwargs else None
        rearranged_outputs = list(outputs)
        if reorder_function is not None:
            rearranged_outputs = reorder_function(list(outputs))
        inputs = make_contiguous(unpack_variables(input_tensors + list(rearranged_outputs)))

        if cache_file:
            cache_file = cache_file + "_backward"
            logger.info('Backwards autotuning cache will be saved to: {}.cuda/options'.format(cache_file))
        kwargs["type"] = "backward"
        options = get_options_from_kwargs_and_tuner_cache(backward_name, cache_file, options_cache, *inputs, **kwargs)
        backward_best_options = self.tune_and_store(
            backward_name, inputs, mapping_options=options, cache_file=cache_file
        )
        # update the cache with the options
        options_cache["backward"] = backward_best_options
        self.tuner_cache[hash_key] = options_cache
        return [forward_best_options, backward_best_options]


###############################################################################
# TC engine - ATen based
###############################################################################
class TcCompilationUnit(object):

    def __init__(self):
        self.cu = ATenCompilationUnit()
        self.tc_lang = None
        self.compilation_cache = {}

    def define(self, tc_lang):
        self.tc_lang = tc_lang
        self.cu.define(tc_lang)

    # we could have multiple TC definitions and want to run one of them
    def compile(self, name, inputs, **kwargs):
        # append the language so that we can use it for creating Autotuner object
        # to load options cache
        kwargs["tc_lang"] = self.tc_lang
        if "type" not in kwargs:
            kwargs["type"] = 'forward'
        options = get_options_from_kwargs(name, *inputs, **kwargs)
        handle = self.cu.compile(name, inputs, options)
        return handle

    def run(self, handle, name, inputs, **kwargs):
        outputs = []
        if "outputs" in kwargs and kwargs["outputs"] is not None:
            outputs = kwargs["outputs"]
            if not isinstance(outputs, list):
                outputs = [outputs]
        self.cu.run(name, inputs, outputs, handle)
        return outputs

    def compile_and_run(self, name, inputs, **kwargs):
        handle = self.compile(name, inputs, **kwargs)
        return self.run(handle, name, inputs, **kwargs)

    def manual_cuda_injection(
        self, name, injected_kernel_name, cuda_code, inputs, grid, block
    ):
        self.cu.inject_cuda(
            name, injected_kernel_name, cuda_code, inputs, grid, block
        )


###############################################################################
# User Facing Proxy object
###############################################################################
class TcUnit(object):

    def __init__(self, lang, **kwargs_define):
        self.cu = TcCompilationUnit()
        self.cu.define(lang)
        self.kwargs_define = kwargs_define
        self.lang = lang
        self.tuner = None   # this tuner maintains a cache for kernels/input sizes tuned so far

    def __call__(self, *inputs, **kwargs):
        r"""Runs the defined TC language on given inputs.

        Args:
            *inputs (required):
                PyTorch Tensors or Variables that TC should
                execute on. The inputs should be passed in the order they
                are also passed in the definition of TC language.

            options (optional):
                Kernel mapping options of type :attr:`tc.Options`. These options
                provide mapping for kernel like grid, blocks, memory etc. It
                is recommended to always pass kernel options. The options can be
                obtained by:

                * Autotuning, (recommended) OR

                * You can create `Options` object by chosing the closely matching "type" of kernel. For example:

                 .. code::

                     import tensor_comprehensions as tc
                     options = tc.Options(type)

                where :attr:`type` is a string with value one of below:

                * :attr:`pointwise`:  if kernel resembles a pointwise operation

                * :attr:`mlp`: if kernel resembles an Linear layer operation

                * :attr:`conv`: if kernel resembles a convolution operation

                * :attr:`group_conv`: if kernel resembles a group convolution operation

                * :attr:`naive`: if none of the above, then chose naive *Default*

                If no :attr:`Options` are passed, the naive options will be used which
                might not yield great performance.

            outputs (optional):
                List of Pytorch tensors/Variables. The number of outputs is
                the same as defined in the TC language and are in the same
                order as in TC language. You can chose to allocate the outputs
                tensors/Variables beforehand. Most common use case is to
                reuse output from a previous operation.

            cache (string, optional):
                A string denoting the absolute filepath which
                contains the mapping options for the kernel. Such file can be created by running
                autotuning.

                 If :attr:`training` = True, then the backward options will be obtained
                 from file cache + '_backward'. For the backward, separate filename
                 is not accepted for now.

            grid (int, 3D list):
                If :attr:`inject_kernel` is `True`, then user
                needs to specify the kernel grid options for running it. TC
                will simply use those options and will not add any optimizations

            block (int, 3D list):
                If :attr:`inject_kernel` is `True`, then user
                needs to specify the kernel `block` options for running it. TC
                will simply use those options and will not add any optimizations

            reorder_function (optional):
                If :attr:`training` is set to true in :attr:`define` call,
                then TC infers the inputs for backward layer for compilation
                (1st time the layer is run). The backward layer should typically
                contain the grad_outputs of the forward layer. The backward
                layer should take TC forward inputs + grad_outputs in the same
                order as the forward TC takes inputs and emits outputs. If
                the order of the outputs is changed, or some output grad are
                not required in backwards, then you can pass a function which
                can reorder/drop the layer grad_outputs according to backwards
                layer inputs your TC needs. The function should return a :attr:`list`.

        Returns:
            List of PyTorch tensors/Variables which is the output of running
            TC layer. The number of outputs is the same as defined in the TC
            language and are in the same order as in TC language.

        Example:
            >>> LANG = MATMUL_LANG
            >>> matmul = tc.define(lang, name="matmul")
            >>> mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
            >>> out = matmul(mat1, mat2, options=Options("mlp"))
        """
        try:
            validate_input(*inputs)
            kwargs.update(self.kwargs_define)
            name, backward_name = get_tc_names_from_kwargs(**kwargs)
            kwargs.pop("name", None)
            backward = True if backward_name is not None else False

            hash_key = get_tc_hash_key(name, *inputs)

            if self.tuner and self.tuner.tuner_cache and hash_key in self.tuner.tuner_cache:
                options_cache = self.tuner.tuner_cache[hash_key]
            else:
                options_cache = {}

            kwargs["options_cache"] = options_cache
            if hash_key in self.cu.compilation_cache:
                tc_info = self.cu.compilation_cache[hash_key]
            else:
                tc_info = {}
                kwargs["type"] = "forward"
                input_tensors = unpack_variables(list(inputs))

                if "inject_kernel" in kwargs and "cuda_code" in kwargs:
                    assert "grid" in kwargs and "block" in kwargs, \
                        "For manual cuda injection, please specify the grid and block settings"
                    self.cu.manual_cuda_injection(
                        name, kwargs["inject_kernel"], kwargs["cuda_code"],
                        input_tensors, kwargs["grid"], kwargs["block"]
                    )
                handle_forward = self.cu.compile(name, input_tensors, **kwargs)
                tc_info["forward_name"], tc_info["handle_forward"] = name, handle_forward

                if backward:
                    tc_info["backward_name"] = backward_name
                self.cu.compilation_cache[hash_key] = tc_info

            if "outputs" in kwargs and kwargs["outputs"] is not None:
                out = kwargs["outputs"]
                tc_info["outputs"] = out
                if not isinstance(out, list):
                    tc_info["outputs"] = [out]
            out = TCFunction.apply(self.cu, tc_info, kwargs, *inputs)
            out = list(out) if (len(out) > 1) else out[0]
            return out
        except Exception as e:
            logger.error("Caught Exception: {}".format(e))
            return None


    def autotune(self, *inputs, **kwargs):
        r"""Evolution based algorithm for autotuning the defined TC language on
        given input tensor sizes

        Args:
            *inputs (required):
                Tuples or PyTorch Tensors / Variables that TC should
                tune kernel on. The inputs should be passed in the order they
                are also passed in the definition of TC language.

            cache (optional):

                * :code:`bool`: Set this to True if you want to save the autotuned options for later use (for example in running the kernel). If set to True, the cache file will look like :code:`/tmp/kernel_name_input_sizes_uuid`.

                * :code:`string`: Set this to the filepath where you want to save the options. Default is None.

                If a string is passed and :attr:`training=True`, then the options for backward kernel will be saved
                to :code:`filepath -> cache_file + '_backward'` i.e. prefix :code:`_backward` will be appended.

            options (optional):
                Kernel mapping options of type :code:`Options`. These options
                provide mapping for kernel like grid, blocks, memory etc. It
                is recommended to always pass kernel options. The options can be
                set by:

                * You can create :code:`Options` object by chosing the closely matching "type" of kernel. For example:

                .. code::

                    import tensor_comprehensions as tc
                    options = tc.Options(type)

                where :attr:`type` is a string with value one of below:

                * :attr:`pointwise`:  if kernel resembles a pointwise operation

                * :attr:`mlp`: if kernel resembles an Linear layer operation

                * :attr:`conv`: if kernel resembles a convolution operation

                * :attr:`group_conv`: if kernel resembles a group convolution operation

                * :attr:`naive`: if none of the above, then chose naive *Default*

                If no :attr:`Options` are passed, the naive options will be used which
                might not yield great performance.

            reorder_function:
                If :attr:`training` is set to true in `define` call,
                then TC infers the inputs for backward layer for compilation
                (1st time the layer is run) and tuning. The backward layer
                should typically contain the grad_outputs of the forward layer.
                The backward layer should take (TC forward inputs + grad_outputs)
                in the same order as the forward TC takes inputs and emits outputs.
                If the order of the outputs is changed, or some output grad are
                not required in backwards, then you can pass a function which
                can reorder/drop the layer grad_outputs according to backwards
                layer inputs your TC needs. The function should return a :attr:`list`.

            generations (int):
                number of tuning generation to be run. Default 25

            pop_size (int):
                number of candidates in each generation. Default 100

            crossover_rate (int):
                rate at which new candidates are bred instead of just surviving across generations. Default 80

            mutation_rate (int):
                rate at which candidate options are randomly changed (mutated). Default 7

            number_elites (int):
                number of best candidates that are preserved intact between generations (without any mutations). Default 10

            threads (int):
                The number of threads that are used to compile different candidates in parallel. Default 1

            gpus (string):
                A comma separated list of GPUs (ids) to use for evaluating candidates (e.g., “0,1,2,3”). Default "0"

            tuner_min_launch_total_threads (int):
                Prune out kernels mapped to fewer than this many threads and block. Set this to 1 to avoid pruning. Default 64

        Returns:
            Object of type :attr:`Options` that can be directly used to run the kernel.
            If :attr:`training` = True, then the list of size two containing
            forward kernel options and backward options will be returned.

        Example:
            >>> LANG = MATMUL_LANG
            >>> matmul = tc.define(lang, name="matmul")
            >>> mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
            >>> options = matmul.autotune(mat1, mat2, cache=True, options=Options("mlp"))

        """
        validate_autotuner_input(*inputs)
        kwargs.update(self.kwargs_define)
        if self.tuner is None:
            self.tuner = TcAutotuner(self.lang, **kwargs)
        else:
            # we do the init again so that the autotuner parameters are updated
            # properly if users change them
            self.tuner.set_autotuner_parameters(**kwargs)
        return self.tuner.autotune(*inputs, **kwargs)

###############################################################################
# User Facing TC call
###############################################################################
def define(lang, **kwargs_define):
    r"""Process and store TC definitions from input TC language where language
    can have many TC definitions. Most common example for multiple TC definitions
    is forward and backward TC for a layer.

    Args:
        lang (string, required):
            a valid TC language defining the operations using
            Einstein notation. It can have multiple TC definitions in the same lang.

        name (string, required):
            A string same as the name of your TC.

        training (bool):
            boolean value describing whether the :attr:`lang` containes two
            TC definitions describing forward and backward operation. If set to
            True, TC will enable the train mode for the lang. If set to False,
            TC will not enable the train mode.

        backward (string, optional):
            A string same as the name of backwards TC if the :attr:`training` is
            set to True. The backward TC name must be specified if training is True
            and the :attr:`lang` should contain both forward and backward TC
            strings.

        constants (dict, optional):
            if your TC uses scalars, for example strides in convolutions,
            you should format the string with the scalar values. For that,
            pass the python dictionary containing scalar name and its value.

        inject_kernel (string, optional):
            If you want to manually inject an external CUDA code
            for a TC definition,  set :attr:`inject_kernel` to the name
            of your kernel you want to inject.

        cuda_code (string, optional):
            If you want to manually inject an external CUDA code for a TC definition,
            then set :attr:`cuda_code` to the CUDA code string you want to inject.

    Returns:
        TC layer that you can run by passing the tensors. If :attr:`training` is True,
        the layer returned will also do the backwards when backwards is called.

    Example:
        >>> LANG = MATMUL_LANG
        >>> matmul = tc.define(lang, name="matmul")
        >>> mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        >>> out = matmul(mat1, mat2)
    """

    if "constants" in kwargs_define and kwargs_define["constants"]:
        # there are some scalars in the lang, replace them with constants
        lang = lang.format(**kwargs_define["constants"])
    tc_unit = TcUnit(lang, **kwargs_define)
    return tc_unit

###############################################################################
# Quick utility to decode the options
###############################################################################
def decode(filepath):
    r"""Decodes the .options file produced by running autotuning on kernel.

    Args:
        filepath (string): file which contains the options. This file should
            have extension '.options'

    Returns:
        A file with path filepath + '.decoded' which contains the decoded options.
    """
    assert os.path.exists(filepath), "The filepath specific doesn't exist."
    cwd = os.path.realpath(os.path.dirname(__file__))
    cmd = 'protoc --decode tc.OptionsCacheProto {cwd}/compilation_cache.proto -I {cwd} < {filepath} >> {filepath}.decoded'
    cmd = cmd.format(cwd=cwd, filepath=filepath)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    stdout = process.communicate()
