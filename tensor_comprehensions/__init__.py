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
from typing import Callable, Iterable, List, Optional, Tuple, Union

import time

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
from tensor_comprehensions.tclib import dump_ptx
from tensor_comprehensions.tclib import cuda_compiler
from tensor_comprehensions.tclib import llvm_flags
from tensor_comprehensions.tclib import nvcc_flags

from tensor_comprehensions.tclib import CompilationCache
from tensor_comprehensions.tclib import MappingOptions
from tensor_comprehensions.tclib import MappingOptionsCache
from tensor_comprehensions.tclib import TcExecutor
from tensor_comprehensions.tclib import Tuner
from tensor_comprehensions.tclib import TunerConfig

import tensor_comprehensions.tclib as tclib

SILENT = False

def assert_almost_equal(actual : torch.Tensor,
                        expected : torch.Tensor,
                        *inputs : torch.Tensor,
                        operations: Optional[int] = 1,
                        precision: Optional[float] = 1e-7):
    r"""Asserts numerical precision requirements.

        :param actual: the PyTorch Tensor to check.
        :param expected: the expected PyTorch Tensor.
        :param inputs: PyTorch Tensors passed as inputs to the TC that produced the
            actual Tensor.
        :param operations: maximum number of iterated operations per produced value.
            This is used to compute the required absolute precision.
        :param precision: relative precision at which to check.
    """
    diff = actual - expected
    max_value = 0.0
    for inp in inputs:
        max_value = max(float(inp.abs().max()), max_value)
    max_diff = float(diff.abs().max())
    assert max_diff <= operations * precision * max_value, (
        ("error at relative precision: {}, #operations: {}, " +
         "max_value: {}, max_diff: {}").format(
            precision, operations, max_value, max_diff)
    )

class Executor(object):
    r"""Callable helper class to hold the result of compiling a TC def with fixed input sizes.
    """

    def __init__(self, executor: TcExecutor):
        self.executor = executor

    def __call__(
            self,
            *inputs: torch.Tensor,
            outputs: Optional[Tuple[torch.Tensor]] = None,
            unchecked: Optional[bool] = False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        r"""Run the compiled TC kernel.

        :param inputs: PyTorch Tensors for which the compiled kernel has been
            specialized. You must use tensors of the same sizes as you have
            specialized for otherwise illegal memory accesses will occur.
        :param outputs: PyTorch Tensors into which the TC kernel will write. If
            left unspecified, new tensors will be allocated (which will have a
            noticeable performance impact until the caching allocator kicks in).
        :param unchecked: Disable shape checks (at your own risk) which reduces
            overhead in the case of low-latency kernels.

        Returns:
            A PyTorch Tensor, or a tuple of Pytorch Tensors in the case of
            multiple return values.

        Example:
            >>> A, B = (
            ...     torch.randn(100, device='cuda').fill_(1),
            ...     torch.randn(100, device='cuda').fill_(1))
            ... add = tc.compile(
            ...     'def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }',
            ...     'add',
            ...     'naive',
            ...     A, B,
            ... )
            ... C = add(A, B)
            >>> print(C.min(), C.max())
            tensor(2., device='cuda:0') tensor(2., device='cuda:0')
        """
        if outputs is None:
            if unchecked:
                return self.executor.unchecked_run(inputs)
            return self.executor.run(inputs)

        if unchecked:
            return self.executor.unchecked_run(inputs, outputs)
        return self.executor.run(inputs, outputs)

def compile(
        tc: str,
        entry_point: str,
        mapping_options: Union[str, MappingOptions],
        *inputs: torch.Tensor) -> Executor:
    r"""Returns a compiled, callable, low-overhead :class:`Executor`.

        An example of usage is provided in :class:`Executor`.

        :param tc: a string containing one of more TC defs.
        :param entry_point: the name of the TC def to compile and execute.
        :param mapping_options: the options to use for compilation.
        :param inputs: PyTorch Tensors for which the compiled kernel is specialized.

        :rtype: :class:`Executor`, a low-overhead callable class to launch the
            kernel compiled from the :code:`entry_point`.
    """
    mapping_options = (
        MappingOptions(mapping_options)
        if isinstance(mapping_options, str) else mapping_options)
    return Executor(tclib.compile(tc, entry_point, inputs, mapping_options))

def autotune(tc: str,
             entry_point: str,
             *inputs: torch.Tensor,
             starting_options: Optional[Union[str, MappingOptions]] = None,
             tuner_config: Optional[TunerConfig] = TunerConfig(),
             cache_filename: Optional[str] = None,
             load_from_cache: Optional[bool] = False,
             store_to_cache: Optional[bool] = False) -> MappingOptions:
    r"""Tunes the defined TC function for given inputs.

        The MappingOptions from which tuning starts is either passed explicitly via
        :code:`starting_options` or loaded from a cache file (when both
        :code:`cache_filename` and :code:`load_from_cache` are properly
        specified). Exactly one of :code:`starting_options` and
        :code:`load_from_cache` must be specified.

        It is possible to obtain a reinforcement tuning behavior by tuning over
        multiple executions and specifying both :code:`load_from_cache` and
        :code:`store_to_cache`. It is recommended to only use a single cache
        file for all TC defs and reinforce it over time.

        An example of usage is provided with :func:`autotune_and_compile`.

        :param tc: a string containing one of more TC defs.
        :param entry_point: the name of the TC def to compile and execute.
        :param inputs: PyTorch Tensors that TC should tune for. The inputs must be
            passed in the order they are also passed in the definition of
            the TC function.
        :param starting_options: :class:`~tclib.MappingOptions` from which tuning should start.
        :param tuner_config: :class:`~tclib.TunerConfig` to control the behavior of the autotuner.
        :param load_from_cache: Get the starting :class:`~tclib.MappingOptions` by loading from
            :code:`cache_filename`. If loading fails to recover an entry
            from the cache file for the given input sizes an assertion error
            will trigger.
        :param store_to_cache: Optionally store the best result by appending it to
            the backing cache file.

        Returns:
            The best options found during this tuning run.
    """

    if cache_filename is not None:
        assert load_from_cache or store_to_cache, ("cache_filename specified" +
            "must also specify load_from_cache or store_to_cache")
    if load_from_cache or store_to_cache:
        assert cache_filename is not None, ("load_from_cache or store_to_cache" +
        " specified, must also specify cache_filename")
    assert starting_options is not None or load_from_cache, (
        "Must specify either starting_options or load_from_cache, choose one!")
    assert starting_options is None or not load_from_cache, (
        "Cannot specify both starting_options and load_from_cache, choose one!")

    base_options = None
    if load_from_cache:
        cache = MappingOptionsCache(cache_filename)
        loaded = cache.load(tc, entry_point, inputs, 1)
        assert len(loaded) > 0, (
            "Could not load from cache for TC {} and sizes {}".format(
                entry_point,
                "".join(str(i.size()) + " " for i in inputs)))
        base_options = loaded[0]
    else:
        base_options = (
            MappingOptions(starting_options)
            if isinstance(starting_options, str) else starting_options)

    # TODO: This is still an implicit store behavior in the C++ API,
    #     make it explicit...
    tuner = Tuner(tc, cache_filename if store_to_cache else "")
    return tuner.tune(
        entry_point,
        inputs,
        base_options,
        tuner_config)

def autotune_and_compile(
        tc: str,
        entry_point: str,
        *inputs: torch.Tensor,
        starting_options: Optional[Union[str, MappingOptions]] = None,
        tuner_config: Optional[TunerConfig] = TunerConfig(),
        cache_filename: Optional[str] = None,
        load_from_cache: Optional[bool] = False,
        store_to_cache: Optional[bool] = False) -> Executor:
    r"""Calls autotune, compiles with best options then returns an Executor.

    Takes the same arguments as the :func:`autotune` function.

    Example:
        >>> A, B = (
        ... torch.randn(10 ** 5, device='cuda').fill_(1.0),
        ... torch.randn(10 ** 5, device='cuda').fill_(1.0))
        ... add = tc.autotune_and_compile(
        ...    "def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }",
        ...    "add",
        ...    A, B,
        ...    starting_options='naive',
        ...    tuner_config=tc.TunerConfig().threads(5).generations(3).pop_size(5)
        ... )
        ... C = add(A, B)
        >>> print(C.min(), C.max())
        tensor(2., device='cuda:0') tensor(2., device='cuda:0')
    """
    best = autotune(
        tc,
        entry_point,
        *inputs,
        starting_options=starting_options,
        tuner_config=tuner_config,
        cache_filename=cache_filename,
        load_from_cache=load_from_cache,
        store_to_cache=store_to_cache)
    if best is None:
        return None
    return compile(tc, entry_point, best, *inputs)

def make_naive_options_factory() -> (
        Callable[[str, str, Iterable[torch.Tensor]], MappingOptions]):
    r"""Return a factory that always generates naive :class:`~tclib.MappingOptions`.

        For easily getting started with TC and debugging purposes only.

        :rtype: a function that takes a string with multiple
            TC defs, an entry_point and input PyTorch Tensors and produces a
            :class:`~tclib.MappingOptions`.
    """
    def generate(tc: str,
                 entry_point: str,
                 *inputs: torch.Tensor) -> MappingOptions:
        return MappingOptions('naive')

    return generate

def make_load_from_cache_options_factory(cache_filename: str) -> (
        Callable[[str, str, Iterable[torch.Tensor]], MappingOptions]):
    r"""Return a factory that loads :class:`~tclib.MappingOptions` from a cache file.

        :param cache_filename: the filename
        :rtype: a function that takes a string with multiple
            TC defs, an entry_point and input PyTorch Tensors and produces a
            :class:`~tclib.MappingOptions`.
    """
    def generate(tc: str,
                 entry_point: str,
                 *inputs: torch.Tensor) -> MappingOptions:
        cache = MappingOptionsCache(cache_filename)
        loaded = cache.load(tc, entry_point, inputs, 1)
        if len(loaded) > 0:
            return loaded[0]
        return None

    return generate

def make_autotuned_options_factory(
        starting_options: Optional[Union[str, MappingOptions]] = None,
        tuner_config: TunerConfig = TunerConfig(),
        cache_filename: Optional[str] = None,
        load_from_cache: Optional[bool] = False,
        store_to_cache: Optional[bool] = False) -> (
            Callable[[str, str, Iterable[torch.Tensor]], MappingOptions]):
    r"""Return a factory that runs autotuning to determine the best :class:`~tclib.MappingOptions`.

        The returned factory just calls the :func:`autotune` function, see
        its documentation for more information.

        :rtype: a function that takes a string with multiple
            TC defs, an entry_point and input PyTorch Tensors and produces a
            :class:`~tclib.MappingOptions`.
    """
    def generate(tc: str,
                 entry_point: str,
                 *inputs: torch.Tensor) -> MappingOptions:
        return autotune(
            tc,
            entry_point,
            *inputs,
            starting_options=starting_options,
            tuner_config=tuner_config,
            cache_filename=cache_filename,
            load_from_cache=load_from_cache,
            store_to_cache=store_to_cache)

    return generate

class TC(object):
    def __init__(
            self,
            tc: str,
            mapping_options_factory: (
                Callable[[str, str, Iterable[torch.Tensor]], MappingOptions])
    ):
        self.tc = tc
        self.mapping_options_factory = mapping_options_factory
        self.compilation_cache = CompilationCache(self.tc)
        # Make each TC def in the tc str a method of the TC object so we can:
        #     T = tc.define("def add() ...")
        #     T.add()
        #
        def make_closure(obj: TC, tc_def_name: str):
            def fun(*inputs: torch.Tensor,
                    outputs: Optional[Tuple[torch.Tensor]] = None,
                    unchecked: Optional[bool] = False) -> List[torch.Tensor] :
                return obj(
                    tc_def_name, *inputs, outputs=outputs, unchecked=unchecked)

            return fun

        for tc_def in tclib.parse_defs(self.tc):
            self.__setattr__(tc_def, make_closure(self, tc_def))

    def __call__(
            self,
            entry_point: str,
            *inputs: torch.Tensor,
            outputs: Optional[Tuple[torch.Tensor]] = None,
            unchecked: Optional[bool] = False) -> List[torch.Tensor]:

        # Locally scoped implicit compilation
        def implicit_compile(tc_obj: TC,
                             entry_point: str,
                             *inputs: torch.Tensor):
            already_compiled = tc_obj.compilation_cache.is_compiled(
                entry_point, inputs)

            if already_compiled:
                return

            global SILENT
            if not SILENT:
                sizes = "".join(str(i.size()) + " " for i in inputs)
                print(
                    "TC \"{}\" was not explicitly compiled for ".format(entry_point) +
                    "inputs of sizes:\n  {}\n".format(sizes) +
                    "....Generate implicit MappingOptions")

            mapping_options = tc_obj.mapping_options_factory(
                tc_obj.tc, entry_point, *inputs)

            assert mapping_options is not None, (
                "No options found for TC {} ".format(entry_point) +
                "with inputs of sizes:\n  {}\n".format(
                    "".join(str(i.size()) + " " for i in inputs)))

            # Compile best options to set the executor for the current
            #     (entry point, inputs)
            start = time.clock()
            tc_obj.compilation_cache.compile(
                entry_point, inputs, mapping_options)
            if not SILENT:
                print(
                    "Done compiling TC \"{}\" (compile time: {}ms)".format(
                        entry_point, int((time.clock() - start) * 10 ** 3)))

        implicit_compile(self, entry_point, *inputs)

        if unchecked:
            return self.compilation_cache.unchecked_run(entry_point, inputs)

        return self.compilation_cache.run(entry_point, inputs)

def define(tc: str,
           mapping_options_factory: Callable[[str, str, Iterable[torch.Tensor]], MappingOptions]) -> TC:
    r"""Create a helper class with methods that implement multiple TC defs.

    Parsing a TC string with multiple defs and return a helper object with
    method names that match each of the TC def.
    Later, JIT compilation occurs on-demand the first time one such method is called
    with PyTorch Tensors of new sizes. The returned :class:`TC` helper class is
    backed by a compilation cache which memoizes the results of compilation and
    avoids spurious recompilations. In order to determine the
    :class:`~tclib.MappingOptions`, used for JIT compiling a particular TC def on
    inputs of particular sizes, the :code:`mapping_options_factory`
    function is called. We provide the factory builder functions
    :func:`make_naive_options_factory`,
    :func:`make_load_from_cache_options_factory` and
    :func:`make_autotuned_options_factory`

    Further user-defined factory functions can be easily written to extend
    the behavior.

    .. warning::

       If you chose to benchmark TC using this high-level API, be sure to
       understand how compilation, tuning and memoization interact. More
       generally, the low-level API should be used for benchmarking purposes.

    :param tc: a string containing one of more TC defs.
    :param mapping_options_factory: a function that takes a string with multiple
        TC defs, an entry_point and input PyTorch Tensors and produces a
        :class:`~tclib.MappingOptions`.
    :rtype: a Callable helper object with methods corresponding to the TC def
        names and backed by a compilation cache.

    Examples:
        One can define TC functions compiled with naive options for the
        purpose of correctness check debugging:

        >>> T = tc.define(
        ... '''
        ... def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }
        ... def sub(float(N) A, float(N) B) -> (C) { C(i) = A(i) - B(i) }
        ... ''',
        ... tc.make_naive_options_factory())
        ... A, B = torch.randn(100, device='cuda'), torch.randn(100, device='cuda')
        ... C = T.add(A, B)
        ... tc.assert_almost_equal(C, torch.add(A, B), A, B)
        ... D = T.sub(A, B)
        ... tc.assert_almost_equal(D, (A - B), A, B)

        One can also obtain a reinforced tuning behavior by:

        >>> tuner_config = tc.TunerConfig().threads(5).generations(3).pop_size(5)
        ... with tempfile.NamedTemporaryFile() as cache_file:
        ...     group_normalization = '''
        ...     def moments(float(N, K) I) -> (mean, var) {
        ...         # var = E(x^2) - mean^2.
        ...         mean(n) +=! I(n, r_k)
        ...          var(n) +=! I(n, r_k) * I(n, r_k)
        ...         mean(n)  = mean(n) / (K)
        ...          var(n)  =  var(n) / (K) - mean(n) * mean(n)
        ...     }
        ...
        ...     def group_normalization(
        ...         float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta,
        ...         float(N, G) mean, float(N, G) var) -> (O)
        ...     {
        ...         O(n, g, d, h, w) = gamma(g, d)
        ...             * ( I(n, g, d, h, w) - mean(n, g) )
        ...             * rsqrt( var(n, g) + 1e-5 )
        ...             + beta(g, d)
        ...     }
        ...     '''
        ...
        ...     N, G, D, H, W = 32, 32, 4, 56, 56
        ...     I, gamma, beta = (
        ...         torch.randn(N, G, D, H, W, device='cuda'),
        ...         torch.randn(G, D, device='cuda'),
        ...         torch.randn(G, D, device='cuda'))
        ...
        ...     T = tc.define(
        ...         group_normalization,
        ...         tc.make_autotuned_options_factory(
        ...             starting_options='naive',
        ...             tuner_config=tuner_config,
        ...             cache_filename=cache_file.name,
        ...             store_to_cache=True))
        ...     # First occurrence triggers tuning from naive options and
        ...     # stores to cache.
        ...     mean, var = T.moments(I.view((N * G, -1)))
        ...     out = T.group_normalization(
        ...         I, gamma, beta, mean.view((N, G)), var.view((N, G)))
        ...
        ...     # Create a new TC object to retrigger tuning, this time
        ...     # starting from MappingOptions loaded from cache.
        ...     T = tc.define(
        ...         group_normalization,
        ...         tc.make_autotuned_options_factory(
        ...             tuner_config=tuner_config,
        ...             cache_filename=cache_file.name,
        ...             load_from_cache=True,
        ...             store_to_cache=True))
        ...     mean, var = T.moments(I.view((N * G, -1)))
        ...     out = T.group_normalization(
        ...         I, gamma, beta, mean.view((N, G)), var.view((N, G)))
    """

    return TC(tc, mapping_options_factory)

class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forward_fun, backward_fun, *inputs):
        ctx.backward_fun = backward_fun
        ctx.save_for_backward(*inputs)
        return forward_fun(*inputs)

    @staticmethod
    def backward(ctx, *gradients):
        if ctx.backward_fun is not None:
            inputs = tuple(list(ctx.saved_tensors) + list(
                t.contiguous() for t in gradients))
            # PyTorch convention: need an extra None return for each of
            # forward_fun and backward_fun,
            return (None, None, *ctx.backward_fun(*inputs))

        return None

class Autograd(object):
    def __init__(self, forward_fun, backward_fun):
        self.forward_fun = forward_fun
        self.backward_fun = backward_fun

    def __call__(self, *inputs):
        return Function.apply(self.forward_fun, self.backward_fun, *inputs)

def make_autograd(forward_fun: Callable[[Iterable[torch.Tensor]], Iterable[torch.Tensor]],
                  backward_fun: Callable[[Iterable[torch.Tensor]], Iterable[torch.Tensor]]):
    r"""Create a Callable helper object with torch.autograd support.

    :param forward_fun: a function that takes PyTorch Tensors and implements the
        forward operation. Returns PyTorch Tensors.
    :param backward_fun: a function that takes PyTorch Tensors and implements the
        forward operation. Returns PyTorch Tensors.
    :rtype: a Callable helper object with torch.autograd support.

    .. warning::

       If you chose to benchmark TC using this high-level API, be sure to
       understand how autogr, compilation, tuning and memoization interact.
       More generally, the low-level API should be used for benchmarking
       purposes.

    Example:
        >>> conv = '''
        ... def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
        ...     O(n, m, h, w) +=!
        ...         I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
        ... }
        ... def convolution_igrad(float(M,C,KH,KW) W1, float(N,M,H,W) d_O)
        ...     -> (d_I)
        ... {
        ...     d_I(n, c, h, w) +=!
        ...         d_O(  n, r_m, h - r_kh, w - r_kw) * W1(r_m, c, r_kh, r_kw)
        ... }
        ... def convolution_wgrad(float(N,C,H,W) I, float(N,M,H,W) d_O) -> (d_W1)
        ... {
        ...     d_W1(m, c, kh, kw) +=!
        ...         d_O(r_n,   m, r_h - kh, r_w - kw) *  I(r_n, c,  r_h,  r_w)
        ... }
        ... '''
        ...
        ... N, C, H, W, O, kH, kW = 32, 4, 56, 56, 16, 1, 1
        ... T = tc.define(
        ...     conv,
        ...     tc.make_autotuned_options_factory(
        ...         starting_options='naive',
        ...         tuner_config=tuner_config))
        ... I, W = (
        ...     torch.randn(N, C, H, W, device='cuda', requires_grad=True),
        ...     torch.randn(O, C, kH, kW, device='cuda', requires_grad=True))
        ...
        ... def convolution_backward(I, W, d_O):
        ...     d_I = T.convolution_igrad(W, d_O)
        ...     d_O = T.convolution_wgrad(I, d_O)
        ...     return (d_I, d_O)
        ...
        ... convolution_function = tc.make_autograd(
        ...     T.convolution, convolution_backward)
        ...
        ... # First occurrence triggers tuning
        ... out = convolution_function(I, W)
        ... out.sum().backward()
        ...
        ... # Subsequent occurrences do not
        ... out = convolution_function(I, W)
        ... out.sum().backward()
    """
    return Autograd(forward_fun, backward_fun)

__all__ = [
    # Debugging functions, pass True to activate
    'logtostderr',
    'debug_lang',
    'debug_halide',
    'debug_tc_mapper',
    'debug_tuner',
    'dump_cuda',
    'dump_ptx',
    'cuda_compiler',
    'llvm_flags',
    'nvcc_flags',
    # Functions exposed by the tclib
    'compile',
    'autotune',
    'autotune_and_compile',
    # Classes exposed by the tclib
    'CompilationCache',
    'MappingOptions',
    'MappingOptionsCache',
    'Tuner',
    'TunerConfig',
    # Python-side functionality
    'assert_almost_equal',
    'define',
    'make_autograd',
    'make_naive_options_factory',
    'make_load_from_cache_options_factory',
    'make_autotuned_options_factory',
    'TC',
    'TCWithAutograd',
]
