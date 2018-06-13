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
import uuid
import unittest

import torch
import torch.cuda
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import tensor_comprehensions as tc

tuner_config = tc.TunerConfig(threads=8, generations=3, pop_size=8)

class TestTC(unittest.TestCase):
    #
    # Self explicit
    #
    def test_imports(self):
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

        print('\nImported tclib successfully')

    #
    # Construct a MappingOptions object programmatically from Python-land
    #
    def test_mapping_options(self):
        print('\nCreating mapping_options')
        options = (
            tc.MappingOptions('naive')
            .useSharedMemory(True)
            .unrollCopyShared(False)
            .mapToBlocks([256, 8])
            .mapToThreads([4, 16, 4])
            .tile([2, 8, 64, 128])
            .unroll(128)
            .fixParametersBeforeScheduling(False)
            .scheduleFusionStrategy("Max")
            .outerScheduleFusionStrategy("Preserve3Coincident"))
        print('Mapping options created successfully')

    #
    # Simple TC example with MappingOptions('naive') fallback
    #
    def test_tc(self):
        add = tc.define(
            "def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }",
            "add",
            tc.MappingOptions('naive')
        )
        A, B = torch.randn(100).cuda(), torch.randn(100).cuda()
        C, = add(A, B)
        tc.assert_almost_equal(C - torch.add(A, B), (A, B))

    #
    # Simple TC example without fallback but with tuning starting from
    # MappingOptions('naive')
    #
    def test_tc_autotune(self):
        add = tc.define(
            "def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }",
            "add"
        )
        A, B = torch.randn(10 ** 5).cuda(), torch.randn(10 ** 5).cuda()
        print("\nTune from naive options, don't store best options")
        options = add.tune(
            A,
            B,
            starting_options=tc.MappingOptions('naive'),
            tuner_config=tuner_config)
        C, = add(A, B)
        tc.assert_almost_equal(C - torch.add(A, B), (A, B))
        C.zero_()
        add(A, B, outputs = [C]) # inplace
        tc.assert_almost_equal(C - torch.add(A, B), (A, B))

    #
    # TC example without fallback but with tuning starting from MappingOptions('naive').
    # Then save to file and reinforce tuning starting from best options reloaded from file.
    #
    def test_tc_autotune_reinforce(self):
        add = tc.define(
            "def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }",
            "add"
        )
        A, B = torch.randn(10 ** 7).cuda(), torch.randn(10 ** 7).cuda()
        cache_filename = "/tmp/" + str(uuid.uuid4())

        print("\nLoad from, tune and store best options @{}".format(cache_filename))
        options = add.tune(
            A, B,
            starting_options=tc.MappingOptions('naive'),
            tuner_config=tuner_config,
            cache_filename=cache_filename,
            store_to_cache=True)
        C, = add(A, B)
        tc.assert_almost_equal(C - torch.add(A, B), (A, B))

        print("Load from {} and reinforce-tune best options, don't store".format(cache_filename))
        options = add.tune(
            A, B,
            tuner_config=tuner_config,
            cache_filename=cache_filename)
        C, = add(A, B)
        tc.assert_almost_equal(C - torch.add(A, B), (A, B))

    #
    # Simple TC test with fake templating by string substitution
    #
    def test_scalar(self):
        import re
        LANG="""
        def avgpool(float(B, C, H, W) input) -> (output) {
            output(b, c, h, w) +=! input(b, c, h * <sh> + kh, w * <sw> + kw) / (<kH> * <kW>) where kh in 0:<kH>, kw in 0:<kW>
        }
        """
        sH, sW, kH, kW = 1, 1, 2, 2
        LANG = re.sub('<sh>', str(sH), LANG)
        LANG = re.sub('<sw>', str(sW), LANG)
        LANG = re.sub('<kH>', str(kH), LANG)
        LANG = re.sub('<kW>', str(kW), LANG)
        avgpool = tc.define(LANG, "avgpool", tc.MappingOptions('naive'))
        inp = torch.ones(1, 1, 4, 4).cuda()
        out = avgpool(inp)
        # TODO: test results!!!

    #
    # Simple TC test operating on PyTorch variables
    #
    def test_pytorch_variable(self):
        add = tc.define(
            "def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }",
            "add",
            tc.MappingOptions('naive')
        )
        A, B = (
            Variable(torch.randn(100).cuda(), requires_grad=True),
            Variable(torch.randn(100).cuda(), requires_grad=True))
        out, = add(A, B)
        add(A, B, outputs=[out], unchecked=True)
        # TODO: test results!!!

    #
    # This test implements group normalization as a single TC kernel.
    # Performance is not expected to be as good as when using 2 kernels.
    #
    def test_group_norm_fused(self):
        group_normalization = """
            def group_normalization(
                float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta) -> (Sum, SumSq, O)
            {
                Sum(n, g) +=! I(n, g, r_d, r_h, r_w)
              SumSq(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)
                O(n, g, d, h, w) = gamma(g, d)
                    * ( I(n, g, d, h, w) - Sum(n, g) / (D * H * W))
                    * rsqrt( (SumSq(n, g) / (D * H * W) - Sum(n, g) * Sum(n, g)) + 1e-5 )
                    + beta(g, d)
            }
        """

        N, G, D, H, W = 32, 32, 4, 56, 56
        group_norm = tc.define(
            group_normalization,
            entry_point="group_normalization")
        I = Variable(torch.randn(N, G, D, H, W).cuda())
        gamma = Variable(torch.randn(G, D).cuda())
        beta = Variable(torch.randn(G, D).cuda())

        print("\nTune from naive options, don't store best options")
        options = group_norm.tune(
            I,
            gamma,
            beta,
            starting_options=tc.MappingOptions('naive'),
            tuner_config=tuner_config)
        print("\nRun best tuned options")
        Sum, SumSq, O = group_norm(I, gamma, beta)
        # TODO: test results!!!

    #
    # This test implements group normalization as 2 TC kernels using the
    # tc.define abstraction. In particular it is possible to insert reshapes
    # between TCs and have moments operate on a 2-D view which is mapped
    # more efficiently to CUDA at this time
    # (Note: loop collapsing modulo conforming strides is an easy TC-level
    #  transformation with nice expected benefits).
    #
    def test_group_norm_2kernels(self):
        group_normalization = """
            def moments(float(N, K) I) -> (mean, var) {
                # var = E(x^2) - mean^2.
                mean(n) +=! I(n, r_k)
                 var(n) +=! I(n, r_k) * I(n, r_k)
                mean(n)  = mean(n) / (K)
                 var(n)  =  var(n) / (K) - mean(n) * mean(n)
            }

            def group_normalization(
                float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta,
                float(N, G) mean, float(N, G) var) -> (O)
            {
                O(n, g, d, h, w) = gamma(g, d)
                    * ( I(n, g, d, h, w) - mean(n, g) )
                    * rsqrt( var(n, g) + 1e-5 )
                    + beta(g, d)
            }
        """

        N, G, D, H, W = 32, 32, 4, 56, 56
        moments = tc.define(
            group_normalization,
            entry_point="moments")
        group_norm = tc.define(
            group_normalization,
            entry_point="group_normalization")
        I = Variable(torch.randn(N, G, D, H, W).cuda())
        gamma = Variable(torch.randn(G, D).cuda())
        beta = Variable(torch.randn(G, D).cuda())

        print("\nTune moments from naive options, don't store best options")
        moments.tune(
            I.view((N * G, -1)),
            starting_options=tc.MappingOptions('naive'),
            tuner_config=tuner_config)
        print("\nRun tuned moments with best options")
        mean, var = moments(I.view((N * G, -1)))

        print("\nTune group_norm from naive options, don't store best options")
        group_norm.tune(
            I,
            gamma,
            beta,
            mean.view((N, G)),
            var.view((N, G)),
            starting_options=tc.MappingOptions('naive'),
            tuner_config=tuner_config)
        print("\nRun tuned group_norm with best options")
        out, = group_norm(I, gamma, beta, mean.view((N, G)), var.view((N, G)))
        # TODO: test results!!!

    #
    # This test implements group normalization as 2 TC kernels using the
    # tc.define_with_autograd abstraction. In particular it is ** NOT **
    # possible to insert reshapes between TCs moments operate on a 5-D tensor
    # (Note: loop collapsing modulo conforming strides is an easy TC-level
    #  transformation with nice expected benefits).
    # The tc.define_with_autograd API is more bloated than hoped but it is
    # unclear we can do better, suggestions most welcome.
    # Maybe consider making cache_filename and tuner_config global parameters.
    #
    def test_group_norm_2kernels_autograd(self):
        group_normalization = """
            def moments(float(N, G, D, H, W) I) -> (mean, var) {
                # var = E(x^2) - mean^2.
                mean(n, g) +=! I(n, g, r_d, r_h, r_w)
                 var(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)
                mean(n, g)  = mean(n, g) / (D * H * W)
                 var(n, g)  =  var(n, g) / (D * H * W) - mean(n, g) * mean(n, g)
            }

            def group_normalization(
                float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta,
                float(N, G) mean, float(N, G) var) -> (O)
            {
                O(n, g, d, h, w) = gamma(g, d)
                    * ( I(n, g, d, h, w) - mean(n, g) )
                    * rsqrt( var(n, g) + 1e-5 )
                    + beta(g, d)
            }
        """

        N, G, D, H, W = 32, 32, 4, 56, 56
        cache_filename = "/tmp/" + str(uuid.uuid4())
        group_norm = tc.define_with_autograd(
            group_normalization,
            forward_entry_points=("moments", "group_normalization"),
            backward_entry_points=(),
            # First: I, gamma, beta -> I
            #        => (0, )
            # Second: I, gamma, beta, mean, var -> I, gamma, beta, mean, var
            #        => (0, 1, 2, 3, 4) => None
            forward_input_indices=((0, ), ),
            cache_filename=cache_filename,
            tuner_config=tuner_config)
        I = Variable(torch.randn(N, G, D, H, W).cuda())
        gamma = Variable(torch.randn(G, D).cuda())
        beta = Variable(torch.randn(G, D).cuda())
        # First occurrence on a new cache filename triggers tuning
        mean, var, out = group_norm(I, gamma, beta)
        # Subsequent occurrences do not
        mean, var, out = group_norm(I, gamma, beta)
        # TODO: test results!!!

    #
    # This test implements single kernel forward and backward with
    # tc.define_with_autograd. This is the "light touch" mode and the function call
    # may already be too verbose.
    # Maybe consider making cache_filename and tuner_config global parameters.
    #
    def test_conv_backward_fused(self):
        conv = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
            O(n, m, h, w) +=!
                I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
        }
        def convolution_grad(
            float(N,C,H,W) I, float(M,C,KH,KW) W1, float(N,M,H,W) d_O)
            -> (d_I, d_W1)
        {
            d_I(n, c, h, w) +=!
                d_O(  n, r_m, h - r_kh, w - r_kw) * W1(r_m, c, r_kh, r_kw)
            d_W1(m, c, kh, kw) +=!
                d_O(r_n,   m, r_h - kh, r_w - kw) *  I(r_n, c,  r_h,  r_w)
        }
        """

        N, C, H, W, O, kH, kW = 32, 4, 56, 56, 16, 1, 1
        cache_filename = "/tmp/" + str(uuid.uuid4())
        convolution = tc.define_with_autograd(
            conv,
            forward_entry_points=("convolution", ),
            backward_entry_points=("convolution_grad", ),
            cache_filename=cache_filename,
            tuner_config=tuner_config)
        I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
        W = Parameter(torch.randn(O, C, kH, kW).cuda())
        # First occurrence on a new cache filename triggers tuning
        out, = convolution(I, W)
        out.sum().backward()
        # Subsequent occurrences do not
        out, = convolution(I, W)
        out.sum().backward()
        # TODO: test results!!!


    #
    # This test implements single kernel forward and 2 kernel backward with
    # tc.define_with_autograd. The performance of the backward is expected to be
    # significantly better because the loops types in the single kernel may not
    # be currently fused profitably (need to investigate the fused case deeper).
    # The API is more bloated than hoped but it is unclear we can do better,
    # suggestions most welcome.
    # Maybe consider making cache_filename and tuner_config global parameters.
    #
    def test_conv_backward_2kernels(self):
        conv = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
            O(n, m, h, w) +=!
                I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
        }
        def convolution_igrad(float(M,C,KH,KW) W1, float(N,M,H,W) d_O)
            -> (d_I)
        {
            d_I(n, c, h, w) +=!
                d_O(  n, r_m, h - r_kh, w - r_kw) * W1(r_m, c, r_kh, r_kw)
        }
        def convolution_wgrad(float(N,C,H,W) I, float(N,M,H,W) d_O) -> (d_W1)
        {
            d_W1(m, c, kh, kw) +=!
                d_O(r_n,   m, r_h - kh, r_w - kw) *  I(r_n, c,  r_h,  r_w)
        }
        """

        N, C, H, W, O, kH, kW = 32, 4, 56, 56, 16, 1, 1
        cache_filename = "/tmp/" + str(uuid.uuid4())
        convolution = tc.define_with_autograd(
            conv,
            forward_entry_points=("convolution", ),
            backward_entry_points=("convolution_igrad", "convolution_wgrad"),
            cache_filename=cache_filename,
            # First: (I, W1, d_O) -> (W1, d_O) => (1, 2)
            # Second: (I, W1, d_O, d_I) -> (I, d_O) => (0, 2)
            backward_input_indices=((1, 2), (0, 2), ),
            tuner_config=tuner_config)
        I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
        W = Parameter(torch.randn(O, C, kH, kW).cuda())
        # First occurrence on a new cache filename triggers tuning
        out, = convolution(I, W)
        out.sum().backward()
        # Subsequent occurrences do not
        out, = convolution(I, W)
        out.sum().backward()
        # TODO: test results!!!

    #
    # This test makes direct use of the pybinds abstraction which is close
    # to C++ land
    #
    def test_matmul_low_level_api(self):
        lang = """
        def matmul(float(M,N) A, float(N,K) B) -> (C) {
            C(m, k) +=! A(m, r_n) * B(r_n, k)
        }
        """
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()

        # Making use of the lower level pybind API
        executor = tc.compile(
            lang, "matmul", (mat1, mat2), tc.MappingOptions('naive'))
        output, = executor.run((mat1, mat2), ())
        torch.cuda.synchronize()
        expected = torch.mm(mat1, mat2)
        torch.cuda.synchronize()
        diff = output - expected
        tc.assert_almost_equal(diff, (mat1, mat2), 4)

        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        output, = executor.run((mat1, mat2), (output, ))
        tc.assert_almost_equal(output - torch.mm(mat1, mat2), (mat1, mat2), 4)

    #
    # This test makes direct use of the pybinds abstraction which is close
    # to C++ land
    #
    def test_tensordot_autotune_pybind(self):
        lang = """
        def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
          O(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
        }
        """
        entry_point = "tensordot"

        N, C1, C2, C3, H, W = 40, 16, 8, 20, 13, 15
        cache_filename = "/tmp/" + str(uuid.uuid4())
        I0 = torch.randn(N, C1, C2, H, W).cuda()
        I1 = torch.randn(N, C2, C3, H, W).cuda()
        inputs = [I0, I1]

        print("\n====> Autotuning kernel and saving results")
        tuner = tc.Tuner(lang, cache_filename)
        top1 = tuner.tune(
            entry_point,
            tuple(inputs),
            tc.MappingOptions('naive'),
            tuner_config)

        print("\n====> Running the kernel with autotuned options")
        executor = tc.compile(lang, entry_point, tuple(inputs), top1)
        output, = executor.run(tuple(inputs), ())

        print("\n====> Loading the autotuned options")
        cache = tc.MappingOptionsCache(cache_filename)
        best_options, = cache.load(lang, entry_point, tuple(inputs), 10)
        assert top1.__str__() == best_options.__str__(), (
            "Expected the same but found {}\nand\n{}".format(top1, best_options))

        print("\n====> Running the kernel with loaded options")
        executor = tc.compile(lang, entry_point, tuple(inputs), best_options)
        output, = executor.run(tuple(inputs), ())
        # TODO: test results!!!

    #
    # This test makes direct use of MultiTcBuilder anc MultiTcFunction that are used
    # to implement tc.define_with_autograd
    #
    def test_low_level_python_tc_function(self):
        from tensor_comprehensions.detail import MultiTcBuilder, MultiTcFunction
        mm = """
        def matmul(float(M,N) A, float(N,K) B) -> (C) {
            C(m, k) +=! A(m, r_n) * B(r_n, k)
        }
        def matmul_grad(float(M,N) A, float(N,K) B, float(M,K) d_O) -> (d_A, d_B) {
            d_A(m, n) +=! d_O(  m, r_k) * B(  n, r_k)
            d_B(n, k) +=! d_O(r_m,   k) * A(r_m,   n)
        }
        """
        tcb = MultiTcBuilder(
            tc=mm,
            # Note the extra commas inside the tuple arguments.
            # They force python to pass arguments by tuples instead of single value
            forward_entry_points=("matmul", ),
            backward_entry_points=("matmul_grad", ),
            tuner_cache_file="/tmp/some_cache_file_we_reuse_for_perf_reinforcement",
            tuner_config=tuner_config,
        )
        mat1, mat2 = torch.randn(300, 400).cuda(), torch.randn(400, 500).cuda()
        MultiTcFunction.apply(tcb, mat1, mat2)
        mat1.requires_grad = True
        mat2.requires_grad = True
        output, = MultiTcFunction.apply(tcb, mat1, mat2)
        v = output.sum()
        v.backward(retain_graph = True)
        # TODO: test results!!!


if __name__ == '__main__':
    unittest.main()
