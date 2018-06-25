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
import unittest

import tempfile

import torch
import torch.cuda

import tensor_comprehensions as tc

tc.SILENT = True
tuner_config = tc.TunerConfig().threads(5).generations(3).pop_size(5)

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

    #
    # Construct a MappingOptions object programmatically from Python-land
    #
    def test_mapping_options(self):
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

    #
    # Simple TC example with explicit 'naive' compilation
    #
    def test_tc(self):
        A, B = torch.randn(100, device='cuda'), torch.randn(100, device='cuda')
        add = tc.compile(
            "def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }",
            "add",
            'naive',
            A, B,
        )
        C = add(A, B)
        tc.assert_almost_equal(C, torch.add(A, B), A, B)

    #
    # Simple TC example without fallback but with tuning starting from
    # MappingOptions('naive')
    #
    def test_tc_autotune(self):
        A, B = (
            torch.randn(10 ** 5, device='cuda'),
            torch.randn(10 ** 5, device='cuda'))
        add = tc.autotune_and_compile(
            "def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }",
            "add",
            A, B,
            starting_options='naive',
            tuner_config=tuner_config,
        )
        C = add(A, B)
        tc.assert_almost_equal(C, torch.add(A, B), A, B)
        C.zero_()
        add(A, B, outputs=(C,)) # inplace
        tc.assert_almost_equal(C, torch.add(A, B), A, B)

    #
    # Simple test of the TC class in debug_mode
    #
    def test_define(self):
        T = tc.define(
            """
            def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }
            def sub(float(N) A, float(N) B) -> (C) { C(i) = A(i) - B(i) }
            """,
            tc.make_naive_options_factory())
        A, B = torch.randn(100, device='cuda'), torch.randn(100, device='cuda')
        C = T.add(A, B)
        tc.assert_almost_equal(C, torch.add(A, B), A, B)
        D = T.sub(A, B)
        tc.assert_almost_equal(D, (A - B), A, B)

    #
    # Simple TC test with fake templating by string substitution
    #
    def test_scalar(self):
        import re
        tc_str="""
        def avgpool(float(B, C, H, W) input) -> (output) {
            output(b, c, h, w) +=!
                input(b, c, h * <sh> + kh, w * <sw> + kw) / (<kH> * <kW>)
                where kh in 0:<kH>, kw in 0:<kW>
        }
        """
        sH, sW, kH, kW = 1, 1, 2, 2
        tc_str = re.sub('<sh>', str(sH), tc_str)
        tc_str = re.sub('<sw>', str(sW), tc_str)
        tc_str = re.sub('<kH>', str(kH), tc_str)
        tc_str = re.sub('<kW>', str(kW), tc_str)
        T = tc.define(tc_str, tc.make_naive_options_factory())
        inp = torch.ones(1, 1, 4, 4, device='cuda')
        out = T.avgpool(inp)

        from torch.nn.modules.pooling import AvgPool2d
        ref = AvgPool2d(2, stride=1).forward(inp)
        tc.assert_almost_equal(ref, out, inp)

    #
    # This test implements group normalization as a single TC kernel.
    # Performance is not expected to be as good as when using 2 kernels.
    #
    def test_group_norm_fused(self):
        group_normalization = """
            def group_normalization(
                float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta)
            -> (Sum, SumSq, O)
            {
                Sum(n, g) +=! I(n, g, r_d, r_h, r_w)
              SumSq(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)
                O(n, g, d, h, w) =  gamma(g, d)
                    * ( I(n, g, d, h, w) - Sum(n, g) / (D * H * W))
                    * rsqrt( (SumSq(n, g) - Sum(n, g) * Sum(n, g) / (D * H * W))
                           / (D * H * W)
                           + 1e-5)
                    + beta(g, d)
            }
        """

        N, G, D, H, W = 32, 32, 4, 56, 56
        T = tc.define(
            group_normalization,
            tc.make_autotuned_options_factory(
                starting_options='naive',
                tuner_config=tuner_config))
        I, gamma, beta = (
            torch.randn(N, G, D, H, W, device='cuda'),
            torch.randn(G, D, device='cuda').fill_(1.0),
            torch.randn(G, D, device='cuda').zero_())
        Sum, SumSq, O = T.group_normalization(I, gamma, beta)

        from torch.nn.modules.normalization import GroupNorm
        GN = GroupNorm(G, G*D).cuda()
        ref = GN.forward(I.view((N, G*D, H, W)))

        tc.assert_almost_equal(ref, O.view((N, G*D, H, W)), I, operations=D*H*W)

    #
    # This test implements group normalization as 2 TC kernels
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
        I, gamma, beta = (
            torch.randn(N, G, D, H, W, device='cuda'),
            torch.randn(G, D, device='cuda').fill_(1.0),
            torch.randn(G, D, device='cuda').zero_())

        T = tc.define(
            group_normalization,
            tc.make_autotuned_options_factory(
                starting_options='naive',
                tuner_config=tuner_config))

        # First occurrence triggers tuning
        mean, var = T.moments(I.view((N * G, -1)))
        out = T.group_normalization(
            I, gamma, beta, mean.view((N, G)), var.view((N, G)))

        mean, var = T.moments(I.view((N * G, -1)))
        out = T.group_normalization(
            I, gamma, beta, mean.view((N, G)), var.view((N, G)))

        from torch.nn.modules.normalization import GroupNorm
        GN = GroupNorm(G, G*D).cuda()
        ref = GN.forward(I.view((N, G*D, H, W)))

        tc.assert_almost_equal(ref, out.view((N, G*D, H, W)), I, operations=D*H*W)

    #
    # TC example without fallback but with tuning starting from MappingOptions('naive').
    # Then save to file and reinforce tuning starting from best options reloaded from file.
    #
    def test_tc_autotune_reinforce(self):
        with tempfile.NamedTemporaryFile() as cache_file:
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
            I, gamma, beta = (
                torch.randn(N, G, D, H, W, device='cuda'),
                torch.randn(G, D, device='cuda').fill_(1.0),
                torch.randn(G, D, device='cuda').zero_())

            T = tc.define(
                group_normalization,
                tc.make_autotuned_options_factory(
                    starting_options='naive',
                    tuner_config=tuner_config,
                    cache_filename=cache_file.name,
                    store_to_cache=True))
            # First occurrence triggers tuning
            mean, var = T.moments(I.view((N * G, -1)))
            out = T.group_normalization(
                I, gamma, beta, mean.view((N, G)), var.view((N, G)))

            # Create a new TC object to retrigger tuning
            T = tc.define(
                group_normalization,
                tc.make_autotuned_options_factory(
                    tuner_config=tuner_config,
                    cache_filename=cache_file.name,
                    load_from_cache=True,
                    store_to_cache=True))
            mean, var = T.moments(I.view((N * G, -1)))
            out = T.group_normalization(
                I, gamma, beta, mean.view((N, G)), var.view((N, G)))

            from torch.nn.modules.normalization import GroupNorm
            GN = GroupNorm(G, G*D).cuda()
            ref = GN.forward(I.view((N, G*D, H, W)))

            tc.assert_almost_equal(ref, out.view((N, G*D, H, W)), I, operations=D*H*W)


    #
    # This tests single kernel forward/backward with tc.make_autograd.
    #
    def test_conv_with_backward_fused(self):
        conv = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(M) Bias)
        -> (O)
        {
            O(n, m, h, w) +=!
                I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
            O(n, m, h, w)  = O(n, m, h, w) + Bias(m)
        }
        def convolution_grad(
            float(N,C,H,W) I, float(M,C,KH,KW) W1, float(M) Bias, float(N,M,H,W) d_O)
            -> (d_I, d_W1, d_Bias)
        {
            d_I(n, c, h, w) +=!
                d_O(  n, r_m, h - r_kh, w - r_kw) * W1(r_m, c, r_kh, r_kw)
            d_W1(m, c, kh, kw) +=!
                d_O(r_n,   m, r_h - kh, r_w - kw) *  I(r_n, c,  r_h,  r_w)
            # TODO: Bias incorrect + check
            d_Bias(m) = Bias(m)
        }
        """

        N, C, H, W, O, kH, kW = 32, 4, 56, 56, 16, 1, 1
        I = torch.randn(N, C, H, W, device='cuda', requires_grad=True)
        T = tc.define(
            conv,
            tc.make_autotuned_options_factory(
                starting_options='naive',
                tuner_config=tuner_config))
        convolution = tc.make_autograd(T.convolution, T.convolution_grad)

        # Reference
        from torch.nn.modules.conv import Conv2d
        Conv = Conv2d(C, O, 1, stride=1).cuda()
        ref = Conv.forward(I)

        W = Conv.weight.clone()
        Bias = Conv.bias.clone()

        # First occurrence triggers tuning (make_autotuned_options_factory)
        out = convolution(I, W, Bias)
        out.sum().backward()

        out = convolution(I, W, Bias)
        out.sum().backward()

        tc.assert_almost_equal(ref, out, I, operations=C * kH * kW)

    #
    # This tests 1-kernel forward/ 2-kernel backward with tc.make_autograd.
    # The performance of the backward is expected to be significantly better
    # because the loops types in the single kernel may not be currently fused
    # profitably (need to investigate the fused case deeper).
    #
    def test_conv_with_backward_2kernels(self):
        conv = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(M) Bias)
        -> (O)
        {
            O(n, m, h, w) +=!
                I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
            O(n, m, h, w)  = O(n, m, h, w) + Bias(m)
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
        def convolution_biasgrad(float(M) Bias) -> (d_Bias)
        {
            # TODO: Bias incorrect + check
            d_Bias(m) = Bias(m)
        }
        """

        N, C, H, W, O, kH, kW = 32, 4, 56, 56, 16, 1, 1
        T = tc.define(
            conv,
            tc.make_autotuned_options_factory(
                starting_options='naive',
                tuner_config=tuner_config))
        I = torch.randn(N, C, H, W, device='cuda', requires_grad=True)

        # Reference
        from torch.nn.modules.conv import Conv2d
        Conv = Conv2d(C, O, 1, stride=1).cuda()
        ref = Conv.forward(I)

        W = Conv.weight.clone()
        Bias = Conv.bias.clone()

        def convolution_backward(I, W, Bias, d_O):
            d_I = T.convolution_igrad(W, d_O)
            d_O = T.convolution_wgrad(I, d_O)
            d_Bias = T.convolution_biasgrad(Bias)
            return (d_I, d_O, d_Bias)

        convolution_function = tc.make_autograd(
            T.convolution, convolution_backward)

        # First occurrence triggers tuning
        out = convolution_function(I, W, Bias)
        out.sum().backward()

        # Subsequent occurrences do not
        out = convolution_function(I, W, Bias)
        out.sum().backward()

        tc.assert_almost_equal(ref, out, I, operations=C * kH * kW)

    #
    # This tests the direct use of pybinds which are closer to C++
    #
    def test_matmul_pybind(self):
        mm_str = """
        def matmul(float(M,N) A, float(N,K) B) -> (C) {
            C(m, k) +=! A(m, r_n) * B(r_n, k)
        }
        """

        A, B = (
            torch.randn(3, 4, device='cuda'),
            torch.randn(4, 5, device='cuda'))

        import tensor_comprehensions.tclib as tclib
        executor = tclib.compile(
            mm_str, "matmul", (A, B), tc.MappingOptions('naive'))
        C = executor.run((A, B), ())
        torch.cuda.synchronize()
        expected = torch.mm(A, B)
        torch.cuda.synchronize()
        tc.assert_almost_equal(C, expected, A, B, operations=4)

        C = executor.run((A, B), (C, ))
        tc.assert_almost_equal(C, torch.mm(A, B), A, B, operations=4)

    #
    # This tests the direct use of pybinds which are closer to C++
    #
    def test_tensordot_autotune_pybind(self):
        tensordot_str = """
        def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1)
            -> (O)
        {
            O(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
        }
        """
        entry_point = "tensordot"

        N, C1, C2, C3, H, W = 40, 16, 8, 20, 13, 15
        with tempfile.NamedTemporaryFile() as cache_file:
            I0 = torch.randn(N, C1, C2, H, W, device='cuda')
            I1 = torch.randn(N, C2, C3, H, W, device='cuda')

            tuner = tc.Tuner(tensordot_str, cache_file.name)
            top1 = tuner.tune(
                entry_point,
                (I0, I1),
                tc.MappingOptions('naive'),
                tuner_config)

            import tensor_comprehensions.tclib as tclib
            executor = tclib.compile(
                tensordot_str, entry_point, (I0, I1), top1)
            O = executor.run((I0, I1), ())

            cache = tc.MappingOptionsCache(cache_file.name)
            best_options, = cache.load(
                tensordot_str, entry_point, (I0, I1), 10)
            assert str(top1) == str(best_options), (
                "Expected the same but found {}\nand\n{}".format(
                    top1, best_options))

            executor = tclib.compile(
                tensordot_str, entry_point, (I0, I1), best_options)
            O = executor.run((I0, I1), ())

            # No simple torch baseline, compare against naive
            executor = tclib.compile(
                tensordot_str, entry_point, (I0, I1), tc.MappingOptions('naive'))
            ref = executor.run((I0, I1), ())

            tc.assert_almost_equal(ref, O, I0, I1, operations=C2)

if __name__ == '__main__':
    unittest.main()
