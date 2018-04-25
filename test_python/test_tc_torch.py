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

import os, unittest, time

import torch
import torch.cuda
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import tensor_comprehensions as tc
from tensor_comprehensions.mapping_options import Options
from common import TestCase, run_tests

tc.SetDebugFlags(dump_cuda=False)


MATMUL_LANG = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""

MATMUL_ABS_LANG = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
def abs(float(M, N) A) -> (O1) {
  O1(m, n) = fabs(A(m, n))
}
"""

CONV_TRAIN = """
def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {{
   O(n, m, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(m, c, kh, kw)
}}
def convolution_grad(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(N,M,H,W) O_grad) -> (I_grad, W1_grad) {{
   I_grad(n, c, h, w) +=! O_grad(n, m, {sh} * h - kh, {sw} * w - kw) * W1(m, c, kh, kw)
   W1_grad(m, c, kh, kw) +=! O_grad(n, m, {sh} * h - kh, {sw} * w - kw) * I(n, c, h, w)
}}
"""

PATH_PREFIX = os.path.join("/tmp/", "tc_test")

if not os.path.exists(PATH_PREFIX):
    os.makedirs(PATH_PREFIX)

###########################################################################
# TC tests without autotuning
###########################################################################
class TestTC(unittest.TestCase):
    def test_indexing(self):
        LANG = """
        def indexing(float(H, W) input, int32(L) index) -> (output) {{
            output(l, w) = input(index(l), w) where l in 0:{L}
        }}
        """
        indexing = tc.define(LANG, name="indexing", constants={"L":2})
        inp = torch.arange(0, 16).view(4, 4).cuda()
        idx = torch.IntTensor([1, 1]).cuda()
        out = indexing(inp, idx)

    def test_avgpool(self):
        # NOTE: take note of use of {{ }}
        LANG = """
        def avgpool(float(B, C, H, W) input) -> (output) {{
            output(b, c, h, w) +=! input(b, c, h * {sH} + kh, w * {sW} + kw) / ({kH} * {kW}) where kh in 0:{kH}, kw in 0:{kW}
        }}
        """
        avgpool = tc.define(LANG, name="avgpool", constants={"sH":1, "sW":1, "kH":2, "kW":2})
        inp = torch.ones(1, 1, 4, 4).cuda()
        out = avgpool(inp)

    def test_matmul(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2)

    def test_manual_options(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        options = Options("naive")
        out = matmul(mat1, mat2, options=options)

    def test_different_input_sizes(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out1 = matmul(mat1, mat2)

        # if the inputs sizes are different, re-compilation will happen
        mat3, mat4 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        out2 = matmul(mat3, mat4)

    def test_same_tc_reuse_outputs(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2)
        # reuse the same outputs now instad of allocating again, so we save
        # overhead of allocating storage again. Also, if the input sizes are same
        # we skip the compilation and run directly
        mat3, mat4 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        matmul(mat3, mat4, outputs=out)

    def test_multiple_tc(self):
        lang = MATMUL_ABS_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2)

        abs = tc.define(lang, name="abs")
        A = torch.randn(3, 4).cuda()
        out = abs(A)

    def test_matmul_variable(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = Variable(torch.randn(3, 4).cuda(), requires_grad=True), Variable(torch.randn(4, 5).cuda(), requires_grad=True)
        out = matmul(mat1, mat2)

    def test_matmul_variable_reuse_outputs(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = Variable(torch.randn(3, 4).cuda(), requires_grad=True), Variable(torch.randn(4, 5).cuda(), requires_grad=True)
        out = matmul(mat1, mat2)

        mat3, mat4 = Variable(torch.randn(3, 4).cuda(), requires_grad=True), Variable(torch.randn(4, 5).cuda(), requires_grad=True)
        matmul(mat3, mat4, outputs=out)

    def test_conv_backward(self):
        lang = CONV_TRAIN
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(lang, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
        I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
        W = Parameter(torch.randn(O, C, kH, kW).cuda())
        out = convolution(I, W)
        out.sum().backward()

    def test_conv_backward_pass_options(self):
        lang = CONV_TRAIN
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(lang, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
        I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
        W = Parameter(torch.randn(O, C, kH, kW).cuda())
        out = convolution(I, W, options=[tc.Options("conv"), tc.Options("group_conv")])
        out.sum().backward()

###########################################################################
# Autotuner tests
###########################################################################
class TestAutotuner(unittest.TestCase):

    def test_autotuner_no_cache_medium_size(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(72, 26).cuda(), torch.randn(26, 72).cuda()
        options = matmul.autotune(mat1, mat2, **tc.autotuner_settings)

    def test_autotuner_cachefile_first(self):
        cache_file = "{}/matmul_100_400_500".format(PATH_PREFIX)    # use argparse if input from command line
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        matmul.autotune(mat1, mat2, cache=cache_file, **tc.autotuner_settings)

    def test_autotuner_cachefile_load_automatic(self):
        lang = MATMUL_LANG
        cache_file = "{}/matmul_100_400_500".format(PATH_PREFIX)    # use argparse if input from command line
        assert os.path.isfile("{}.cuda".format(cache_file)), "looks like the cache_file doesn't exist"

        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        out1 = matmul(mat1, mat2, cache=cache_file)
        # the second time we run the kernel, we skip the compilation since it was
        # already compiled earlier
        out2 = matmul(mat1, mat2)

    def test_autotuner_no_cache_and_run_kernel_automatic(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        matmul.autotune(mat1, mat2, **tc.autotuner_settings)
        out = matmul(mat1, mat2)

    def test_autotuner_multiple_tc(self):
        lang = MATMUL_ABS_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        matmul.autotune(mat1, mat2, cache=True, **tc.autotuner_settings)
        out = matmul(mat1, mat2)

        absolute = tc.define(lang, name="abs")
        A = torch.randn(100, 400).cuda()
        absolute.autotune(A, cache=True, **tc.autotuner_settings)
        out = absolute(A)

    ##########################################################################
    # Training layer autotuning
    ##########################################################################
    def test_conv_train_autotune_no_cache_no_options(self):
        lang = CONV_TRAIN
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(lang, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
        I, W = torch.randn(N, C, H, W).cuda(), torch.randn(O, C, kH, kW).cuda()
        options = convolution.autotune(I, W, **tc.autotuner_settings)

    def test_conv_train_autotune_no_cache_no_options_seed(self):
        lang = CONV_TRAIN
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(lang, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
        I, W = torch.randn(N, C, H, W).cuda(), torch.randn(O, C, kH, kW).cuda()
        convolution.autotune(I, W, **tc.autotuner_settings)
        # on the second call, autotuning will be seeded from previous best options,
        # verify the seeding and new tuning settings being picked up
        convolution.autotune(I, W, generations=3, pop_size=5)

    def test_conv_train_autotune_cache_no_options_seed(self):
        lang = CONV_TRAIN
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(lang, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
        I, W = torch.randn(N, C, H, W).cuda(), torch.randn(O, C, kH, kW).cuda()
        convolution.autotune(I, W, cache=True, **tc.autotuner_settings)
        # on the second call, autotuning will be seeded from previous best options
        convolution.autotune(I, W, cache=True, **tc.autotuner_settings)

    def test_conv_train_autotune_cache_to_default(self):
        lang = CONV_TRAIN
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(lang, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
        I, W = torch.randn(N, C, H, W).cuda(), torch.randn(O, C, kH, kW).cuda()
        options = convolution.autotune(I, W, cache=True, **tc.autotuner_settings)

    def test_conv_train_autotune_to_cache_file_seed(self):
        lang = CONV_TRAIN
        cache_file = "{}/CONV_32_4_56_56_16_1_1_1_1".format(PATH_PREFIX)
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(lang, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
        I, W = torch.randn(N, C, H, W).cuda(), torch.randn(O, C, kH, kW).cuda()
        convolution.autotune(I, W, cache=cache_file, **tc.autotuner_settings)
        # the second call should be seeded from the previous call
        convolution.autotune(I, W, cache=cache_file, **tc.autotuner_settings)


if __name__ == '__main__':
    unittest.main()
