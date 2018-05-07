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

import tensor_comprehensions as tc

import torch
import torch.cuda
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import unittest


class TestTrainConvolutionStrided(unittest.TestCase):

    def test_train_convolution_strided(self):
        # NOTE: take note of use of {{ }} below for handling TC with scalars
        LANG = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {{
            O(n, m, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(m, c, kh, kw)
        }}
        def convolution_grad(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(N,M,H,W) d_O)
        -> (d_I, d_W1)
        {{
            d_I(n, c, h, w) +=! d_O(n, m, {sh} * h - kh, {sw} * w - kw) * W1(m, c, kh, kw)
            d_W1(m, c, kh, kw) +=! d_O(n, m, {sh} * h - kh, {sw} * w - kw) * I(n, c, h, w)
        }}
        """

        # NOTE: TC doesn't support padding yet
        # see https://github.com/facebookresearch/TensorComprehensions/issues/11
        # due to this reason, we use kernel=1 for now (only because we want to)
        # do the backwards as well. If kernel != 1 then we will have inconsistent
        # values of H, W in the backward TC
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(LANG, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
        I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
        W = Parameter(torch.randn(O, C, kH, kW).cuda())
        out = convolution(I, W)
        out[0].sum().backward()


if __name__ == '__main__':
    unittest.main()
