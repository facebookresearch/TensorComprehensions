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
import unittest, pdb


class TestTrainConvolutionReorder(unittest.TestCase):

    def test_train_convolution_reorder(self):
        LANG = """
        def convolution(float(N, C, H, W) I, float(M, C, KH, KW) W1, float(M) B) -> (tmp, O) {
          tmp(n, m, h, w) +=! I(n, c, h + kh, w + kw) * W1(m, c, kh, kw)
          O(n, m, h, w) = tmp(n, m, h, w) + B(m)
        }
        def convolution_grad(float(N, C, H, W) I, float(M, C, KH, KW) W1, float(M) B, float(N, M, H, W) d_O)
        -> (d_I, d_W1, d_B) {
          d_I(n, c, h, w) +=! d_O(n, m, h - kh, w - kw) * W1(m, c, kh, kw)
          d_W1(m, c, kh, kw) +=! d_O(n, m,  h - kh, w - kw) * I(n, c, h, w)
          d_B(m) +=! d_O(n, m, h, w)
        }
        """

        # since the forward layer produces two outputs, one is temporary which is
        # not needed in the forward pass, we can reorder the grad_outputs accordingly
        def reorder():
            def reorder_function(grad_outputs):
                return [grad_outputs[1]]
            return reorder_function

        N, C, H, W, M, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
        convolution = tc.define(LANG, training=True, name="convolution", backward="convolution_grad")
        I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
        W = Parameter(torch.randn(M, C, kH, kW).cuda())
        B = Parameter(torch.randn(M).cuda())
        out = convolution(I, W, B, reorder_function=reorder())
        out[0].sum().backward()


if __name__ == '__main__':
    unittest.main()
