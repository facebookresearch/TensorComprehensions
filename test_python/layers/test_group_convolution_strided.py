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
import unittest


class TestGroupConvolutionStrided(unittest.TestCase):

    def test_group_convolution_strided(self):
        LANG = """
        def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B) -> (O)
        {{
            O(n, g, f, h, w) +=! I(n, g, c, {sh} * h + kh, {sw} * w + kw) * W1(g, f, c, kh, kw)
            O(n, g, f, h, w) = O(n, g, f, h, w) + B(g, f)
        }}
        """
        N, G, C, H, W, F, KH, KW, sH, sW = 32, 32, 4, 56, 56, 4, 3, 3, 1, 1
        group_convolution = tc.define(LANG, name="group_convolution", constants={"sh":sH, "sw":sW})
        I, W1 = torch.randn(N, G, C, H, W).cuda(), torch.randn(G, F, C, KH, KW).cuda()
        B = torch.randn(G, F).cuda()
        out = group_convolution(I, W1, B)


if __name__ == '__main__':
    unittest.main()
