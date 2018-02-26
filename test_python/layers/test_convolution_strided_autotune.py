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


class TestAutotuneConvolutionStrided(unittest.TestCase):

    def test_autotune_convolution_strided(self):
        # NOTE: take note of use of {{ }} below for handling TC with scalars
        LANG = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(M) B) -> (O) {{
            O(n, m, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(m, c, kh, kw)
            O(n, m, h, w) = O(n, m, h, w) + B(m)
        }}
        """
        N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 3, 3, 1, 1
        convolution = tc.define(LANG, name="convolution", constants={"sh":sH, "sw":sW})
        I, W1, B = torch.randn(N, C, H, W).cuda(), torch.randn(O, C, kH, kW).cuda(), torch.randn(O).cuda()
        best_options = convolution.autotune(I, W1, B, **tc.autotuner_settings)
        out = convolution(I, W1, B, options=best_options)


if __name__ == '__main__':
    unittest.main()
