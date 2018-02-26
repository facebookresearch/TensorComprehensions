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


class TestSmallMobileNet(unittest.TestCase):

    def test_small_mobilenet(self):
        LANG = """
        def small_mobilenet(float(C1, H, W) I, float(C1, KH1, KW1) W1,
                            float(C1) B1, float(C2, C1) W2, float(C2) B2)
        -> (O1, O2)
        {
            O1(c1, h, w) +=! I(c1, h + kh, w + kw) * W1(c1, kh, kw)
            O1(c1, h, w)  = O1(c1, h, w) + B1(c1)
            O1(c1, h, w)  = fmax(O1(c1, h, w), 0)

            O2(c2, h, w) +=! O1(c1, h, w) * W2(c2, c1)
            O2(c2, h, w)  = O2(c2, h, w) + B2(c2)
            O2(c2, h, w)  = fmax(O2(c2, h, w), 0)
        }
        """
        C1, C2, H, W, KH1, KH2 = 128, 128, 16, 16, 3, 3
        small_mobilenet = tc.define(LANG, name="small_mobilenet")
        I, W1 = torch.randn(C1, H, W).cuda(), torch.randn(C1, KH1, KH2).cuda()
        B1, W2= torch.randn(C1).cuda(), torch.randn(C2, C1).cuda()
        B2 = torch.randn(C2).cuda()
        best_options = small_mobilenet.autotune(I, W1, B1, W2, B2, **tc.autotuner_settings)
        out = small_mobilenet(I, W1, B1, W2, B2, options=best_options)


if __name__ == '__main__':
    unittest.main()
