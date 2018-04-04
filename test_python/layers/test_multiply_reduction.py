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
import unittest


class TestMultReduction(unittest.TestCase):

   def test_mult_reduction(self):
       LANG = """
       def conv_mult(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
           O(n, m, h, w) *=! I(n, c, h + kh, w + kw) * W1(m, c, kh, kw)
       }
       """
       N, C, H, W, O, kH, kW = 64, 10, 24, 24, 10, 7, 7
       conv_mult = tc.define(LANG, name="conv_mult")
       I, W1 = torch.ones(N, C, H, W).cuda(), torch.ones(O, C, kH, kW).cuda()
       # Note: There is no bias here
       out = conv_mult(I, W1)
       assert out.data.min() > 0


if __name__ == '__main__':
   unittest.main()
