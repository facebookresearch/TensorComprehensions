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


class TestBroadcastFCRelu(unittest.TestCase):

    def test_broadcast_fcrelu(self):
        LANG = """
        def fcrelu(float(B, M) I, float(N, M) W1, float(N) B1) -> (O1) {
            O1(b, n) = B1(n) where b in 0:B
            O1(b, n) += I(b, m) * W1(n, m)
            O1(b, n) = fmax(O1(b, n), 0)
        }
        """
        B, M, N = 100, 128, 100
        fcrelu = tc.define(LANG, name="fcrelu")
        I, W1, B1 = torch.randn(B, M).cuda(), torch.randn(N, M).cuda(), torch.randn(N).cuda()
        out = fcrelu(I, W1, B1)


if __name__ == '__main__':
    unittest.main()
