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


class TestLookupTable(unittest.TestCase):

    def test_lookup_table(self):
        LANG = """
        def lut(float(B, R) LUT, int32(B, N) I) -> (O) {
          O(b, n) +=! LUT(I(b, n), r)
        }
        """
        lut = tc.define(LANG, name="lut")
        inp = torch.rand(17, 22).cuda()
        idx = torch.IntTensor(17, 82).fill_(1).cuda()
        out = lut(inp, idx)


if __name__ == '__main__':
    unittest.main()
