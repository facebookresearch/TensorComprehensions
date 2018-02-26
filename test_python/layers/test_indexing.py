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


class TestIndexing(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
