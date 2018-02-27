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


class TestScale(unittest.TestCase):

    # NOTE: take note of use of {{ }}
    def test_scale(self):
        LANG = """
        def scale(float(M, N) I) -> (O) {{
          O(m, n) = I(m, n) * {s}
        }}
        """
        scale = tc.define(LANG, name="scale", constants={"s": 10})
        inp = torch.randn(100, 128).cuda()
        out = scale(inp)


if __name__ == '__main__':
    unittest.main()
