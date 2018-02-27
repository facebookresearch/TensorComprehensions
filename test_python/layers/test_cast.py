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


class TestCast(unittest.TestCase):

    def test_cast(self):
        LANG = """
        def cast(float(M,N) A) -> (int32(M,N) O1) {{
            O1(m, n) = int32(A(m, n) + {constant})
        }}
        """
        cast = tc.define(LANG, name="cast", constants={"constant": 0.3})
        A = torch.randn(32, 16).cuda()
        out = cast(A)


if __name__ == '__main__':
    unittest.main()
