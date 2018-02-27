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


class TestCosineSimilarity(unittest.TestCase):

    # NOTE: TC can't do allocations itself, so everything has to be declared
    # as input or output. Hence, we return the temporary outputs as well
    def test_cosine_similarity(self):
        LANG = """
        def cosine_similarity(float(M, N) I1, float(M, N) I2) -> (O, sumI1, sumI2) {{
            sumI1(m) +=! I1(m, n) * I1(m, n)
            sumI2(m) +=! I2(m, n) * I2(m, n)
            O(m) +=! (I1(m, n) * I2(m, n)) / fmax(rsqrt(sumI1(m)) * sqrt(sumI2(m)), {eps})
        }}
        """
        cosine_similarity = tc.define(LANG, name="cosine_similarity", constants={"eps": 1e-5})
        inp1, inp2 = torch.randn(100, 128).cuda(), torch.randn(100, 128).cuda()
        out = cosine_similarity(inp1, inp2)


if __name__ == '__main__':
    unittest.main()
