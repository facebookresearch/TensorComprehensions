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


class TestSoftmax(unittest.TestCase):

    # one can't reuse names on the LHS once they've been used in another tensor
    # otherwise the size of the tensors is ill-defined
    def test_softmax(self):
        LANG = """
        def softmax(float(N, D) I) -> (O, maxVal, expDistance, expSum) {
          maxVal(n) max= I(n, d)
          expDistance(n, d) = exp(I(n, d) - maxVal(n))
          expSum(n) +=! expDistance(n, d)
          O(n, d) = expDistance(n, d) / expSum(n)
        }
        """
        softmax = tc.define(LANG, name="softmax")
        inp = torch.randn(32, 16).cuda()
        out = softmax(inp)


if __name__ == '__main__':
    unittest.main()
