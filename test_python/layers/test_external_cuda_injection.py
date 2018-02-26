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


class TestManualCudaInjection(unittest.TestCase):
    def test_simple_cuda_injection(self):
        lang = """
        def add(float(N) A, float(N) B) -> (output) {
            output(i) = A(i) + B(i)
        }
        """

        cuda_code = """
        extern "C"{
        __global__ void my_add(float* __restrict__ output, const float* __restrict__ A, const float* __restrict B)
        {
            int t = threadIdx.x;
            output[t] = A[t] + B[t];
        }
        }
        """

        add = tc.define(lang, name="add", inject_kernel="my_add", cuda_code=cuda_code)
        a, b = torch.randn(100).cuda(), torch.randn(100).cuda()
        out = add(a, b, grid=[1, 1, 1], block=[100, 1, 1])


if __name__ == '__main__':
    unittest.main()


# TODO: add test for 'where', cpu codepath
