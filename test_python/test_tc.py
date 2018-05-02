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

import os

import torch
import torch.cuda

from tensor_comprehensions import TcCompilationUnit
from common import TestCase, run_tests

PATH_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tc_test")

if not os.path.exists(PATH_PREFIX):
    os.makedirs(PATH_PREFIX)


class TensorDotTest(TestCase):
    def build_tc_problem(self):
        lang = """
        def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
          O(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
        }
        """
        tc_name = "tensordot"
        tc_type = "conv"

        N = 32
        C1 = 512
        C2 = 8
        C3 = 2
        H = 28
        W = 28
        cache_filename = "{}/tensordot_cache_N_{}_C1_{}_C2_{}_C3_{}_H_{}_W{}".format(
            PATH_PREFIX, N, C1, C2, C3, H, W)
        I0 = torch.randn(N, C1, C2, H, W).cuda()
        I1 = torch.randn(N, C2, C3, H, W).cuda()
        inputs = [I0, I1]
        return lang, tc_name, tc_type, cache_filename, inputs

    def test_tensordot_autotune_first(self):
        lang, tc_name, tc_type, cache_filename, inputs = self.build_tc_problem()
        print("\n====> Autotuning kernel and saving results")
        options = self.autotune_store(cache_filename, lang, tc_name, inputs, tc_type)
        print("\n====> Running the kernel with autotuned options")
        self.check(lang, tc_name, options, inputs, outputs=None)

    def test_tensordot_autotune_load(self):
        lang, tc_name, tc_type, cache_filename, inputs = self.build_tc_problem()
        print("\n====> Loading the autotuned options")
        options = self.autotune_load(cache_filename, lang, tc_name, inputs)
        print("\n====> Running the kernel with loaded options")
        self.check(lang, tc_name, options, inputs, outputs=None)


class TestMatmul(TestCase):
    def test_matmul(self):
        lang = """
        def matmul(float(M,N) A, float(N,K) B) -> (output) {
          output(i, j) +=! A(i, kk) * B(kk, j)
        }
        """
        cu = TcCompilationUnit()
        cu.define(lang)

        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        inputs = [mat1, mat2]
        cu.compile("matmul", [mat1, mat2], options="mlp")
        outputs = cu.run("matmul", inputs)
        torch.cuda.synchronize()
        expected = torch.mm(mat1, mat2)
        torch.cuda.synchronize()
        diff = outputs[0] - expected
        self.assert_almost_equal(diff, inputs, 4)

        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        inputs = [mat1, mat2]
        cu.run("matmul", inputs, outputs=outputs)
        expected = torch.mm(mat1, mat2)
        diff = outputs[0] - expected
        self.assert_almost_equal(diff, inputs, 4)


if __name__ == '__main__':
    run_tests()
