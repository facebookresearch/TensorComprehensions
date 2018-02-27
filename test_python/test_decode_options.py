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
import unittest, os

PATH_PREFIX = os.path.join("/tmp/", "tc_test")
if not os.path.exists(PATH_PREFIX):
    os.makedirs(PATH_PREFIX)


class TestCase(unittest.TestCase):

    def test_decode_options(self):
        cache = "{}/matmul_3_4_5".format(PATH_PREFIX)
        lang = """
        def matmul(float(M,N) A, float(N,K) B) -> (output) {
          output(i, j) +=! A(i, kk) * B(kk, j)
        }
        """
        matmul = tc.define(lang, name="matmul")
        matmul.autotune((3, 4), (4, 5), cache=cache, **tc.small_sizes_autotuner_settings)
        tc.decode(cache + ".options")


if __name__ == '__main__':
    unittest.main()
