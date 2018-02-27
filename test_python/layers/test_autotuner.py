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
from tensor_comprehensions.mapping_options import Options

import torch
import torch.cuda
import os, unittest

MATMUL_LANG = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""

PATH_PREFIX = os.path.join("/tmp/", "tc_test")

if not os.path.exists(PATH_PREFIX):
    os.makedirs(PATH_PREFIX)


###########################################################################
# Autotuner tests
###########################################################################
class TestAutotuner(unittest.TestCase):

    ###########################################################################
    # Pass tuple inputs for autotuning
    ###########################################################################
    def test_autotuner_tuple_size_no_cache(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        matmul.autotune((3, 4), (4, 5), **tc.small_sizes_autotuner_settings)
        matmul.autotune((100, 400), (400, 500), **tc.autotuner_settings)

    def test_autotuner_tuple_size_cache_to_default(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        matmul.autotune((3, 4), (4, 5), cache=True, **tc.small_sizes_autotuner_settings)
        matmul.autotune((100, 400), (400, 500), cache=True, **tc.autotuner_settings)

    def test_autotuner_tuple_size_cache_to_file_run_kernel(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        cache1 = "{}/matmul_3_4_5".format(PATH_PREFIX)
        cache2 = "{}/matmul_100_400_500".format(PATH_PREFIX)
        matmul.autotune((3, 4), (4, 5), cache=cache1, **tc.small_sizes_autotuner_settings)
        matmul.autotune((100, 400), (400, 500), cache=cache2, **tc.autotuner_settings)

        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2, cache=cache1)

        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        out = matmul(mat1, mat2, cache=cache2)

    ###########################################################################
    # Pass Tensors for autotuning
    ###########################################################################
    # NOTE: Use "--tuner_min_launch_total_threads=1" for running small sizes
    # tc.small_sizes_autotuner_settings has this option set already
    def test_autotuner_no_cache_small_size(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        options = matmul.autotune(mat1, mat2, **tc.small_sizes_autotuner_settings)

    def test_autotuner_no_cache(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        options = matmul.autotune(mat1, mat2, **tc.autotuner_settings)

    def test_autotuner_no_cache_explicit_set(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        options = matmul.autotune(mat1, mat2, cache=False, **tc.autotuner_settings)

    def test_autotuner_cache_to_default(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        matmul.autotune(mat1, mat2, cache=True, **tc.autotuner_settings)

    def test_autotuner_cachefile_first(self):
        cache_file = "{}/matmul_100_400_500".format(PATH_PREFIX)    # use argparse if input from command line
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        matmul.autotune(mat1, mat2, cache=cache_file, **tc.autotuner_settings)

    def test_autotuner_cachefile_load(self):
        lang = MATMUL_LANG
        cache_file = "{}/matmul_100_400_500".format(PATH_PREFIX)    # use argparse if input from command line
        assert os.path.isfile("{}.cuda".format(cache_file)), "looks like the cache_file doesn't exist"

        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        out = matmul(mat1, mat2, cache=cache_file)

    def test_autotuner_no_cache_and_run_kernel(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        options = matmul.autotune(mat1, mat2, **tc.autotuner_settings)
        out = matmul(mat1, mat2, options=options)

    def test_autotuner_start_options_and_run_kernel(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        options = Options("mlp")
        best_options = matmul.autotune(mat1, mat2, cache=True, options=options, **tc.autotuner_settings)
        out = matmul(mat1, mat2, options=best_options)


if __name__ == '__main__':
    unittest.main()
