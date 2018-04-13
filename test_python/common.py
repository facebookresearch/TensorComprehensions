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

import unittest

from tensor_comprehensions import TcCompilationUnit
from tensor_comprehensions import TcAutotuner


def run_tests():
    unittest.main()


class TestCase(unittest.TestCase):

    def assert_almost_equal(self, diff, inputs, operations, precision=1e-7):
        max_value = 0.0
        for inp in inputs:
            max_value = max(float(inp.abs().max()), max_value)
        max_diff = float(diff.abs().max())
        self.assertLess(
            0, operations * precision * max_value,
            "error at relative precision: {}, #operations: {}, max_value: {}, max_diff: {}".format(precision, operations, max_value, max_diff))

    def autotune_store(self, cache_file, lang, tc_name, inputs, tc_type):
        tuner = TcAutotuner(
            lang, threads=16, pop_size=10, generations=1
        )
        best_options = tuner.tune_and_store(
            tc_name, inputs, mapping_options=tc_type, cache_file=cache_file
        )
        return best_options

    def autotune_load(
        self, cache_file, lang, tc_name, inputs, num_candidates=1
    ):
        tuner = TcAutotuner(lang)
        best_options = tuner.load(cache_file, tc_name, inputs, num_candidates)
        return best_options

    def check(self, lang, tc_name, options, inputs, outputs=None):
        cu = TcCompilationUnit()
        cu.define(lang)
        handle = cu.compile(tc_name, inputs, options=options)
        outputs = cu.run(handle, tc_name, inputs, outputs=outputs)
        return outputs
