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
import os, unittest

PATH_PREFIX = os.path.join("/tmp/", "tc_test")

if not os.path.exists(PATH_PREFIX):
    os.makedirs(PATH_PREFIX)


class TestAvgPoolAutotune(unittest.TestCase):

    def test_avgpool_autotune(self):
        # NOTE: take note of use of {{ }} below for handling TC with scalars
        LANG = """
        def avgpool(float(B, C, H, W) input) -> (output) {{
            output(b, c, h, w) +=! input(b, c, h * {sH} + kh, w * {sW} + kw) / ({kH} * {kW}) where kh in 0:{kH}, kw in 0:{kW}
        }}
        """
        avgpool = tc.define(LANG, name="avgpool", constants={"sH":1, "sW":1, "kH":2, "kW":2})
        inp = torch.ones(32, 3, 10, 10).cuda()
        best_options = avgpool.autotune(inp, **tc.small_sizes_autotuner_settings)
        out = avgpool(inp, options=best_options)


if __name__ == '__main__':
    unittest.main()
