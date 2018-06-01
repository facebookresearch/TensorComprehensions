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

tc.SetDebugFlags(debug_tuner=False, debug_tc_mapper=False)


class TestLayerNorm(unittest.TestCase):

    def test_layernorm(self):
        # NOTE: take note of use of {{ }} below for handling TC with scalars
        lang = """
        def layernorm(float(T, B, C) I) -> (O, mean, centered, var)
        {{
           mean(t, b) +=! I(t, b, c) / C
           centered(t, b, c) = I(t, b, c) - mean(t, b)
           var(t, b) +=! centered(t, b, c) * centered(t, b, c)
           var(t, b) = (var(t, b) + {eps}) / C
           O(t, b, c) = centered(t, b, c) / rsqrt(var(t, b))
        }}
        """
        layernorm = tc.define(lang, name="layernorm", constants={"eps": 1e-5})
        inp = torch.randn(7, 32, 64).cuda()
        options = tc.CudaMappingOptions("mlp")
        options = layernorm.autotune(inp, **tc.autotuner_settings)
        out = layernorm(inp, options=options)


if __name__ == '__main__':
    unittest.main()
