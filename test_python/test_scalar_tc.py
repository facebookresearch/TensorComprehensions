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
import unittest, os


class TestCase(unittest.TestCase):

    def test_avgpool_option1(self):
        # NOTE: take note of use of {{ }} below for handling TC with scalars
        LANG = """
        def avgpool(float(B, C, H, W) input) -> (output) {{
            output(b, c, h, w) += input(b, c, h * {sH} + kh, w * {sW} + kw) where kh in 0:{kH}, kw in 0:{kW}
        }}
        """
        avgpool = tc.define(LANG, name="avgpool", constants={"sH":1, "sW":1, "kH":2, "kW":2})
        inp = torch.ones(32, 3, 10, 10).cuda()
        out = avgpool(inp)

    def test_avgpool_option2(self):
        # NOTE: take note of use of {{ }}
        LANG="""
        def avgpool(float(B, C, H, W) input) -> (output) {{
            output(b, c, h, w) += input(b, c, h * {sh} + kh, w * {sw} + kw) where kh = [0, {kH}[, kw = [0, {kW}[
        }}
        """
        sH, sW, kH, kW = 1, 1, 2, 2
        # format the strings yourself before passing to TC backend.
        LANG = LANG.format(sh=sH, sw=sW, kH=kH, kW=kW)
        avgpool = tc.define(LANG, name="avgpool")
        inp = torch.ones(1, 1, 4, 4).cuda()
        out = avgpool(inp)

    def test_avgpool_option3(self):
        # If you prefer to do string substitutions yourself, here is another way below
        import re
        LANG="""
        def avgpool(float(B, C, H, W) input) -> (output) {
            output(b, c, h, w) += input(b, c, h * <sh> + kh, w * <sw> + kw) where kh in 0:<kH>, kw in 0:<kW>
        }
        """
        sH, sW, kH, kW = 1, 1, 2, 2
        LANG = re.sub('<sh>', str(sH), LANG)
        LANG = re.sub('<sw>', str(sW), LANG)
        LANG = re.sub('<kH>', str(kH), LANG)
        LANG = re.sub('<kW>', str(kW), LANG)
        avgpool = tc.define(LANG, name="avgpool")
        inp = torch.ones(1, 1, 4, 4).cuda()
        out = avgpool(inp)


if __name__ == '__main__':
    unittest.main()
