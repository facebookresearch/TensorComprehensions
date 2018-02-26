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

import unittest, os
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, dyndep

tc_c2_lib = os.path.join(os.environ.get("CONDA_PREFIX"), "lib/libtc_c2.so")
dyndep.InitOpsLibrary(tc_c2_lib)


class TestCaffe2(unittest.TestCase):

    def test_matmul_caffe2(self):
        lang = """
        def matmul(float(M,N) A, float(N,K) B) -> (output) {
          output(i, j) +=! A(i, kk) * B(kk, j)
        }
        """
        # TODO: (prigoyal) serialize the options
        # options = Options("mlp")
        mat1, mat2 = np.random.rand(100, 400), np.random.rand(400, 500)
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            workspace.FeedBlob('mat1', mat1.astype(np.float32))
            workspace.FeedBlob('mat2', mat2.astype(np.float32))
            matmul = core.CreateOperator(
                "TcOp", ["mat1", "mat2"], ["out"], lang=lang, tcName="matmul"
            )
        workspace.RunOperatorOnce(matmul)
        out = workspace.FetchBlob("out")


if __name__ == '__main__':
    unittest.main()
