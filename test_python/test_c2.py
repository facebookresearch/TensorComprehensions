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
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import tensor_comprehensions as tc

from hypothesis import given, settings
from caffe2.python import core, dyndep


CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
if CONDA_PREFIX:
    tc_c2_lib = os.path.join(CONDA_PREFIX, "lib/libtc_c2.so")
else:
    dyndep.InitOpsLibrary("@/tc/tc:tc_c2")

MATMUL_LANG = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
output(i, j) +=! A(i, kk) * B(kk, j)
}
"""

class TestCaffe2(hu.HypothesisTestCase):
    @given(n=st.integers(1, 128),
           m=st.integers(1, 128),
           k=st.integers(1, 128),
           seed=st.integers(min_value=0, max_value=2**32 - 1),
           **hu.gcs_gpu_only)
    def test_matmul(self, n, m, k, seed, gc, dc):
        np.random.seed(seed)

        X = np.random.rand(m, k).astype(np.float32)
        W = np.random.rand(k, n).astype(np.float32)

        def ref(X, W):
            return [np.dot(X, W)]

        op = core.CreateOperator(
            "TcOp", ["X", "Y"], "out",
            tcDef=MATMUL_LANG,
            tcName="matmul",
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, W],
            reference=ref,
        )

    @given(n=st.integers(1, 128),
           m=st.integers(1, 128),
           k=st.integers(1, 128),
           seed=st.integers(min_value=0, max_value=2**32 - 1),
           **hu.gcs_gpu_only)
    @settings(max_examples=2)
    def test_matmul_tune_and_run(self, n, m, k, seed, gc, dc):
        matmul = tc.define(MATMUL_LANG, name="matmul")

        mapping_options = matmul.autotune(
            (n, k), (k, m),
            generations=1,
            threads=32,
            pop_size=2,
            tuner_min_launch_total_threads=1,
        )

        X = np.random.rand(m, k).astype(np.float32)
        W = np.random.rand(k, n).astype(np.float32)

        def ref(X, W):
            return [np.dot(X, W)]

        op = core.CreateOperator(
            "TcOp", ["X", "Y"], "out",
            tcDef=MATMUL_LANG,
            tcName="matmul",
            mappingOptions=mapping_options.serialize(),
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, W],
            reference=ref,
        )

if __name__ == '__main__':
    unittest.main()
