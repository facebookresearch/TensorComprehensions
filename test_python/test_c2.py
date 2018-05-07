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
    output(m, n) +=! A(m, r_n) * B(r_n, k)
}
"""

MATMUL_GRAD_LANG = """
def matmul_grad(float(M, N) A, float(N, K) B, float(M, K) d_O) -> (d_A, d_B) {
    d_A(m, n) +=! d_O(m, r_k) * B(n, r_k)
    d_B(n, k) +=! d_O(r_m, k) * A(r_m, n)
}
"""

class TestCaffe2(hu.HypothesisTestCase):
    @given(n=st.integers(1, 4),
           m=st.integers(1, 4),
           k=st.integers(1, 4),
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
            tc_def=MATMUL_LANG,
            tc_name="matmul",
            tc_grad_def=MATMUL_GRAD_LANG,
            tc_grad_name="matmul_grad",
            inputs_used_by_gradient=[0, 1],
            output_gradients_used_by_gradient=[0],
            inputs_to_compute_gradients_of=[0, 1],
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, W],
            reference=ref,
        )

        for i in range(2):
            self.assertGradientChecks(
                device_option=gc,
                op=op,
                inputs=[X, W],
                outputs_to_check=i,
                outputs_with_grads=[0],
            )

    @given(n=st.integers(1, 4),
           m=st.integers(1, 4),
           k=st.integers(1, 4),
           seed=st.integers(min_value=0, max_value=2**32 - 1),
           **hu.gcs_gpu_only)
    @settings(max_examples=2)
    def test_matmul_tune_and_run(self, n, m, k, seed, gc, dc):
        matmul = tc.define(MATMUL_LANG, name="matmul")
        matmul_grad = tc.define(MATMUL_GRAD_LANG, name="matmul_grad")

        mapping_options = matmul.autotune(
            (n, k), (k, m),
            generations=3,
            threads=32,
            pop_size=2,
            tuner_min_launch_total_threads=1,
        )

        grad_mapping_options = matmul_grad.autotune(
            (n, k), (k, m), (n, m),
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
            tc_def=MATMUL_LANG,
            tc_name="matmul",
            tc_grad_def=MATMUL_GRAD_LANG,
            tc_grad_name="matmul_grad",
            inputs_used_by_gradient=[0, 1],
            output_gradients_used_by_gradient=[0],
            inputs_to_compute_gradients_of=[0, 1],
            mapping_options=mapping_options.serialize(),
            grad_mapping_options=grad_mapping_options.serialize(),
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, W],
            reference=ref,
        )

        for i in range(2):
            self.assertGradientChecks(
                device_option=gc,
                op=op,
                inputs=[X, W],
                outputs_to_check=i,
                outputs_with_grads=[0],
            )

if __name__ == '__main__':
    unittest.main()
