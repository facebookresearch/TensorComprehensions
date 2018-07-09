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
import torch

from hypothesis import given, settings
from caffe2.python import core, dyndep


CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
if CONDA_PREFIX:
    tc_c2_lib = os.path.join(CONDA_PREFIX, "lib/libtc_c2.so")
else:
    dyndep.InitOpsLibrary("@/tc/tc:tc_c2")

MATMUL_LANG = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
    output(m, k) +=! A(m, r_n) * B(r_n, k)
}
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
            tc_grad_def=MATMUL_LANG,
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
        tuner = tc.Tuner(MATMUL_LANG)
        tuner_config = (
            tc.TunerConfig().generations(3).threads(32).pop_size(2)
            .tuner_min_launch_total_threads(1))
        matmul_top1 = tuner.tune(
            'matmul',
            (torch.randn(n, k, device='cuda'),
             torch.randn(k, m, device='cuda')),
            tc.MappingOptions('naive'),
            tuner_config)
        matmul_grad_top1 = tuner.tune(
            'matmul_grad',
            (torch.randn(n, k, device='cuda'),
             torch.randn(k, m, device='cuda'),
             torch.randn(n, m, device='cuda')),
            tc.MappingOptions('naive'),
            tuner_config)

        X = np.random.rand(m, k).astype(np.float32)
        W = np.random.rand(k, n).astype(np.float32)

        def ref(X, W):
            return [np.dot(X, W)]

        op = core.CreateOperator(
            "TcOp", ["X", "Y"], "out",
            tc_def=MATMUL_LANG,
            tc_name="matmul",
            tc_grad_def=MATMUL_LANG,
            tc_grad_name="matmul_grad",
            inputs_used_by_gradient=[0, 1],
            output_gradients_used_by_gradient=[0],
            inputs_to_compute_gradients_of=[0, 1],
            mapping_options=matmul_top1.serialize(),
            grad_mapping_options=matmul_grad_top1.serialize(),
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
