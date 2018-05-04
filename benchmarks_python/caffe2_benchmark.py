# Copyright (c) 2018-present, Facebook, Inc.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import numpy as np
import os
import tensor_comprehensions as tc

from caffe2.python import core, dyndep, workspace, utils


CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
if CONDA_PREFIX:
    tc_c2_lib = os.path.join(CONDA_PREFIX, "lib/libtc_c2.so")
else:
    dyndep.InitOpsLibrary("@/tc/tc:tc_c2")

FC_LANG = """
  def func_fc(float(B,M) I, float(N,M) W1, float(N) B1) -> (O1) {
    O1(b, n) +=! I(b, r_m) * W1(n, r_m)
    O1(b, n) = O1(b, n) + B1(n)
  }
"""


def GetArgumentParser():
    parser = argparse.ArgumentParser(
        description="Caffe2 benchmark. Extra args will be passed to Caffe2")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The batch size.")
    parser.add_argument("--input_dim", type=int, default=512,
                        help="The input dense dimension.")
    parser.add_argument("--output_dim", type=int, default=128,
                        help="The input dense dimension.")
    parser.add_argument("--num_runs", type=int, default=1000,
                        help="how many times to run each op")
    parser.add_argument("--num_tuner_generations", type=int, default=10,
                        help="Number of iterations to tune")
    parser.add_argument("--tuner_cache_file", type=str,
                        default="tuner_cache",
                        help="File to store tuned mapping options")
    return parser


def main():
    parser = GetArgumentParser()
    args, extra_args = parser.parse_known_args()

    core.GlobalInit([
        'tc_bench',
        '--caffe2_logging_operator_dyno_sampling_rate=0',
        '--tuner_gpus=4',
        '--caffe2_simple_net_benchmark_run_whole_net=0',
    ] + extra_args)
    mapping_options = tune(args)
    compare_fcs(
        args.batch_size,
        args.input_dim,
        args.output_dim,
        args.num_runs,
        mapping_options,
    )


@utils.debug
def tune(args):
    fc = tc.define(FC_LANG, name="func_fc")
    options = fc.autotune(
        (args.batch_size, args.input_dim),
        (args.output_dim, args.input_dim),
        (args.output_dim,),
        generations=args.num_tuner_generations,
        threads=32,
        pop_size=2,
        cache=args.tuner_cache_file,
    )
    print(options.serializeAsText())
    return options


@utils.debug
def compare_fcs(B, M, N, num_runs, mapping_options=None):
    X = np.random.rand(B, M).astype(np.float32) - 0.5
    W = np.random.rand(N, M).astype(np.float32) - 0.5
    b = np.random.rand(N).astype(np.float32) - 0.5

    with core.DeviceScope(core.DeviceOption(1)):
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)
        workspace.FeedBlob("b", b)

    net = core.Net("test")

    with core.DeviceScope(core.DeviceOption(1)):
        net.FC(["X", "W", "b"], "Y_baseline")
        net.TcOp(
            ["X", "W", "b"], "Y_TC",
            tcDef=FC_LANG,
            tcName="func_fc",
            mappingOptions=(
                mapping_options.serialize() if mapping_options else None),
            checkSizes=True,
        )

    workspace.CreateNet(net)
    workspace.RunNet(net)

    baseline_value = workspace.blobs["Y_baseline"]
    tc_value = workspace.blobs["Y_TC"]
    np.testing.assert_allclose(
        baseline_value,
        tc_value,
        rtol=1e-4,
        atol=1e-4,
    )

    runtimes = workspace.BenchmarkNet(
        net.Name(),
        0,  # warmpup was already done
        num_runs,
        True,  # run individual ops
    )[1:]

    print(runtimes)


if __name__ == '__main__':
    main()
