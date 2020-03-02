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
import time
import torch

# Define a timing function to print some results
def time_tc(iters, prepend, runFun, entry_point, inputs):
    timesCPU = []
    timesCPUAndGPU = []
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.clock()
        outputs = runFun(entry_point, inputs)
        timesCPU.append(time.clock() - start)
        torch.cuda.synchronize()
        timesCPUAndGPU.append(time.clock() - start)
    print("#################################################################")
    timesCPU = sorted(timesCPU)
    print("{} Total CPU time to launch kernel: min {}us, p50 {}us, p90 {}us, max {}us".format(
        prepend,
        int(timesCPU[0] * 1e6),
        int(timesCPU[int(len(timesCPU) // 2)] * 1e6),
        int(timesCPU[int((len(timesCPU) * 9) // 10)] * 1e6),
        int(timesCPU[len(timesCPU) - 1] * 1e6),
    ))
    timesCPUAndGPU = sorted(timesCPUAndGPU)
    print("{} Total CPU launch + GPU kernel time: min {}us, p50 {}us, p90 {}us, max {}us".format(
        prepend,
        int(timesCPUAndGPU[0] * 1e6),
        int(timesCPUAndGPU[int(len(timesCPUAndGPU) // 2)] * 1e6),
        int(timesCPUAndGPU[int((len(timesCPUAndGPU) * 9) // 10)] * 1e6),
        int(timesCPUAndGPU[len(timesCPUAndGPU) - 1] * 1e6),
    ))
