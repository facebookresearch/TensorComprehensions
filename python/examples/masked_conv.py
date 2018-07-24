#!/usr/bin/env python3

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

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as functional

torch.backends.cudnn.benchmark = True


def GetArgumentParser():
    parser = argparse.ArgumentParser(
        description='Lengths Cosine Coherence benchmark.'
    )
    parser.add_argument(
        '--tuner_threads', type=int, default=16,
        help='Number of CPU tuning threads.',
    )
    parser.add_argument(
        '--tuner_generations', type=int, default=25,
        help='Number of tuning generations.',
    )
    parser.add_argument(
        '--tuner_pop_size', type=int, default=100,
        help='Number candidates per tuning generations.',
    )
    parser.add_argument(
        '--tuner_number_elites', type=int, default=5,
        help='Number of best tuning candidates that survive each generation.',
    )
    parser.add_argument(
        '--tuner_devices', type=str, default='0',
        help='Comma separated list of tuning devices.',
    )
    parser.add_argument(
        '--tuner_cache_file',
        type=str,
        default='/tmp/cache_condensenet',
        help='File to store tuned mapping options',
    )
    return parser


parser = GetArgumentParser()
args, extra_args = parser.parse_known_args()


###############################################################################
# TC equivalent converting control-flow to data dependencies
###############################################################################
MASKED_CONVOLVE = '''
def masked_convolve(float(B, C, H, W) Input,
                    float(F, C, K, K) Weights,
                    uint8(F, C) Mask) -> (Output) {
    Output(b, f, h, w) +=! (Mask(f, r_c) == 1) ?
        fmax(0.0, Input(b, r_c, h + r_k1, w + r_k2)) *
        Weights(f, r_c, r_k1, r_k2) :
        0.0
}
'''

###############################################################################
# Implicit compilation and tuning behavior
###############################################################################
tuner_config = (
    tc.TunerConfig()
    .threads(args.tuner_threads)
    .generations(args.tuner_generations)
    .pop_size(args.tuner_pop_size)
    .number_elites(args.tuner_number_elites)
    .devices(args.tuner_devices))
reinforce_list = ['']


def generate_options(tc_str: str,
                     entry_point: str,
                     *inputs: torch.Tensor) -> tc.MappingOptions:
    global reinforce

    # TODO: comment the line below which serves the purpose of not blowing up
    # CI time
    return tc.make_naive_options_factory()(tc_str, entry_point, *inputs)

    if entry_point == 'make_idx':
        return tc.make_naive_options_factory()(tc_str, entry_point, *inputs)

    loaded = tc.make_load_from_cache_options_factory(args.tuner_cache_file)(
        tc_str, entry_point, *inputs)

    if loaded is None or entry_point in reinforce_list or '*' in reinforce_list:
        start = loaded if loaded is not None else 'naive'
        return tc.make_autotuned_options_factory(
            starting_options=start,
            tuner_config=tuner_config,
            cache_filename=args.tuner_cache_file,
            store_to_cache=True,)(tc_str, entry_point, *inputs)

    assert loaded is not None, 'None found'

    return loaded


###############################################################################
# Define the TC for MASKED_CONVOLVE
###############################################################################
TC = tc.define(MASKED_CONVOLVE, generate_options)

###############################################################################
# Run with implicit compilation and tuning
###############################################################################

# sizes:
H, W, C, B, F, K = 56, 56, 128, 32, 32, 1

# Pytorch:
conv = nn.Conv2d(C, F, K, 1, 0, 1, groups=1, bias=False).cuda()
relu = nn.ReLU(inplace=True).cuda()
input_data = torch.zeros(B, C, H, W).cuda(non_blocking=True)
mask = torch.randn(F, C, K, K).gt_(0.).cuda(non_blocking=True)
torch.cuda.synchronize()

weight = conv.weight * mask
rectified_input = relu(input_data)
output = functional.conv2d(rectified_input, weight, None, conv.stride,
                           conv.padding, conv.dilation, 1)

# TC:
InputData = input_data
Weights = conv.weight
Mask = mask.view(F, C).byte()
torch.cuda.synchronize()
Output = TC.masked_convolve(InputData, Weights, Mask)


###############################################################################
# Check
###############################################################################
tc.assert_almost_equal(
    output.cpu(),
    Output.cpu(),
    input_data.cpu(), conv.weight.cpu(), mask.cpu(),
    operations=C * K * K,
    precision=1e-7)

print('SUCCESS')
