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
import numpy as np
import torch

def GetArgumentParser():
    parser = argparse.ArgumentParser(
        description='Lengths Cosine Coherence benchmark.'
    )
    parser.add_argument(
        '--num_segs', type=int, default=4, help='The number of segments.'
    )
    parser.add_argument(
        '--seg_length', type=int, default=100, help='The length of each segment.'
    )
    parser.add_argument(
        '--num_of_channels', type=int, default=128, help='The dimension of embeddings.'
    )
    parser.add_argument(
        '--pos_dist', type=int, default=0, help='The positive window size.'
    )
    parser.add_argument(
        '--neg_dist', type=int, default=0, help='The negative window size.'
    )

    parser.add_argument(
        '--tuner_threads', type=int, default=16, help='Number of CPU tuning threads.'
    )
    parser.add_argument(
        '--tuner_generations', type=int, default=25, help='Number of tuning generations.'
    )
    parser.add_argument(
        '--tuner_pop_size', type=int, default=100, help='Number candidates per tuning generations.'
    )
    parser.add_argument(
        '--tuner_number_elites', type=int, default=5, help='Number of best tuning candidates that survive each generation.'
    )
    parser.add_argument(
        '--tuner_devices', type=str, default='0', help='Comma separated list of tuning devices.'
    )
    parser.add_argument(
        '--tuner_cache_file',
        type=str,
        default='/tmp/cache_tum',
        help='File to store tuned mapping options',
    )
    return parser


parser = GetArgumentParser()
args, extra_args = parser.parse_known_args()

###############################################################################
# Reference python impl
###############################################################################
def reference(D, L):
    R = np.zeros(shape=(L.size,), dtype=D.dtype)
    Normed_DATA = D * 0
    Norm_of_Vector = np.zeros(shape=(D.shape[0],), dtype=D.dtype)
    POS_C = np.zeros(shape=(L.size,), dtype=np.long)
    NEG_C = np.zeros(shape=(L.size,), dtype=np.long)
    line = 0
    kEps = 1e-12

    def dot(a, b):
        return np.sum(a * b)

    for i in range(D.shape[0]):
        Norm_of_Vector[i] = dot(D[i], D[i])
        Normed_DATA[i] = D[i] / np.sqrt(max(Norm_of_Vector[i], kEps))

    for g in range(L.size):
        if L[g] <= 1:
            line += L[g]
            continue
        pos_res = 0
        neg_res = 0
        for i in range(L[g] - 1):
            for j in range(i + 1, L[g]):
                sqrt_norm = np.sqrt(
                    max(Norm_of_Vector[line + i], kEps)
                    * max(Norm_of_Vector[line + j], kEps)
                )
                if args.pos_dist == 0 or j - i <= args.pos_dist:
                    pos_res += dot(D[line + i], D[line + j]) / sqrt_norm
                    POS_C[g] += 1
                if args.neg_dist > 0 and j - i >= args.neg_dist:
                    neg_res += dot(D[line + i], D[line + j]) / sqrt_norm
                    NEG_C[g] += 1
        pos_res = 0 if POS_C[g] < 1 else pos_res / POS_C[g]
        neg_res = 0 if NEG_C[g] < 1 else neg_res / NEG_C[g]
        R[g] = pos_res - neg_res
        line += L[g]
    return [R, Normed_DATA, Norm_of_Vector, POS_C, NEG_C]

###############################################################################
# TC equivalent converting control-flow to data dependencies
###############################################################################
LENGTHS_COSINE_COHERENCE = '''
# TODO: this is just a scan but currently implemented as K reductions
def make_idx(int32(K) Segments) -> (Idx) {
    Idx(k) +=! (r_k < k) ? Segments(r_k) : 0 where k in 0:K+1
}
def make_alpha(int32(KP1) Idx, int32(MAX_L) SegmentsMetaData) -> (Alpha) {
    # Triangular compute
    Alpha(k, max_l_1, max_l_2) = (max_l_1 >= max_l_2) ? 0.0 :
        # This computes an approximation using the maximal segment length
        ((<pos_dist> == 0 || fabs(float(max_l_1 - max_l_2)) <= float(<pos_dist>)) ? 1.0 :
         (<neg_dist> == 0 && fabs(float(max_l_1 - max_l_2)) >= float(<neg_dist>)) ? -1.0 : 0.0)
    *
        # Filter against the true value of Idx
        ((Idx(k) + max_l_1 < Idx(k + 1) && Idx(k) + max_l_2 < Idx(k + 1))
         ? 1.0 : 0.0)
            where k in 0:KP1-1, max_l_1 in 0:MAX_L, max_l_2 in 0:MAX_L
}
def make_counts(float(K, MAX_L, MAX_L) Alpha, int32(MAX_L) SegmentsMetaData)
        -> (PosCount, NegCount, tmpPos, tmpNeg)
{
    # Triangular compute
    # tmp is necessary for reasonable performance with the current mapper
    # because we don't yet support 2-D reductions
    # Note that in practice, tmp also gives strictly more parallelism and
    # allows exploiting 2 levels or thread parallelism (doall and reduction) or
    # (in the future when block syncrhonization is supported) 1 level of block
    # parallelism without cross-block reductions.
    tmpPos(k, max_l_1) +=! (max_l_1 >= r_max_l_2) ? 0.0 :
        (Alpha(k, max_l_1, r_max_l_2) > 0.0) ? 1.0 : 0.0
            # TODO: annotation should not be needed
            where k in 0:K, max_l_1 in 0:MAX_L, r_max_l_2 in 0:MAX_L
    PosCount(k) +=! tmpPos(k, max_l_1)
            # TODO: annotation should not be needed
            where k in 0:K, max_l_1 in 0:MAX_L

    # Triangular compute
    # tmp is necessary because we don't yet support 2-D reductions
    # But in practice, tmp also gives strictly more parallelism and allows
    # exploiting blocks without cross-block reductions
    tmpNeg(k, max_l_1) +=! (max_l_1 >= r_max_l_2) ? 0.0 :
        (Alpha(k, max_l_1, r_max_l_2) < 0.0) ? 1.0 : 0.0
            # TODO: annotation should not be needed
            where k in 0:K, max_l_1 in 0:MAX_L, r_max_l_2 in 0:MAX_L
    NegCount(k) +=! tmpNeg(k, max_l_1)
            # TODO: annotation should not be needed
            where k in 0:K, max_l_1 in 0:MAX_L
}
def make_beta(float(K, MAX_L, MAX_L) Alpha,
              float(K) PosCount,
              float(K) NegCount,
              int32(MAX_L) SegmentsMetaData) -> (Beta)
{
    Beta(k, max_l_1, max_l_2) = (max_l_1 >= max_l_2) ? 0.0 :
        (Alpha(k, max_l_1, max_l_2) ==  1.0) ?  1.0 / float(PosCount(k)) :
        (Alpha(k, max_l_1, max_l_2) == -1.0) ? -1.0 / float(NegCount(k)) :
        0.0
        # TODO: annotation should not be needed
        where k in 0:K, max_l_1 in 0:MAX_L, max_l_2 in 0:MAX_L
}

def normalize(float(N, C) Input) -> (NormData, Square) {
    Square(n)     +=! Input(n, r_c) * Input(n, r_c)
    NormData(n, c) =  Input(n,   c) / sqrt(Square(n) + 1e-12)
}
def dots(float(N, C) NormData, int32(KP1) Idx, float(K, MAX_L, MAX_L) Beta) -> (Dots) {
    # Triangular compute
    Dots(k, max_l_1, max_l_2) +=! (max_l_1 >= max_l_2) ? 0.0 :
        # Avoid out of bounds Idx computations
        ((Idx(k) + max_l_1 >= Idx(k + 1) || Idx(k) + max_l_2 >= Idx(k + 1)) ?
          0.0 :
          NormData(Idx(k) + max_l_1, r_c) * NormData(Idx(k) + max_l_2, r_c))
            # TODO: annotation should not be needed
            where k in 0:K, max_l_1 in 0:MAX_L, max_l_2 in 0:MAX_L
}
def result(float(K, MAX_L, MAX_L) Beta, float(K, MAX_L, MAX_L) Dots) -> (O, tmpO) {
    # Triangular compute
    # tmp is necessary because we don't yet support 2-D reductions
    # But in practice, tmp also gives strictly more parallelism and allows
    # exploiting blocks without cross-block reductions
    tmpO(k, max_l_1) +=! (max_l_1 >= r_max_l_2) ? 0.0 :
        Dots(k, max_l_1, r_max_l_2) * Beta(k, max_l_1, r_max_l_2)
    O(k) +=! tmpO(k, max_l_1)
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

# This function is used for reinforcing tuning
# 1. make_idx is small and does not get tuned or saved, just using naive
#    options on it is fine;
# 2. if we find an option in the cache, use it either as is or as starting
#    point for reinforcement, depending on whether the entry_point is in the
#    reinforcement list;
# 3. dots will benefit from being reinforced a few times (reaching 90us on P100)
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
# Define the TC for LENGTHS_COSINE_COHERENCE, use
###############################################################################
TC = tc.define(
    (LENGTHS_COSINE_COHERENCE
     .replace('<pos_dist>', str(args.pos_dist))
     .replace('<neg_dist>', str(args.neg_dist))),
    generate_options,
)

###############################################################################
# Run with implicit compilation and tuning
###############################################################################
# Input(N x C) random floats is partitioned into K buckets each of length L(K)
# We then sum within each bucket (with a positive-pair / negative-pair twist)
# This first impl uses the max bucket length and makes the computation dense
InputData = torch.randn(
    args.num_segs * args.seg_length, args.num_of_channels, device='cuda')
# Assume all segments of same length for now
Segments = torch.ones(args.num_segs, dtype=torch.int, device='cuda').fill_(args.seg_length)

Idx = TC.make_idx(Segments)
SegmentsMetaData = torch.ones((torch.max(Segments)[0],), dtype=torch.int, device='cuda')
Alpha = TC.make_alpha(Idx, SegmentsMetaData)
PosCount, NegCount, _1, _2 = TC.make_counts(Alpha, SegmentsMetaData)
Beta = TC.make_beta(Alpha, PosCount, NegCount, SegmentsMetaData)
NormData, Square = TC.normalize(InputData)
Dots = TC.dots(NormData, Idx, Beta)
Output, _ = TC.result(Beta, Dots)

R, Normed_DATA, Norm_of_Vector, POS_C, NEG_C = (
    reference(InputData.cpu().numpy(), Segments.cpu().numpy()))

###############################################################################
# Check
###############################################################################
tc.assert_almost_equal(
    PosCount.cpu(),
    torch.from_numpy(POS_C).float(),
    torch.from_numpy(POS_C).float(),
    precision=0)
tc.assert_almost_equal(
    NegCount.cpu(),
    torch.from_numpy(NEG_C).float(),
    torch.from_numpy(NEG_C).float(),
    precision=0)
tc.assert_almost_equal(
    Output.cpu(),
    torch.from_numpy(R),
    Dots.cpu(),
    Beta.cpu(),
    operations=SegmentsMetaData.size(0) * (SegmentsMetaData.size(0) + 1) // 2,
)

print('SUCCESS, maxdiff={}'.format((Output.cpu() - torch.from_numpy(R)).abs().max()))
