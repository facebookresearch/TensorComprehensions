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

import numpy as np
import torch

#
## Example submitted by @mdouze, originally related to uint8 type support
#

debug = False
tc.logtostderr(debug)
tc.debug_tc_mapper(debug)

N = 1000
M = 32

codes = np.random.randint(1<<32, size=(N, M // 4)).astype('uint32')
codes = codes.view('uint8')
luts = np.random.randn(M, 256).astype('float32')

codes_t = torch.from_numpy(codes).cuda()
luts_t = torch.from_numpy(luts).cuda()

lang = """
# mindis as a single kernel will require grid synchronization to run efficiently
def mindis(float(M, 256) L, uint8(N, M) Codes) -> (S, v, min_idx) {
          S(n) +=! L(r_m, int32(Codes(n, r_m)))
          v  min=! S(r_n)
    min_idx  min=! (S(r_n) == v) ? r_n : N
}

# Even when splitting in 3 kernels, global device reduction will be needed to
# run efficiently
# don't try to run it with large sizes for now
def reduce_codes(float(M, 256) L, uint8(N, M) Codes) -> (S) {
    S(n) +=! L(r_m, int32(Codes(n, r_m)))
}
def min_2d(float(N) S) -> (v) {
    v min=! S(r_n)
}
def argmin_2d(float(N) S, float v) -> (min_idx) {
    min_idx min=! (S(r_n) == v) ? r_n : N
}
"""

mindis = tc.define(
    lang, entry_point="mindis", fallback=tc.MappingOptions('naive'))
S, v, min_idx = mindis(luts_t, codes_t)
print("minval: {} minidx: {}".format(v, min_idx))

reduce_codes = tc.define(
    lang, entry_point="reduce_codes", fallback=tc.MappingOptions('naive'))
min_2d = tc.define(
    lang, entry_point="min_2d", fallback=tc.MappingOptions('naive'))
argmin_2d = tc.define(
    lang, entry_point="argmin_2d", fallback=tc.MappingOptions('naive'))

S, = reduce_codes(luts_t, codes_t)
v, = min_2d(S)
min_idx = argmin_2d(S, v)

print("minval: {} minidx: {}".format(v, min_idx))

################################################################################
# Each reduction is probably easier to optimize with a 2-staged TC where we
# artifically increase parallelism and finish the reduction in a second kernel.
# Properly choosing D such that N = D * (N / D) should result in a good version
# with 5 kernels total.
################################################################################
N = 10 ** 5 # bump to 10**7 when ready for primetime
D = 1000
assert N % D == 0, "D={} must divide N={}".format(D, N)
M = 32

lang = """
def reduce_codes(float(M, 256) L, uint8(N, M) Codes) -> (S) {
    S(n) +=! L(r_m, int32(Codes(n, r_m)))
}
def min_2d(float(D, NBYD) S) -> (V) {
    V(d) min=! S(d, r_nbyd)
}
def min_1d(float(D) V) -> (v) {
    v min=! V(r_d)
}
def argmin_2d(float(D, NBYD) S, float v) -> (MinIdx) {
    MinIdx(d) min=!
        (S(d, r_nbyd) == v) ? d * NBYD + r_nbyd : NBYD * D
}
def argmin_1d(float(N) S, int32(D) MinIdx) -> (min_idx) {
    min_idx min=! (MinIdx(r_d) < N) ? r_d : N
}
"""

codes = np.random.randint(1<<32, size=(N, M // 4)).astype('uint32')
codes = codes.view('uint8')
luts = np.random.randn(M, 256).astype('float32')

codes_t = torch.from_numpy(codes).cuda()
luts_t = torch.from_numpy(luts).cuda()

reduce_codes = tc.define(
    lang, entry_point="reduce_codes", fallback=tc.MappingOptions('naive'))
min_2d = tc.define(
    lang, entry_point="min_2d", fallback=tc.MappingOptions('naive'))
min_1d = tc.define(
    lang, entry_point="min_1d", fallback=tc.MappingOptions('naive'))
argmin_2d = tc.define(
    lang, entry_point="argmin_2d", fallback=tc.MappingOptions('naive'))
argmin_1d = tc.define(
    lang, entry_point="argmin_1d", fallback=tc.MappingOptions('naive'))

S, = reduce_codes(luts_t, codes_t)
V, = min_2d(S.view((D, N / D)))
v, = min_1d(V)
MinIdx, = argmin_2d(S.view((D, N / D)), v)
min_idx, = argmin_1d(S, MinIdx)
print("minval: {} minidx: {}".format(v, min_idx))

################################################################################
# Longer form version has an extra k dimension we could use for parallelism
# Unfortunately is it a small dimension (16) so it won't saturate Pascal/Volta.
# So we may want to split in 5 to run efficiently.
################################################################################
N = 10 ** 5 # bump to 10**7 when ready for primetime
D = 1000
assert N % D == 0, "D={} must divide N={}".format(D, N)
M = 32
K = 16
codes = np.random.randint(1<<32, size=(N, M // 4)).astype('uint32')
codes = codes.view('uint8')
luts = np.random.randn(K, M, 256).astype('float32')

codes_t = torch.from_numpy(codes).cuda()
luts_t = torch.from_numpy(luts).cuda()

lang = """
def mindis(float(K, M, 256) L, uint8(N, M) Codes) -> (S, V, MinIdx) {
         S(k, n)   +=!  L(k, r_m, int32(Codes(n, r_m)))
         V(k)    min=!  S(k, r_n)
    MinIdx(k)    min=! (S(k, r_n) == V(k)) ? r_n : N
}
"""

mindis = tc.define(
    lang, entry_point="mindis", fallback=tc.MappingOptions('naive'))
S, V, MinIdx = mindis(luts_t, codes_t)
print("minvals: {}\nminidxs: {}".format(V, MinIdx))

lang = """
def reduce_codes(float(K, M, 256) L, uint8(N, M) Codes) -> (S) {
    S(k, n) +=! L(k, r_m, int32(Codes(n, r_m)))
}
def min_2d(float(K, D, NBYD) S) -> (V2) {
    V2(k, d) min=! S(k, d, r_nbyd)
}
def min_1d(float(K, D) V2) -> (V) {
    V(k) min=! V2(k, r_d)
}
def argmin_2d(float(K, D, NBYD) S, float(K) V) -> (MinIdx2) {
    MinIdx2(k, d) min=!
        (S(k, d, r_nbyd) == V(k)) ? d * NBYD + r_nbyd : NBYD * D
}
def argmin_1d(float(K, N) S, int32(K, D) MinIdx2) -> (MinIdx) {
    MinIdx(k) min=! (MinIdx2(k, r_d) < N) ? r_d : N
}
"""

reduce_codes = tc.define(
    lang, entry_point="reduce_codes", fallback=tc.MappingOptions('naive'))
min_2d = tc.define(
    lang, entry_point="min_2d", fallback=tc.MappingOptions('naive'))
min_1d = tc.define(
    lang, entry_point="min_1d", fallback=tc.MappingOptions('naive'))
argmin_2d = tc.define(
    lang, entry_point="argmin_2d", fallback=tc.MappingOptions('naive'))
argmin_1d = tc.define(
    lang, entry_point="argmin_1d", fallback=tc.MappingOptions('naive'))

S, = reduce_codes(luts_t, codes_t)
V2, = min_2d(S.view((K, D, N / D)))
V, = min_1d(V2)
MinIdx2, = argmin_2d(S.view((K, D, N / D)), V)
MinIdx, = argmin_1d(S, MinIdx2)
print("minval: {} minidx: {}".format(V, MinIdx))
