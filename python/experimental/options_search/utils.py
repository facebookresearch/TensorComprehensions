import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tensor_comprehensions as tc
import numpy as np
from enum import IntEnum

NB_HYPERPARAMS = 26

class ExpTunerConfig:
    def __init__(self, use_max_shared_memory=0):
        self.INIT_INPUT_SZ = -1
        self.USE_MAX_SHARED_MEMORY = use_max_shared_memory
        self.tc_code = "" 
        self.tc_name = ""
        self.inp = -1
        self.cat_val = -1
        self.cat_sz = -1

    def set_convolution_tc(self, size_type="default", inp_sz_list=[], use_max_shared_memory=False):
        self.INIT_INPUT_SZ = 7
        self.tc_name = "convolution"
        self.tc_code = """
                def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
                O(n, m, h, w) +=! I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
                }
        """

        if(size_type=="input"):
                N, C, H, W, O, kH, kW = tuple(inp_sz_list)
        elif(size_type=="default"):
                N, C, H, W, O, kH, kW = 16, 4, 56, 56, 16, 1, 1 #8, 2, 28, 28, 8, 1, 1
        elif(size_type=="random"):
                N, C, H, W, O, kH, kW = \
                getrand([8, 16, 32, 64]), \
                getrand([2, 4, 8, 16]), \
                getrand([28, 56, 112]), \
                getrand([28, 56, 112]), \
                getrand([8, 16, 32]), \
                getrand([1, 2, 4]), \
                getrand([1, 2, 4])
        else:
                print("Unknown size type")
                exit()
        I, W1 = torch.randn(N, C, H, W, device='cuda'), torch.randn(O, C, kH, kW, device='cuda')
        self.inp = (I, W1)
        self.init_input_sz = np.array([N,C,H,W,O, kH, kW])
        print(self.init_input_sz)
        self.init_input_sz = torch.from_numpy(self.init_input_sz).float()

        self.computeCat()

    def computeCat(self):
        inp = self.inp
        self.cat_sz = np.zeros(NB_HYPERPARAMS).astype(int)
        self.cat_val = [[] for _ in range(NB_HYPERPARAMS)]

        divs = getAllDivs(inp)
        if(self.USE_MAX_SHARED_MEMORY):
            divs2 = getAllDivs([np.array([tc.tclib.shared_memory_size()])])

        self.cat_val[MappingOptionsIdx.outerScheduleFusionStrategy] = \
                [0,1,2]
        self.cat_val[MappingOptionsIdx.intraTileScheduleFusionStrategy] = \
                [0,1,2]
        self.cat_val[MappingOptionsIdx.fixParametersBeforeScheduling] = \
                [0,1]
        self.cat_val[MappingOptionsIdx.nTiledDims] = \
                [i+1 for i in range(6)]
        for i in range(6): #tiling
                self.cat_val[MappingOptionsIdx.tiling1 + i] = \
                        divs + [0]
        self.cat_val[MappingOptionsIdx.unroll] = \
                [2**i for i in range(8)]
        self.cat_val[MappingOptionsIdx.matchLibraryCalls] = \
                [0,1]
        self.cat_val[MappingOptionsIdx.nMappedToBlocksDims] = \
                [i+1 for i in range(3)]
        for i in range(3): #mapping to blocks
                self.cat_val[MappingOptionsIdx.mappingToBlocks1 + i] = \
                        divs
        self.cat_val[MappingOptionsIdx.nMappedToThreadsDims] = \
                [i+1 for i in range(3)]
        for i in range(3): #mapping to threads
                self.cat_val[MappingOptionsIdx.mappingToThreads1 + i] = \
                        divs
        self.cat_val[MappingOptionsIdx.useSharedMemory] = \
                [0,1]
        self.cat_val[MappingOptionsIdx.usePrivateMemory] = \
                [0,1]
        self.cat_val[MappingOptionsIdx.unrollCopyShared] = \
                [0,1]
        self.cat_val[MappingOptionsIdx.maxSharedMemory] = \
                divs2 if USE_MAX_SHARED_MEMORY else [0]
        self.cat_val[MappingOptionsIdx.useReadOnlyCache] = \
                [0,1]
        self.cat_val[MappingOptionsIdx.privateDepth] = \
                [i for i in range(6)]

        for i in range(NB_HYPERPARAMS):
                self.cat_sz[i] = len(self.cat_val[i])

    def catVec_to_optVec(self, catVec):
        opt = [self.cat_val[i][catVec[i]] for i in range(NB_HYPERPARAMS)]
        return opt


class MappingOptionsIdx(IntEnum):
    outerScheduleFusionStrategy   = 0
    intraScheduleFusionStrategy   = 1
    fixParametersBeforeScheduling = 2
    nTiledDims                    = 3
    tiling1                       = 4
    tiling2                       = 5
    tiling3                       = 6
    tiling4                       = 7
    tiling5                       = 8
    tiling6                       = 9
    unroll                        = 10
    matchLibraryCalls             = 11
    nMappedToBlocksDims           = 12
    mappingToBlocks1              = 13
    mappingToBlocks2              = 14
    mappingToBlocks3              = 15
    nMappedToThreadsDims          = 16
    mappingToThreads1             = 17
    mappingToThreads2             = 18
    mappingToThreads3             = 19
    useSharedMemory               = 20
    usePrivateMemory              = 21
    unrollCopyShared              = 22
    maxSharedMemory               = 23
    useReadOnlyCache              = 24
    privateDepth                  = 25

def get_rand(l):
    return np.random.choice(l).item()

def print_opt(options):
    print(options.tolist())

def evalTime(opt, exptuner_config, iters=50, warmup=10, estimator="mean", prune=-1, curr_best=-1):
    tc_code, tc_name, inp = \
        exptuner_config.tc_code, exptuner_config.tc_name, exptuner_config.inp
    infty = 30000
    opt = exptuner_config.catVec_to_optVec(opt)
    opt = optionsFromVector(opt)
    try:
        tc_prog = tc.compile(tc_code, tc_name, opt, *inp)
        first_ft = tc_prog.executor.profile_kernel(inp)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        return infty
    if(prune != -1 and first_ft > 100*curr_best):
        return first_ft
    for _ in range(warmup-1):
        tc_prog.executor.profile_kernel(inp)

    first_t = tc_prog.executor.profile_kernel(inp)

    if(prune != -1 and first_t > prune*curr_best):
        return first_t

    tc_time_list = [first_t]
    for i in range(iters-1):
        iter_time = tc_prog.executor.profile_kernel(inp)
        tc_time_list.append(iter_time)
    if(estimator == "mean"):
        mean_time = np.mean(tc_time_list)
        return mean_time
    elif(estimator == "median"):
        median_time = np.median(tc_time_list)
        return median_time
    elif(estimator == "p25"):
        p25_time = np.percentile(tc_time_list, 25)
        return p25_time
    print("Unknown estimator")
    return infty

def getRawVectorFromTcOpt(tc_opt):
    tr_dic = {"Max":0, "Preserve3Coincident":1, "Min":2}
    opt_vect = np.zeros(NB_HYPERPARAMS).astype(int)
    opt_vect[MappingOptionsIdx.outerScheduleFusionStrategy] = \
            tr_dic[tc_opt["outerScheduleFusionStrategy"]]
    opt_vect[MappingOptionsIdx.intraTileScheduleFusionStrategy] = \
            tr_dic[tc_opt["intraTileScheduleFusionStrategy"]]
    opt_vect[MappingOptionsIdx.fixParametersBeforeScheduling] = \
            tc_opt["fixParametersBeforeScheduling"]
    opt_vect[MappingOptionsIdx.nTiledDims] = \
            len(tc_opt["tile"])
    assert opt_vect[MappingOptionsIdx.nTiledDims] < 7, "Too many tilings"
    opt_vect[
            MappingOptionsIdx.tiling1 : MappingOptionsIdx.tiling1 + opt_vect[MappingOptionsIdx.nTiledDims]] = \
                    tc_opt["tile"]
    opt_vect[MappingOptionsIdx.unroll] = \
            tc_opt["unroll"]
    #opt_vect[MappingOptionsIdx.tileImperfectlyNested] = \
    #        tc_opt["tileImperfectlyNested"] #todo: pybind
    opt_vect[MappingOptionsIdx.matchLibraryCalls] = \
            tc_opt["matchLibraryCalls"]
    opt_vect[MappingOptionsIdx.nMappedToBlocksDims] = \
            len(tc_opt["mapToBlocks"])
    opt_vect[
            MappingOptionsIdx.mappingToBlocks1 : MappingOptionsIdx.mappingToBlocks1 + opt_vect[MappingOptionsIdx.nMappedToBlocksDims]] = \
                    tc_opt["mapToBlocks"]
    opt_vect[MappingOptionsIdx.nMappedToThreadsDims] = \
            len(tc_opt["mapToThreads"])
    opt_vect[
            MappingOptionsIdx.mappingToThreads1 : MappingOptionsIdx.mappingToThreads1 + opt_vect[MappingOptionsIdx.nMappedToThreadsDims]] = \
                    tc_opt["mapToThreads"]
    opt_vect[MappingOptionsIdx.useSharedMemory] = \
            tc_opt["useSharedMemory"]
    opt_vect[MappingOptionsIdx.usePrivateMemory] = \
            tc_opt["usePrivateMemory"]
    opt_vect[MappingOptionsIdx.unrollCopyShared] = \
            tc_opt["unrollCopyShared"]
    if(USE_MAX_SHARED_MEMORY and "maxSharedMemory" in tc_opt):
        opt_vect[MappingOptionsIdx.maxSharedMemory] = \
                tc_opt["maxSharedMemory"]
    opt_vect[MappingOptionsIdx.useReadOnlyCache] = \
            tc_opt["useReadOnlyCache"]
    opt_vect[MappingOptionsIdx.privateDepth] = \
            tc_opt["privateDepth"]
    return opt_vect

def optionsFromVector(vect):
    strat_str = ["Max", "Preserve3Coincident", "Min"]
    options = tc.MappingOptions("naive")
    options.outerScheduleFusionStrategy(
            strat_str[vect[
                MappingOptionsIdx.outerScheduleFusionStrategy]])
    options.intraTileScheduleFusionStrategy(
            strat_str[vect[
                MappingOptionsIdx.intraTileScheduleFusionStrategy]])
    options.fixParametersBeforeScheduling(
            vect[MappingOptionsIdx.fixParametersBeforeScheduling])
    options.tile(
            list(vect[
                MappingOptionsIdx.tiling1 : MappingOptionsIdx.tiling1 + vect[MappingOptionsIdx.nTiledDims]]))
    options.unroll(
            vect[MappingOptionsIdx.unroll])
    options.matchLibraryCalls(
            vect[MappingOptionsIdx.matchLibraryCalls])
    options.mapToBlocks(
            list(vect[
                MappingOptionsIdx.mappingToBlocks1 : MappingOptionsIdx.mappingToBlocks1 + vect[MappingOptionsIdx.nMappedToBlocksDims]]))
    options.mapToThreads(
            list(vect[
                MappingOptionsIdx.mappingToThreads1 : MappingOptionsIdx.mappingToThreads1 + vect[MappingOptionsIdx.nMappedToThreadsDims]]))
    options.useSharedMemory(
            vect[MappingOptionsIdx.useSharedMemory])
    options.usePrivateMemory(
            vect[MappingOptionsIdx.usePrivateMemory])
    options.unrollCopyShared(
            vect[MappingOptionsIdx.unrollCopyShared])
    if(USE_MAX_SHARED_MEMORY):
        options.maxSharedMemory(
                vect[MappingOptionsIdx.maxSharedMemory])
    options.useReadOnlyCache(
            vect[MappingOptionsIdx.useReadOnlyCache])
    options.privateDepth(
            vect[MappingOptionsIdx.privateDepth])
    return options

def computeDivs(sz):
    l = []
    for i in range(sz):
        if(2 ** i > sz):
            break
        l.append((sz + 2 ** i - 1) // (2 ** i))
    return l

def getAllDivs(inp, maxp2=8):
    p2 = [2**i for i in range(maxp2 + 1)]
    l = []
    for elem in inp:
        for sz in elem.shape:
            l += computeDivs(sz)
    divs_list = list(set(l + p2))
    return sorted(divs_list)
