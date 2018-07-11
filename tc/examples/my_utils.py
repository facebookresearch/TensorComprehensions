import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tensor_comprehensions as tc
import numpy as np

NB_HYPERPARAMS, INIT_INPUT_SZ = 26, 7

def getrand(l):
    return np.random.choice(l).item()

def get_convolution_example():
    global INIT_INPUT_SZ
    INIT_INPUT_SZ = 7
    tc_name = "convolution"
    tc_code = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
            O(n, m, h, w) +=! I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
        }
    """

    N, C, H, W, O, kH, kW = \
        getrand([8, 16, 32, 64]), \
        getrand([2, 4, 8, 16]), \
        getrand([28, 56, 112]), \
        getrand([28, 56, 112]), \
        getrand([8, 16, 32]), \
        getrand([1, 2, 4]), \
        getrand([1, 2, 4])
    I, W1 = torch.randn(N, C, H, W, device='cuda'), torch.randn(O, C, kH, kW, device='cuda')
    init_input = (I, W1)
    init_input_sz = np.array([N,C,H,W,O, kH, kW])
    init_input_sz = torch.from_numpy(init_input_sz).float()

    return (tc_code, tc_name, init_input, init_input_sz)

def print_opt(options):
    print(options.tolist())

def set_tc(tc_code_arg, tc_name_arg):
    global tc_code, tc_name
    tc_code = tc_code_arg
    tc_name = tc_name_arg

def set_inp(inp_arg):
    global inp
    inp = inp_arg

def set_vars(tc_prog_arg, inp_arg, cat_val_arg, cat_sz_arg):
    global tc_prog, inp, cat_val, cat_sz
    tc_prog = tc_prog_arg
    inp = inp_arg
    cat_val = cat_val_arg
    cat_sz = cat_sz_arg

def catVec_to_optVec(catVec):
    global cat_val
    opt = [cat_val[i][catVec[i]] for i in range(NB_HYPERPARAMS)]
    opt[18] = min(opt[18], 1024//opt[17])
    opt[19] = min(opt[19], 1024//(opt[17] * opt[18]))
    return opt

def evalTime(opt, iters=50, warmup=10, naive=False, prune=-1, curr_best=-1):
    global tc_code, tc_name, inp, cat_val
    #print(opt)
    #print(cat_val)

    infty = 30000
    opt = catVec_to_optVec(opt)
    if naive:
        opt = tc.MappingOptions("naive")
    else:
        opt = optionsFromVector(opt)
    tc_prog = tc.compile(tc_code, tc_name, opt, *inp)

    try:
        first_ft = tc_prog.executor.profile_kernel(inp)
    except:
        return infty
    if(prune != -1 and first_ft > 100*curr_best):
        return first_ft
    for i in range(warmup):
        tc_prog.executor.profile_kernel(inp)

    first_t = tc_prog.executor.profile_kernel(inp)

    if(prune != -1 and first_t > prune*curr_best):
        return first_t

    liste_t_tc = []
    for i in range(iters):
        iter_time = tc_prog.executor.profile_kernel(inp)
        liste_t_tc.append(iter_time)
    mean_time = np.mean(liste_t_tc)
    return mean_time

def getRawVectorFromTcOpt(tc_opt):
    tr_dic = {"Max":0, "Preserve3Coincident":1, "Min":2}
    opt_vect = np.zeros(NB_HYPERPARAMS).astype(int)
    opt_vect[0] = tr_dic[tc_opt["outerScheduleFusionStrategy"]]
    opt_vect[1] = tr_dic[tc_opt["intraTileScheduleFusionStrategy"]]
    opt_vect[2] = tc_opt["fixParametersBeforeScheduling"]
    opt_vect[3] = len(tc_opt["tile"])
    opt_vect[4:4+opt_vect[3]] = tc_opt["tile"]
    opt_vect[10] = tc_opt["unroll"]
    #opt_vect[11] = tc_opt["tileImperfectlyNested"]
    opt_vect[11] = tc_opt["matchLibraryCalls"]
    opt_vect[12] = len(tc_opt["mapToBlocks"])
    opt_vect[13:13+opt_vect[12]] = tc_opt["mapToBlocks"]
    opt_vect[16] = len(tc_opt["mapToThreads"])
    opt_vect[17:17+opt_vect[16]] = tc_opt["mapToThreads"]
    opt_vect[20] = tc_opt["useSharedMemory"]
    opt_vect[21] = tc_opt["usePrivateMemory"]
    opt_vect[22] = tc_opt["unrollCopyShared"]
    #opt_vect[23] = tc_opt["maxSharedMemory"]
    opt_vect[24] = tc_opt["useReadOnlyCache"]
    opt_vect[25] = tc_opt["privateDepth"]
    return opt_vect

def optionsFromVector(vect):
    strat_str = ["Max", "Preserve3Coincident", "Min"]
    options = tc.MappingOptions("naive")
    options.outerScheduleFusionStrategy(strat_str[vect[0]])
    options.intraTileScheduleFusionStrategy(strat_str[vect[1]])
    options.fixParametersBeforeScheduling(vect[2])
    options.tile(list(vect[4:(4+vect[3])]))
    options.unroll(vect[10])
    options.matchLibraryCalls(vect[11])
    options.mapToBlocks(list(vect[13:13+vect[12]]))
    options.mapToThreads(list(vect[17:17+vect[16]])) #grid?
    options.useSharedMemory(vect[20])
    options.usePrivateMemory(vect[21])
    options.unrollCopyShared(vect[22])
    #options.maxSharedMemory(vect[23])
    options.useReadOnlyCache(vect[24])
    options.privateDepth(vect[25])
    return options

def computeDivs(sz):
    l = []
    for i in range(sz): #or 10?
        if(2**i > sz):
            break
        l.append((sz+2**i-1)//(2**i))
    return l

def getAllDivs(inp, maxp2=8):
    p2 = []
    pp=1
    for i in range(maxp2+1):
        p2.append(pp)
        pp*=2
    l = []
    #for sz in inp[0].shape[:1]:
    #    l+=computeDivs(sz)
    for elem in inp:
        for sz in elem.shape:
            l += computeDivs(sz)
    return list(set(l+p2))

def computeCat(inp_arg):
    global cat_sz, cat_val, inp
    inp = inp_arg
    cat_sz = np.zeros(NB_HYPERPARAMS).astype(int)
    cat_val = []

    divs = getAllDivs(inp)
    #divs2 = getAllDivs([np.array([tc.tclib.shared_memory_size()])])

    cat_val.append([0,1,2])
    cat_val.append([0,1,2])
    cat_val.append([0,1])
    cat_val.append([i+1 for i in range(6)])
    for i in range(6): #tiling
        cat_val.append(divs + [0])
    cat_val.append([2**i for i in range(8)])
    cat_val.append([0,1])
    cat_val.append([i+1 for i in range(3)])
    for i in range(3):
        cat_val.append(divs) #blocks #maximum 2^31-1 for the first value and 65535 for the second and third
    cat_val.append([i+1 for i in range(3)])
    for i in range(3):
        cat_val.append(divs) #threads #maximum 1024 for the first and second value, 32 for the third, product below 1024
    cat_val.append([0,1])
    cat_val.append([0,1])
    cat_val.append([0,1])
    cat_val.append([0]) #cat_val.append(divs2)
    cat_val.append([0,1])
    cat_val.append([i for i in range(6)]) #6 ou 7 ??

    for i in range(NB_HYPERPARAMS):
        cat_sz[i] = len(cat_val[i])
