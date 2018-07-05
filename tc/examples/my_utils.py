import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tensor_comprehensions as tc
import numpy as np

NB_HYPERPARAMS, INIT_INPUT_SZ = 13, 7
N, G, D, H, W = 5, 5, 5, 1, 1

def get_convolution_example():
    tc_name = "convolution"
    tc_code = """
        def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
            O(n, m, h, w) +=! I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
        }
    """

    N, C, H, W, O, kH, kW = 32, 4, 56, 56, 16, 1, 1
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

def evalTime(opt, iters=50, warmup=30, naive=False):
    global tc_code, tc_name, inp, cat_val
    #print(opt)
    #print(cat_val)
    opt = [cat_val[i][opt[i]] for i in range(NB_HYPERPARAMS)]
    if naive:
        opt = tc.MappingOptions("naive")
    else:
        opt = optionsFromVector(opt)
    tc_prog = tc.compile(tc_code, tc_name, opt, *inp)
    for i in range(warmup):
        tc_prog.executor.profile(inp)

    liste_t_tc = []
    now = time.clock()
    for i in range(iters):
        iter_time = tc_prog.executor.profile(inp)
        liste_t_tc.append(iter_time)
    mean_time = np.mean(liste_t_tc)
    return mean_time

def optionsFromVector(vect):
    strat_str = ["Max", "Preserve3Coincident", "Min"]
    options = tc.MappingOptions("naive")
    options.outerScheduleFusionStrategy(strat_str[vect[0]])
    options.intraTileScheduleFusionStrategy(strat_str[vect[1]])
    options.fixParametersBeforeScheduling(vect[2])
    options.tile([vect[3]]) #why list in doc?
    options.unroll(2**vect[4]) #128 is too big? trying 30
    options.matchLibraryCalls(vect[5])
    options.mapToBlocks([vect[6]])
    options.mapToThreads([vect[7]]) #grid?
    options.useSharedMemory(vect[8])
    options.usePrivateMemory(vect[9])
    options.unrollCopyShared(vect[10])
    #options.maxSharedMemory(vect[11]) #todo 40000 / 0 et divs
    options.useReadOnlyCache(vect[12])
    return options

def computeDivs(sz):
    l = []
    for i in range(sz): #or 10?
        l.append((sz+i)//(i+1))
    return l

def getAllDivs(inp, maxp2=8):
    p2 = []
    pp=1
    for i in range(maxp2+1):
        p2.append(pp)
        pp*=2
    l = []
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

    cat_val.append([0,1,2])
    cat_val.append([0,1,2])
    cat_val.append([0,1])
    cat_val.append(divs + [0])
    cat_val.append([i for i in range(10)])
    cat_val.append([0,1])
    cat_val.append(divs)
    cat_val.append(divs)
    cat_val.append([0,1])
    cat_val.append([0,1])
    cat_val.append([0,1])
    cat_val.append([0])
    cat_val.append([0,1])

    for i in range(13):
        cat_sz[i] = len(cat_val[i])
