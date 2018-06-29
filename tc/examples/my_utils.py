import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tensor_comprehensions as tc
import numpy as np

NB_HYPERPARAMS, INIT_INPUT_SZ = 13, 7
N, G, D, H, W = 10, 10, 10, 10, 10

def print_opt(options):
    print(options.tolist())

def set_tcprog(tc_prog_arg):
    global tc_prog
    tc_prog = tc_prog_arg

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
    global tc_prog, inp, cat_val
    #print(opt)
    #print(cat_val)
    opt = [cat_val[i][opt[i]] for i in range(NB_HYPERPARAMS)]
    if naive:
        opt = tc.CudaMappingOptions("naive")
    else:
        opt = optionsFromVector(opt)
    #warmup = 5
    #iters  = 20
    for i in range(warmup):
        tc_prog(*inp, options=opt)
        torch.cuda.synchronize()

    liste_t_tc = []
    now = time.clock()
    for i in range(iters):
        before = time.clock()
        tc_prog(*inp, options=opt)
        #tcwavenet(Data)
        torch.cuda.synchronize()
        after = time.clock()
        liste_t_tc.append(after - before)
        torch.cuda.synchronize()
    total_time = (time.clock() - now)
    mean_time = total_time / iters
    return mean_time

def optionsFromVector(vect):
    strat_str = ["Max", "Preserve3Coincident", "Min"]
    options = tc.CudaMappingOptions("naive")
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
