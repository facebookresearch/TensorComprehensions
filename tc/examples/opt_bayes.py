import time
import torch
import tensor_comprehensions as tc
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import train_test_split
#from tensor_comprehensions.mapping_options import Options
from multiprocessing import Pool
from itertools import repeat
import bayesopt as bo
from bayesoptmodule import BayesOptDiscrete

nb_cat = 0
cat_val = 0

code = """
def group_normalization(
    float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta)
    -> (O, mean, var)
{
    mean(n, g) +=! I(n, g, r_d, r_h, r_w)
     var(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)

    O(n, g, d, h, w) = gamma(g, d)
      * ( I(n, g, d, h, w) - mean(n, g) * 4 )
      * rsqrt( var(n, g) * 4
            - mean(n, g) * mean(n, g) * 4 * 4
            + 1e-5)
      + beta(g, d)
}
"""

def evalTime(opt):
    global tc_prog, inp, cat_val
    print(opt[0])
    #print(i)
    opt = np.array([cat_val[opt[i]] for i in range(13)])
    opt = optionsFromVector(opt)
    warmup = 10
    iters  = 50
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

def computeDivs(sz):
    l = []
    for i in range(sz): #or 10?
        l.append((sz+i)//(i+1))
    return l

def getAllDivs(inp, maxp2=31):
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

def optionsFromVector(vect):
    options = tc.CudaMappingOptions("naive")
    #options.outerScheduleFusionStrategy(vect[0], todo, todo)
    #options.intraTileFusionStrategy(vect[1], todo, todo)
    options.fixParametersBeforeScheduling(vect[2])
    #print(vect[3])
    options.tile([vect[3]]) #why list in doc?
    options.unroll(2**vect[4]) #128 is too big? trying 30
    options.matchLibraryCalls(vect[5])
    options.mapToBlocks([vect[6]])
    options.mapToThreads([vect[7]]) #grid?
    options.useSharedMemory(vect[8])
    options.usePrivateMemory(vect[9])
    options.unrollCopyShared(vect[10])
    #options.maxSharedMemory(vect[11]) #todo
    options.useReadOnlyCache(vect[12])
    return options

def computeCat(inp):
    global cat_sz, cat_val
    cat_sz = np.zeros(13).astype(int)
    cat_val = []
    #opt_v = np.zeros(13).astype(int) ## + 0/1/2 + 0/1/2 (-0/-1)

    divs = getAllDivs(inp)

    cat_val.append([1,2,3])
    cat_val.append([1,2,3])
    cat_val.append([0,1])
    cat_val.append(divs + [0])
    cat_val.append([i for i in range(31)])
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

gn = tc.define(code, name="group_normalization")

N, G, D, H, W = 5, 5, 5, 5, 5
I, gamma, beta = torch.randn(N, G, D, H, W).cuda(), torch.randn(G, D).cuda(), torch.randn(G, D).cuda()

#out = gn(I, gamma, beta)
#print(out)

inp = (I, gamma, beta)

print("coco salut")

computeCat(inp)

print("salut coco")

def bidule(opt):
    return 1

params = {}
#params['n_iterations'] = 50
#params['n_iter_relearn'] = 5
#params['n_init_samples'] = 2
#print(cat_sz)
#cat_sz = 2*cat_sz
y_out, x_out, error = bo.optimize_categorical(bidule, cat_sz, params)
