import numpy as np
import ipdb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import ipdb
from itertools import count
from collections import namedtuple
from torch.distributions import Categorical
import time
import tensor_comprehensions as tc
from visdom import Visdom

NB_HYPERPARAMS, INIT_INPUT_SZ = 13, 0
NB_EPOCHS = 10000
BATCH_SZ = 8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0

viz = Visdom()
win0 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))

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
    #print(opt)
    #print(cat_val)
    opt = [cat_val[i][opt[i]] for i in range(NB_HYPERPARAMS)]
    opt = optionsFromVector(opt)
    warmup = 5
    iters  = 20
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
    options = tc.CudaMappingOptions("naive")
    #options.outerScheduleFusionStrategy("Max")
    #options.intraTileScheduleFusionStrategy("Min")
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

def computeCat(inp):
    global cat_sz, cat_val
    cat_sz = np.zeros(NB_HYPERPARAMS).astype(int)
    cat_val = []

    divs = getAllDivs(inp)

    cat_val.append([1,2,3])
    cat_val.append([1,2,3])
    cat_val.append([0,1])
    cat_val.append(divs + [0])
    cat_val.append([i for i in range(30)])
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

def getRandom():
    global cat_sz
    opt_v = np.zeros(NB_HYPERPARAMS).astype(int)
    for i in range(opt_v.shape[0]):
        opt_v[i] = np.random.randint(cat_sz[i])
    return opt_v

N, G, D, H, W = 5, 5, 5, 5, 5
I, gamma, beta = torch.randn(N, G, D, H, W).cuda(), torch.randn(G, D).cuda(), torch.randn(G, D).cuda()

init_input = (I, gamma, beta)
init_input_sz = np.array([N,G,D,H,W])

inp = init_input
computeCat(inp)

eps = np.finfo(np.float32).eps.item()

tc_prog = tc.define(code, name="group_normalization")

INTER_DISP = 20

running_reward = -0.5
tab_rewards=[]
tab_best=[]
best=-0.5
v_losses=[]
p_losses=[]
for i in range(NB_EPOCHS):
    rewards = []
    for j in range(BATCH_SZ):
        out = getRandom()
        reward = -evalTime(out.astype(int))
        reward=100*reward#+0.45
        rewards.append(reward)
    best = max(best, np.max(rewards))
    running_reward = running_reward * 0.99 + np.mean(rewards) * 0.01
    tab_rewards.append(-running_reward)
    tab_best.append(-best)
    if i % INTER_DISP == 0:
        viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), Y=np.column_stack((np.array(tab_rewards), np.array(tab_best))), win=win0, opts=dict(legend=["Geometric run", "Best time"]))
        #viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), Y=np.column_stack((np.array(v_losses), np.array(p_losses))), win=win1, opts=dict(legend=["Value loss", "Policy loss"]))
    print(-running_reward)
    print(-best)
np.save("randomsearch.npy", (-best))
