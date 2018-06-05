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

def evalTime(tc_prog, opt, inp):
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

def genOptionsVector(inp):
    opt_v = np.zeros(13).astype(int) ## + 0/1/2 + 0/1/2 (-0/-1)

    opt_v[0] = np.random.randint(1,3) #outer_schedule_options.fusion_strategy
    opt_v[1] = np.random.randint(1,3) #intra_tile_schedule_options.fusion_strategy
    opt_v[2] = np.random.randint(0,1) #fix_parameters_before_scheduling
    divs = getAllDivs(inp)
    opt_v[3] = np.random.choice(divs + [0]) #todo, "divisuers" + 0 tiling
    opt_v[4] = np.random.randint(0, 30) #unroll. 2^1, ..., 2^128
    opt_v[5] = np.random.randint(0,1) #match_library_calls
    opt_v[6] = np.random.choice(divs) #todo, "diviseurs" block
    opt_v[7] = np.random.choice(divs) #same grid
    opt_v[8] = np.random.randint(0,1) #use_shared_memory
    opt_v[9] = np.random.randint(0,1) #use_private_memory
    opt_v[10] = np.random.randint(0,1) #unroll_copy_shared what if usesharedmemory = False ??)
    opt_v[11] = 0 #np.random.randint(0, sqrt(memoire_gpu) #max_shared_memory, useless if usesharedmemory is   false
    opt_v[12] = np.random.randint(0,1) #usereadonlycache
    return opt_v

def createY(x, tc_prog, inp):
    y = evalTime(tc_prog, optionsFromVector(x), inp)
    return y

def makeDataset(tc_prog, inp):
    #def createY(x):
    #    y = evalTime(tc_prog, optionsFromVector(x), inp)
    #    return y

    sz = 500
    datasetX, datasetY = [], []
    for i in range(sz):
        opt = genOptionsVector(inp)
        yi = evalTime(tc_prog, optionsFromVector(opt), inp)
        datasetX.append(opt)
        datasetY.append(yi)
    #with Pool(sz) as p:
    #    datasetY = p.starmap(createY, zip(datasetX, repeat(tc_prog), repeat(inp)))
    return np.array(datasetX), np.array(datasetY)

def learn(tc_prog, inp):
    datasetX, datasetY = makeDataset(tc_prog, inp)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(datasetX, datasetY, test_size=0.2, random_state = 42)
    model1 = GradientBoostingRegressor(n_estimators=1000)
    model1.fit(Xtrain, Ytrain)
    pred0 = model1.predict(Xtrain)
    pred1 = model1.predict(Xtest)
    score0 = model1.score(Xtrain, Ytrain)
    score1 = model1.score(Xtest, Ytest)
    #print(score0)
    #print(score1)
    print(np.corrcoef(pred0, Ytrain)[0, 1]**2)
    print(np.corrcoef(pred1, Ytest)[0,1]**2)

gn = tc.define(code, name="group_normalization")

N, G, D, H, W = 5, 5, 5, 5, 5
I, gamma, beta = torch.randn(N, G, D, H, W).cuda(), torch.randn(G, D).cuda(), torch.randn(G, D).cuda()

#out = gn(I, gamma, beta)
#print(out)

inp = (I, gamma, beta)

learn(gn, inp)
