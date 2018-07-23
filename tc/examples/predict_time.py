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
import my_utils

(tc_code, tc_name, inp, init_input_sz) = my_utils.get_convolution_example(size_type="input", inp_sz_list=[8,2,28,28,8,1,1])

my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)
NB_HYPERPARAMS, INIT_INPUT_SZ = my_utils.NB_HYPERPARAMS, my_utils.INIT_INPUT_SZ

def createY(x):
    y = my_utils.evalTime(x)
    return y

def getRandom():
    opt_v = np.zeros(NB_HYPERPARAMS).astype(int)
    for i in range(opt_v.shape[0]):
        opt_v[i] = np.random.randint(my_utils.cat_sz[i])
    return opt_v

def makeDataset():
    sz = 500
    datasetX, datasetY = [], []
    for _ in range(sz):
        opt = getRandom()
        yi = createY(opt)
        datasetX.append(opt)
        datasetY.append(yi)
    #with Pool(sz) as p:
    #    datasetY = p.starmap(createY, datasetX)
    return np.array(datasetX), np.array(datasetY)

def learn():
    datasetX, datasetY = makeDataset()
    print(min(datasetY))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(datasetX, datasetY, test_size=0.2, random_state = 42)
    model1 = GradientBoostingRegressor(n_estimators=1000)
    model1.fit(Xtrain, Ytrain)
    pred0 = model1.predict(Xtrain)
    pred1 = model1.predict(Xtest)
    score0 = model1.score(Xtrain, Ytrain)
    score1 = model1.score(Xtest, Ytest)
    print(score0)
    print(score1)
    print(np.corrcoef(pred0, Ytrain)[0, 1]**2)
    print(np.corrcoef(pred1, Ytest)[0,1]**2)

learn()
