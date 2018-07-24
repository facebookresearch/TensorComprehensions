import time
import torch
import tensor_comprehensions as tc
#import sklearn
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
#from sklearn.model_selection import train_test_split
#from tensor_comprehensions.mapping_options import Options
from multiprocessing import Pool
from itertools import repeat
import utils
#from tqdm import tqdm

exptuner_config = utils.ExpTunerConfig()
exptuner_config.set_convolution_tc()

NB_HYPERPARAMS = utils.NB_HYPERPARAMS

def createY(x):
    y = utils.evalTime(x, exptuner_config)
    return y

def getRandom():
    opt_v = np.zeros(NB_HYPERPARAMS).astype(int)
    for i in range(opt_v.shape[0]):
        opt_v[i] = np.random.randint(exptuner_config.cat_sz[i])
    return opt_v

def makeDataset():
    from tqdm import tqdm
    sz = 500
    datasetX, datasetY = [], []
    for _ in tqdm(range(sz)):
        opt = getRandom()
        yi = createY(opt)
        datasetX.append(opt)
        datasetY.append(yi)
    #with Pool(sz) as p:
    #    datasetY = p.starmap(createY, datasetX)
    return np.array(datasetX), np.array(datasetY)

def learn():
    #from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    datasetX, datasetY = makeDataset()
    print(min(datasetY))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(datasetX, datasetY, test_size=0.2, random_state = 42)
    model1 = GradientBoostingRegressor(n_estimators=1000)
    model1.fit(Xtrain, Ytrain)
    pred0 = model1.predict(Xtrain)
    pred1 = model1.predict(Xtest)
    print(np.corrcoef(pred0, Ytrain)[0, 1]**2)
    print(np.corrcoef(pred1, Ytest)[0,1]**2)

#learn()
