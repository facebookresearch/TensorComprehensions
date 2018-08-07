import utils
from scipy.optimize import minimize
import numpy as np

NB_EPOCHS = 1000
BATCH_SZ = 1

#viz = Visdom()
#win0 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))

exptuner_config = utils.ExpTunerConfig()
exptuner_config.set_convolution_tc()

NB_HYPERPARAMS = utils.NB_HYPERPARAMS

best=99999999
nb_iters=0

def getRandom():
    opt_v = np.zeros(NB_HYPERPARAMS).astype(int)
    for i in range(opt_v.shape[0]):
        opt_v[i] = np.random.randint(exptuner_config.cat_sz[i])
    return opt_v

def my_fun(x):
    global best
    y = x.astype(int)
    for i in range(y.shape[0]):
        if(y[i] < 0):
            y[i] = 0
        if(y[i] >= exptuner_config.cat_sz[i]):
            y[i] = exptuner_config.cat_sz[i]-1
    time_t = utils.evalTime(y, exptuner_config)
    if(time_t < best):
        best=time_t
        print("iter " + str(nb_iters))
        utils.print_opt(y)
        print(best)
    return time_t

def counter(x):
    global nb_iters
    nb_iters+=1

x0 = getRandom()
res=minimize(my_fun, x0, method="Powell", callback=counter)
print(res)