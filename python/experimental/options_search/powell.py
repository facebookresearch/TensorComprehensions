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

def my_fun(x):
    global best
    y = x.astype(int)
    time_t = utils.evalTime(y, exptuner_config)
    if(time_t < best):
        best=time_t
        utils.print_opt(y)
        print(best)

def printer(x):
    pass

x0 = np.zeros(NB_HYPERPARAMS)#.astype(int)
res=minimize(my_fun, x0, method="Powell", callback=printer)
print(res)