import utils
from scipy.optimize import minimize
import numpy as np

NB_EPOCHS = 1000
BATCH_SZ = 1

viz = Visdom(server="100.97.69.78")
win0 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))

exptuner_config = utils.ExpTunerConfig()
exptuner_config.set_convolution_tc()

NB_HYPERPARAMS = utils.NB_HYPERPARAMS

best=99999999
nb_iters=0
rws=[]
best_rewards=[]
running_reward=1.

def getRandom():
    opt_v = np.zeros(NB_HYPERPARAMS).astype(int)
    for i in range(opt_v.shape[0]):
        opt_v[i] = np.random.randint(exptuner_config.cat_sz[i])
    return opt_v

def my_fun(x):
    global best, nb_iters, best_rewards, rws, running_reward
    nb_iters+=1
    y = x.astype(int)
    for i in range(y.shape[0]):
        if(y[i] < 0):
            y[i] = 0
        if(y[i] >= exptuner_config.cat_sz[i]):
            y[i] = exptuner_config.cat_sz[i]-1
    time_t = utils.evalTime(y, exptuner_config)
    reward = -np.log(time_t)
    if(nb_iters==1):
        running_reward = reward
    running_reward = running_reward * 0.99 + reward * 0.01
    if(time_t < best):
        best=time_t
        print("iter " + str(nb_iters))
        utils.print_opt(y)
        print(best)
    best_rewards.append(np.log(best))
    rws.append(-running_reward)
    if nb_iters % INTER_DISP == 0:
        viz.line(X=np.column_stack((np.arange(nb_iters)), np.arange(nb_iters))), \
        Y=np.column_stack((np.array(rws), np.array(best_rewards))), \
        win=win0, opts=dict(legend=["Geometric run", "Best time"]))
    return time_t

x0 = getRandom()
res=minimize(my_fun, x0, method="Powell")
print(res)