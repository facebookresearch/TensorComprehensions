import numpy as np
import ipdb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import ipdb
import tensor_comprehensions as tc
from visdom import Visdom

import my_utils

NB_EPOCHS = 10000
BATCH_SZ = 1

viz = Visdom()
win0 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))

(tc_code, tc_name, inp, init_input_sz) = my_utils.get_convolution_example()

my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)

NB_HYPERPARAMS, INIT_INPUT_SZ = my_utils.NB_HYPERPARAMS, my_utils.INIT_INPUT_SZ

def getRandom():
    opt_v = np.zeros(NB_HYPERPARAMS).astype(int)
    for i in range(opt_v.shape[0]):
        opt_v[i] = np.random.randint(my_utils.cat_sz[i])
    return opt_v


INTER_DISP = 20

running_reward = -0.5
tab_rewards=[]
tab_best=[]
best=-12
best_options = -1
for i in range(NB_EPOCHS):
    rewards = []
    opts=[]
    for j in range(BATCH_SZ):
        out = getRandom()
        reward = my_utils.evalTime(out.astype(int), prune=2, curr_best=np.exp(-best))
        reward = -np.log(reward)
        rewards.append(reward)
        opts.append(out.astype(int))
    if(best < np.max(rewards) or i==0):
        best = np.max(rewards)
        ind=np.argmax(rewards)
        best_options = opts[ind]
        my_utils.print_opt(best_options)
    if(i==0):
        running_reward = reward
    running_reward = running_reward * 0.99 + np.mean(rewards) * 0.01
    tab_rewards.append(-running_reward)
    tab_best.append(-best)
    if i % INTER_DISP == 0:
        viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), Y=np.column_stack((np.array(tab_rewards), np.array(tab_best))), win=win0, opts=dict(legend=["Geometric run", "Best time"]))
    print(-running_reward)
    print(-best)
tab_best = np.array(tab_best)
np.save("randomsearch.npy", tab_best)
print("Finally, best options are:")
my_utils.print_opt(best_options)
