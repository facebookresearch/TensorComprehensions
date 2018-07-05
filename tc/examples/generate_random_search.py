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

import my_utils

NB_HYPERPARAMS, INIT_INPUT_SZ = my_utils.NB_HYPERPARAMS, my_utils.INIT_INPUT_SZ
NB_EPOCHS = 10000
BATCH_SZ = 1
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0

viz = Visdom(server="http://100.97.69.78")
win0 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))

(tc_code, tc_name, inp, init_input_sz) = my_utils.get_convolution_example()

my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)

def getRandom():
    opt_v = np.zeros(NB_HYPERPARAMS).astype(int)
    for i in range(opt_v.shape[0]):
        opt_v[i] = np.random.randint(my_utils.cat_sz[i])
    return opt_v

eps = np.finfo(np.float32).eps.item()


INTER_DISP = 20

running_reward = -0.5
tab_rewards=[]
tab_best=[]
best=-12
v_losses=[]
p_losses=[]
for i in range(NB_EPOCHS):
    rewards = []
    for j in range(BATCH_SZ):
        out = getRandom()
        reward = my_utils.evalTime(out.astype(int), prune=2, curr_best=np.exp(-best))
        #reward=100*reward#+0.45
        reward = -np.log(reward)
        rewards.append(reward)
    if(best < np.max(rewards) or i==0):
        best = np.max(rewards) #max(best, np.max(rewards))
    if(i==0):
        running_reward = reward
    running_reward = running_reward * 0.99 + np.mean(rewards) * 0.01
    tab_rewards.append(-running_reward)
    tab_best.append(-best)
    if i % INTER_DISP == 0:
        viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), Y=np.column_stack((np.array(tab_rewards), np.array(tab_best))), win=win0, opts=dict(legend=["Geometric run", "Best time"]))
        #viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), Y=np.column_stack((np.array(v_losses), np.array(p_losses))), win=win1, opts=dict(legend=["Value loss", "Policy loss"]))
    print(-running_reward)
    print(-best)
tab_best = np.array(tab_best)
np.save("randomsearch.npy", tab_best)
