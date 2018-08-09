import numpy as np
import ipdb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tensor_comprehensions as tc
from visdom import Visdom

import utils

NB_EPOCHS = 2500
BATCH_SZ = 1

viz = Visdom(server="http://100.97.69.78")
win0 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))
win1 = viz.histogram(X=np.arange(NB_EPOCHS))

exptuner_config = utils.ExpTunerConfig()
exptuner_config.set_convolution_tc()

NB_HYPERPARAMS = utils.NB_HYPERPARAMS

def getRandom():
    opt_v = np.zeros(NB_HYPERPARAMS).astype(int)
    for i in range(opt_v.shape[0]):
        opt_v[i] = np.random.randint(exptuner_config.cat_sz[i])
    return opt_v

INTER_DISP = 20

running_reward = -0.5
tab_rewards=[]
tab_best=[]
liste_rew=[]
best_options = -1
cur_vect = getRandom()
best=utils.evalTime(cur_vect, exptuner_config)
best = -np.log(best)
nbTries=0
end=False
nbtt = np.sum(exptuner_config.cat_sz)
print("complete turn = " + str(nbtt)
while(nbTries < NB_EPOCHS):
    for i in range(NB_HYPERPARAMS):
        print("changing coordinate i=" + str(i))
        coor_best = best
        best_of_coor = cur_vect[i]
        for j in range(exptuner_config.cat_sz[i]):
            cur_vect[i]=j
            reward = utils.evalTime(cur_vect, exptuner_config, prune=2, curr_best=np.exp(-best))
            reward = -np.log(reward)
            if(reward > coor_best):
                coor_best = reward
                best_of_coor = j
            if(reward > best):
                best=reward
                utils.print_opt(cur_vect)
                print(-best)
            if(nbTries==0):
                running_reward = reward
            running_reward = running_reward * 0.99 + reward * 0.01

            liste_rew.append(-reward)
            tab_rewards.append(-running_reward)
            tab_best.append(-best)
            if nbTries % INTER_DISP == 0:
                #ipdb.set_trace()
                if(len(liste_rew) > 1):
                    viz.histogram(X=np.array(liste_rew).astype(int), win=win1)
                viz.line(X=np.column_stack((np.arange(nbTries+1), np.arange(nbTries+1))), Y=np.column_stack((np.array(tab_rewards), np.array(tab_best))), win=win0, opts=dict(legend=["Geometric run", "Best time"]))

            print(-running_reward)
            print(-best)
            nbTries+=1
            if(nbTries==NB_EPOCHS):
                end=True
                break
        cur_vect[i]=best_of_coor
        if(end):
            break

tab_best = np.array(tab_best)
np.save("taxicab.npy", tab_best)
print("Finally, best options are:")
utils.print_opt(cur_vect)
