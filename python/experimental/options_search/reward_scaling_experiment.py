import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import ipdb
from itertools import count
from collections import namedtuple
from torch.distributions import Categorical
import tensor_comprehensions as tc
from visdom import Visdom
from collections import deque

import my_utils

NB_EPOCHS = 10000
BATCH_SZ = 16
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0
buff = deque()
MAXI_BUFF_SZ = 50

(tc_code, tc_name, inp, init_input_sz) = my_utils.get_convolution_example()

my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)

NB_HYPERPARAMS, INIT_INPUT_SZ = my_utils.NB_HYPERPARAMS, my_utils.INIT_INPUT_SZ

viz = Visdom()
win0 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))
win1 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

layer_sz = 32

class Predictor(nn.Module):
    def __init__(self, nb_inputs, nb_actions):
        super(Predictor, self).__init__()
        self.affine1 = nn.Linear(nb_inputs, layer_sz)
        self.affine15 = nn.Linear(layer_sz, layer_sz)
        self.affine2 = nn.Linear(layer_sz, nb_actions)
        self.affine3 = nn.Linear(layer_sz, 1)

        self.W = nn.Linear(nb_inputs, nb_inputs)

    def forward(self, x):
        #ipdb.set_trace()
        #x = F.softmax(self.W(x), dim=-1) * x
        tmp1 = F.relu(self.affine1(x))
        tmp1 = F.relu(self.affine15(tmp1))
        out_action = F.softmax(self.affine2(tmp1), dim=-1)
        out_value = self.affine3(tmp1)
        return out_action, out_value

class FullNetwork(nn.Module):
    def __init__(self, nb_hyperparams, init_input_sz):
        super(FullNetwork, self).__init__()
        self.nb_hyperparams = nb_hyperparams
        self.init_input_sz = init_input_sz
        self.nets = [Predictor(init_input_sz + i, int(my_utils.cat_sz[i])) for i in range(nb_hyperparams)]
        self.nets = nn.ModuleList(self.nets)

    def select_action(self, x, i, out_sz):
        geps = 0.1
        proba = np.random.rand()
        probs, state_value = self.nets[i](x)
        if(proba <= geps):
            probs = torch.FloatTensor([1./out_sz]*out_sz)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), state_value

    def forward(self, x):
        actions_prob = []
        values = []
        for i in range(self.nb_hyperparams):
            sym, action_prob, value = self.select_action(x, i, int(my_utils.cat_sz[i]))
            actions_prob.append(action_prob)
            values.append(value)
            x = torch.cat([x, torch.FloatTensor([sym])])
        return x[INIT_INPUT_SZ:], actions_prob, values

net = FullNetwork(NB_HYPERPARAMS, INIT_INPUT_SZ)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
eps = np.finfo(np.float32).eps.item()

def finish_episode(actions_probs, values, final_rewards):
    policy_losses = [[] for i in range(BATCH_SZ)]
    value_losses = [[] for i in range(BATCH_SZ)]
    final_rewards = torch.tensor(list(final_rewards))
    #final_rewards = (final_rewards - final_rewards.mean()) / (final_rewards.std() + eps)
    for batch_id in range(BATCH_SZ):
        for (log_prob, value) in zip(actions_probs[batch_id], values[batch_id]):
            reward = final_rewards[batch_id] - value.item()
            policy_losses[batch_id].append(-log_prob * reward)
            value_losses[batch_id].append(F.smooth_l1_loss(value, torch.tensor([final_rewards[batch_id]])))
    optimizer.zero_grad()
    vloss = torch.stack([torch.stack(value_losses[i]).sum() for i in range(BATCH_SZ)]).mean()
    ploss = torch.stack([torch.stack(policy_losses[i]).sum() for i in range(BATCH_SZ)]).mean()
    loss = ploss + vloss
    loss.backward(retain_graph=True)
    optimizer.step()
    return vloss.item(), ploss.item()

def add_to_buffer(actions_probs, values, reward):
    global buff
    #if(len(buff) > 0):
    #    min_reward = np.min(np.array(buff)[:,2])
    #    if(reward < 10*min_reward):
    #        return
    if len(buff) == MAXI_BUFF_SZ:
        buff.popleft()
    buff.append((actions_probs, values, reward))

def select_batch():
    #random.sample()
    batch = [buff[np.random.randint(len(buff))] for i in range(BATCH_SZ-1)]
    batch.append(buff[-1])
    batch=np.array(batch)
    return batch[:,0], batch[:,1], batch[:,2]

INTER_DISP = 20

running_reward = -0.5
tab_rewards=[]
tab_best=[]
best=-12
v_losses=[]
p_losses=[]
best_options = np.zeros(13).astype(int)
for i in range(NB_EPOCHS):
    rewards = []
    out_actions, out_probs, out_values = net(init_input_sz)
    reward = my_utils.evalTime(out_actions.numpy().astype(int), prune=-1, curr_best=np.exp(-best))
    #reward=100*reward
    reward = -((reward)/100)
    add_to_buffer(out_probs, out_values, reward)
    actions_probs, values, rewards = select_batch()
    for j in range(1):
        vloss, ploss = finish_episode(actions_probs, values, rewards)
    v_losses.append(vloss)
    p_losses.append(ploss)
    if(best < reward or i==0):
        best=reward
        best_options = out_actions.numpy().astype(int)
        my_utils.print_opt(best_options)
    if(i==0):
        running_reward = reward
    running_reward = running_reward * 0.99 + reward * 0.01
    tab_rewards.append(-(running_reward))
    tab_best.append(-best)
    if i % INTER_DISP == 0:
        viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), Y=np.column_stack((np.array(tab_rewards), np.array(tab_best))), win=win0, opts=dict(legend=["Geometric run", "Best time"]))
        viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), Y=np.column_stack((np.array(v_losses), np.array(p_losses))), win=win1, opts=dict(legend=["Value loss", "Policy loss"]))
    print(-running_reward)
    print(-best)

print("Finally, best options are:")
my_utils.print_opt(best_options)
