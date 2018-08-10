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
from heapq import heappush, heappop
import pickle

import utils

NB_EPOCHS = 1000
BATCH_SZ = 16
buff = deque()
MAXI_BUFF_SZ = 50

exptuner_config = utils.ExpTunerConfig()
exptuner_config.set_convolution_tc(size_type="input", inp_sz_list=[8, 2, 28, 28, 8, 1, 1])

NB_HYPERPARAMS = utils.NB_HYPERPARAMS
INIT_INPUT_SZ = exptuner_config.INIT_INPUT_SZ
init_input_sz = exptuner_config.init_input_sz

print(NB_HYPERPARAMS)
print(init_input_sz)
print(INIT_INPUT_SZ)

viz = Visdom(server="http://100.97.69.78", port=8098)
win0 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))
win1 = viz.line(X=np.arange(NB_EPOCHS), Y=np.random.rand(NB_EPOCHS))
win2 = viz.histogram(X=np.arange(NB_EPOCHS))


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

layer_sz = 32

def save_model(network, filename):
    big_liste=[]
    for i in range(len(network.nets)):
        aff1 = network.nets[i].affine1.weight.data.numpy()
        aff2 = network.nets[i].affine2.weight.data.numpy()
        aff3 = network.nets[i].affine3.weight.data.numpy()
        c=[aff1,aff2,aff3]
        big_liste.append(c)
    with open(filename, 'wb') as handle:
        pickle.dump(big_liste, handle)

def load_model():
    with open('model_bigliste.pickle', 'rb') as handle:
        a = pickle.load(handle)

    net = FullNetwork(NB_HYPERPARAMS, INIT_INPUT_SZ)
    for i in range(len(net.nets)):
        net.nets[i].affine1.weight.data = torch.from_numpy(a[i][0])
        net.nets[i].affine2.weight.data = torch.from_numpy(a[i][1])
        net.nets[i].affine3.weight.data = torch.from_numpy(a[i][2])

    ipdb.set_trace()
    return net

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
        #x = F.softmax(self.W(x), dim=-1) * x #attention mecanism
        tmp1 = F.relu(self.affine1(x))
        #tmp1 = F.relu(self.affine15(tmp1))
        out_action = F.softmax(self.affine2(tmp1), dim=-1)
        out_value = self.affine3(tmp1)
        return out_action, out_value

class FullNetwork(nn.Module):
    def __init__(self, nb_hyperparams, init_input_sz):
        super(FullNetwork, self).__init__()
        self.nb_hyperparams = nb_hyperparams
        self.init_input_sz = init_input_sz
        self.nets = [Predictor(init_input_sz + i, int(exptuner_config.cat_sz[i])) for i in range(nb_hyperparams)]
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
            sym, action_prob, value = self.select_action(x, i, int(exptuner_config.cat_sz[i]))
            actions_prob.append(action_prob)
            values.append(value)
            x = torch.cat([x, torch.FloatTensor([sym])])
        return x[INIT_INPUT_SZ:], actions_prob, values

net = load_model()#FullNetwork(NB_HYPERPARAMS, INIT_INPUT_SZ)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
eps = np.finfo(np.float32).eps.item()

def finish_episode(actions_probs, values, final_rewards):
    policy_losses = [[] for i in range(BATCH_SZ)]
    value_losses = [[] for i in range(BATCH_SZ)]
    final_rewards = torch.tensor(list(final_rewards))
    final_rewards = (final_rewards - final_rewards.mean()) / (final_rewards.std() + eps)
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
    #for f in net.parameters():
    #    print("grad is")
    #    print(f.grad)
    optimizer.step()
    return vloss.item(), ploss.item()

def add_to_buffer(actions_probs, values, reward):
    global buff
    #if(len(buff) > 0):
    #    min_reward = np.min(np.array(buff)[:,2])
    #    if(reward < 10*min_reward):
    #        return
    if len(buff) == MAXI_BUFF_SZ:
        #heappop(buff)
        buff.popleft()
    #heappush(buff, (reward, actions_probs, values))
    buff.append((reward, actions_probs, values))

def select_batch():
    #random.sample()
    batch = [buff[np.random.randint(len(buff))] for i in range(BATCH_SZ)]
    #batch.append(buff[-1])
    batch=np.array(batch)
    return batch[:,1], batch[:,2], batch[:,0]

def get_best_buff():
    return np.max(np.array(buff)[:,0])

INTER_DISP = 20

running_reward = -0.5
tab_rewards=[]
tab_best=[]
best=-12
v_losses=[]
p_losses=[]
liste_rew=[]
best_options = np.zeros(NB_HYPERPARAMS).astype(int)
for i in range(NB_EPOCHS):
    rewards = []
    out_actions, out_probs, out_values = net(init_input_sz)
    #utils.print_opt(out_actions.numpy().astype(int))
    reward = utils.evalTime(out_actions.numpy().astype(int), exptuner_config, prune=-1, curr_best=np.exp(-best))
    #reward=100*reward
    #reward = -((reward)/1000)
    reward = -np.log(reward)
    add_to_buffer(out_probs, out_values, reward)
    best_in_buffer = get_best_buff()
    if(i >= 20):
        actions_probs, values, rewards = select_batch()
        for j in range(1):
            vloss, ploss = finish_episode(actions_probs, values, rewards)
        v_losses.append(vloss)
        p_losses.append(ploss)
    if(best < reward or i==0):
        best=reward
        best_options = out_actions.numpy().astype(int)
        utils.print_opt(best_options)
    if(i==0):
        running_reward = reward
    running_reward = running_reward * 0.99 + reward * 0.01
    tab_rewards.append(-(running_reward))
    tab_best.append(-best)
    liste_rew.append(-reward)
    if i % INTER_DISP == 0:
        if(len(liste_rew) > 1):
            viz.histogram(X=np.array(liste_rew).astype(int), win=win2)
        viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), Y=np.column_stack((np.array(tab_rewards), np.array(tab_best))), win=win0, opts=dict(legend=["Geometric run", "Best time"]))
        if(len(v_losses) > 0):
            viz.line(X=np.column_stack((np.arange(len(v_losses)), np.arange(len(v_losses)))), Y=np.column_stack((np.array(v_losses), np.array(p_losses))), win=win1, opts=dict(legend=["Value loss", "Policy loss"]))
    print(-running_reward)
    print(-best)
    print("Best in buffer: " + str(-best_in_buffer))

print("Finally, best options are:")
utils.print_opt(best_options)
save_model(net, "bigliste_standard.pickle")
#ipdb.set_trace()
#torch.save(net.state_dict,'rl_memory_trained_conv_standard.pt')