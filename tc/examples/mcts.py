import tensor_comprehensions as tc
import torch
import my_utils
import numpy as np
from tqdm import tqdm
from visdom import Visdom

viz = Visdom(server="http://100.97.69.78")

class Node:
    def __init__(self, father=None, new_act=0):
        self.value = 0
        self.nbVisits=0
        self.nbChildrenSeen = 0
        self.pos=0
        #self.hasSeen = {} #todo
        self.children=[]
        self.parent = father
        self.stateVector = [0] * my_utils.NB_HYPERPARAMS
        if(father != None):
            self.pos = father.pos+1
            #self.hasSeen = {} #todo
            self.stateVector = father.stateVector[:]
            self.stateVector[self.pos-1] = new_act

    def getRoot(self):
        return self

    def getParent(self):
        return self.parent

    def notRoot(self):
        return (self.parent != None)

class MCTS:
    def __init__(self):
        self.C = 1 #to tune

        (tc_code, tc_name, inp, _) = my_utils.get_convolution_example(already_set=True, inp_sz_list=[8,2,28,28,8,1,1])

        my_utils.computeCat(inp)
        my_utils.set_tc(tc_code, tc_name)

        self.nbActions = my_utils.cat_sz
        self.tree = Node()

        self.best_rewards = []
        self.rws = []

        self.curIter=0
        self.curr_best=0
        self.running_reward=0
        self.win0 = viz.line(X=np.arange(5), Y=np.random.rand(5))

    def main_search(self, starting_pos): #, init_inp):
        node = starting_pos
        #node.nbVisits+=1
        ttNbIters = 20#2*self.nbActions[node.pos]
        for _ in range(max(ttNbIters, self.nbActions[node.pos])):
            leaf = self.getLeaf(node)
            val = self.evaluate(leaf)
            self.backup(leaf, val)
            #print(node.value / node.nbVisits)
        _, action = self.getBestChild(node)
        return action

    def take_action(self, node, act):
        if(node.nbChildrenSeen > act):
            return node.children[act]
        new_child = Node(father=node, new_act=act)
        node.children.append(new_child)
        #node.hasSeen[act]=1
        node.nbChildrenSeen += 1
        return node.children[-1]

    def getLeaf(self, node):
        first=True
        while(node.pos < my_utils.NB_HYPERPARAMS and (first or node.nbVisits != 0)):
            first=False
            pos = node.pos
            if(node.nbChildrenSeen == self.nbActions[pos]):
                node, _ = self.getBestChild(node)
            else:
                act=node.nbChildrenSeen
                self.take_action(node, act)
                return node.children[-1]
        return node

    def getBestChild(self, node):
        bestIndic = 0.
        bestAction = 0
        first=True
        pos = node.pos
        for act in range(self.nbActions[pos]):
            child = node.children[act]
            indic = child.value / child.nbVisits + self.C * np.sqrt(2*np.log(node.nbVisits) / child.nbVisits)
            if(first or indic > bestIndic):
                bestIndic = indic
                bestAction = act
                first=False
        return node.children[bestAction], bestAction

    def saveReward(self, reward):
        INTER_DISP = 20
        #print(-reward)
        if(self.curIter == 0):
            self.running_reward = reward
            self.curr_best = reward
        self.curIter += 1
        self.running_reward = self.running_reward * 0.99 + reward * 0.01
        self.curr_best = max(self.curr_best, reward)
        #self.rewards.append(-reward)
        self.best_rewards.append(-self.curr_best)
        self.rws.append(-self.running_reward)
        if self.curIter % INTER_DISP == 0:
            viz.line(X=np.column_stack((np.arange(i+1), np.arange(i+1))), \
            Y=np.column_stack((np.array(self.rws), np.array(self.best_rewards))), \
            win=self.win0, opts=dict(legend=["Geometric run", "Best time"]))

    def randomSampleScoreFrom(self, node):
        pos = node.pos
        optsVector = node.stateVector
        for i in range(my_utils.NB_HYPERPARAMS - (pos)):
            a = np.random.randint(self.nbActions[i+pos])
            optsVector[i+(pos)] = a
        print(optsVector)
        reward = -np.log(my_utils.evalTime(optsVector))
        self.saveReward(reward)
        return reward

    def evaluate(self, leaf):
        score = 0
        nb_iters=2
        for _ in range(nb_iters):
            score += self.randomSampleScoreFrom(leaf)
        return score / nb_iters

    def backup(self, leaf, val):
        node = leaf
        while(node.notRoot()):
            node.nbVisits += 1
            node.value += val
            node = node.getParent()
        node.nbVisits += 1
        node.value += val

mcts = MCTS()

opts = []
curr_node = mcts.tree
for i in tqdm(range(my_utils.NB_HYPERPARAMS)):
    opts.append(mcts.main_search(curr_node))
    curr_node = mcts.take_action(curr_node, opts[-1])
    print(opts)
opts = np.array(opts).astype(int)
print(my_utils.evalTime(opts.tolist()))
my_utils.print_opt(opts)
