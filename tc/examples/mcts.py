import tensor_comprehensions as tc 
import torch
import my_utils
import numpy as np

class Node:
    def __init__(self, father=None, new_act=0):
        self.value = 0
        self.nbVisits=0
        self.nbChildrenSeen = 0
        self.pos=0
        self.hasSeen = {} #todo
        self.children=[]
        self.parent = father
        self.stateVector = [0] * my_utils.NB_HYPERPARAMS
        if(father != None):
            self.pos = father.pos+1
            self.hasSeen = {} #todo
            self.stateVector = father.stateVector
            self.stateVector[self.pos-1] = new_act
    
    def getRoot(self):
        return self

    def getParent(self):
        return self.parent

    def notRoot(self):
        return (self.parent != None)

class MCTS:
    def __init__(self):
        self.C = 1. #to tune

        (tc_code, tc_name, inp, _) = my_utils.get_convolution_example()

        my_utils.computeCat(inp)
        my_utils.set_tc(tc_code, tc_name)

        self.nbActions = my_utils.cat_sz
        self.tree = Node()

    def main_search(self, starting_pos): #, init_inp):
        node = starting_pos
        for _ in range(1000):
            leaf = self.getLeaf(node)
            val = self.evaluate(leaf)
            self.backup(leaf, val)
        _, action = self.getBestChild(node)
        return action

    def take_action(self, node, act):
        return Node(father=node, new_act=act)
    
    def getLeaf(self, node):
        first=True
        while(first or node.nbVisits != 0):
            first=False
            pos = node.pos
            if(node.nbChildrenSeen == self.nbActions[pos]):
                node, _ = self.getBestChild(node)
            else:
                act=node.nbChildrenSeen
                node.hasSeen[act]=1
                node.nbChildrenSeen += 1
                node.children.append(self.take_action(node, act))
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
    
    def randomSampleScoreFrom(self, node):
        pos = node.pos
        optsVector = node.stateVector
        for i in range(my_utils.NB_HYPERPARAMS - (pos)):
            a = np.random.randint(self.nbActions[pos])
            optsVector[i+(pos)] = a
        print(optsVector)
        reward = -np.log(my_utils.evalTime(optsVector))
        return reward

    def evaluate(self, leaf):
        score = 0
        for _ in range(10):
            score += self.randomSampleScoreFrom(leaf)
        return score / 10

    def backup(self, leaf, val):
        node = leaf
        while(node.notRoot()):
            node.nbVisits += 1
            node.value += val
            node = node.getParent()

mcts = MCTS()

opts = []
curr_node = mcts.tree
for i in range(my_utils.NB_HYPERPARAMS):
    opts.append(mcts.main_search(curr_node))
    curr_node = mcts.take_action(curr_node, opts[-1])
print(my_utils.evalTime(opts))
my_utils.print_opt(opts)