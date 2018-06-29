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
import tensor_comprehensions as tc
#from visdom import Visdom
from collections import deque

import my_utils

set_options = [
        ,
        [1, 1, 0, 0, 8, 1, 2, 7, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 8, 0, 7, 8, 1, 1, 0, 0, 1],
        #[ 0, 0, 0, 3, 26, 1, 14, 26, 0, 1, 0, 0, 1]
        ]

code = """
def group_normalization(
    float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta)
    -> (O, mean, var)
{
    mean(n, g) +=! I(n, g, r_d, r_h, r_w)
     var(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)

    O(n, g, d, h, w) = gamma(g, d)
      * ( I(n, g, d, h, w) - mean(n, g) * 4 )
      * rsqrt( var(n, g) * 4
            - mean(n, g) * mean(n, g) * 4 * 4
            + 1e-5)
      + beta(g, d)
}
"""

N, G, D, H, W = 10, 10, 10, 10, 10
I, gamma, beta = torch.randn(N, G, D, H, W).cuda(), torch.randn(G, D).cuda(), torch.randn(G, D).cuda()

init_input = (I, gamma, beta)
init_input_sz = np.array([N,G,D,H,W])
init_input_sz = torch.from_numpy(init_input_sz).float()

inp = init_input
my_utils.computeCat(inp)

tc_prog = tc.define(code, name="group_normalization")
my_utils.set_tcprog(tc_prog)

cachef = "my_file_2.txt"
config = tc.autotuner_settings
config["pop_size"]=50
config["generations"]=1

NB_HYPERPARAMS, INIT_INPUT_SZ = 13, 5

opts = np.array(set_options[2])
#print(opts[3])
#print(my_utils.cat_val[3])
#opts = [my_utils.cat_val[i][opts[i]] for i in range(NB_HYPERPARAMS)]
print(opts)
opts = my_utils.optionsFromVector(opts)

tc_prog.autotune(I, gamma, beta, options=opts, **config, cache=cachef)
print(tc.decode(cachef+".options"))


