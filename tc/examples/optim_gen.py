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

inp = (I, gamma, beta)
my_utils.computeCat(inp)

tc_prog = tc.define(code, name="group_normalization")
cache = "my_file.txt"
config = tc.autotuner_settings
config["pop_size"]=50
config["generations"]=1
opts = [1, 0, 0, 1, 8, 0, 7, 8, 1, 1, 0, 0, 1]
opts = [1, 1, 0, 0, 8, 1, 2, 7, 0, 1, 1, 0, 1]
opts = [my_utils.cat_val[i][opt] for i,opt in enumerate(opts)]
opts = my_utils.optionsFromVector(opts)
opts = tc.CudaMappingOptions("naive")
tc_prog.autotune(I, gamma, beta, options=opts, **config, cache="./bidule.txt")
print(tc.decode(cache+".options"))


