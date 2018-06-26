import numpy as np
import time
import my_utils
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tensor_comprehensions as tc

set_options = [
[ 1, 1, 1, 13, 25, 0, 27, 18, 1, 0, 0, 0, 0],
[ 2, 1, 0, 26, 27, 0, 0, 32, 0, 0, 0, 0, 1],
[ 0, 0, 0, 3, 26, 1, 14, 26, 0, 1, 0, 0, 1],
[ 2, 0, 0, 20, 27, 0, 28, 1, 1, 0, 1, 0, 0],
[ 1, 2, 1, 2, 15, 1, 16, 25, 0, 0, 0, 0, 1] #batch 16 etc
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

N, G, D, H, W = 5, 5, 5, 5, 5
I, gamma, beta = torch.randn(N, G, D, H, W).cuda(), torch.randn(G, D).cuda(), torch.randn(G, D).cuda()

init_input = (I, gamma, beta)
init_input_sz = np.array([N,G,D,H,W])
init_input_sz = torch.from_numpy(init_input_sz).float()

inp = init_input
my_utils.computeCat(inp)

tc_prog = tc.define(code, name="group_normalization")
my_utils.set_tcprog(tc_prog)

perm = np.random.permutation(len(set_options))
print(perm)
#set_options = set_options[perm]
for i in range(len(set_options)):
    opts = np.array(set_options[perm[i]])
    temps = my_utils.evalTime(opts, 1000, 50)
    print(temps)
print("and naive")
print(my_utils.evalTime(set_options[0], 1000, 50, naive=True))
