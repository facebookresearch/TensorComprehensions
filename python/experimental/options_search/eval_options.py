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
[0, 0, 0, 2, 4, 13, 8, 1, 8, 7, 3, 0, 2, 12, 11, 5, 2, 3, 5, 0, 1, 1, 1, 0, 1, 1]
#[0, 1, 0, 0, 1, 8, 9, 18, 10, 16, 6, 0, 18, 13, 1, 0, 1, 0, 1],
#[1, 1, 0, 1, 1, 20, 14, 3, 8, 5, 3, 0, 18, 15, 1, 0, 0, 0, 1]
]

(tc_code, tc_name, inp, init_input_sz) = my_utils.get_convolution_example()

my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)

NB_HYPERPARAMS, INIT_INPUT_SZ = my_utils.NB_HYPERPARAMS, my_utils.INIT_INPUT_SZ

for i in range(len(set_options)):
    opts = np.array(set_options[i])
    time = my_utils.evalTime(opts)
    print(time)
