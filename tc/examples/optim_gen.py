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

tc_code, tc_name, inp, init_input_sz = my_utils.get_convolution_example()
my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)
#config = tc.autotuner_settings
#config["pop_size"]=50
#config["generations"]=1
opts = tc.MappingOptions("naive")
#vec = [1, 2, 1, 1, 7, 4, 12, 0, 6, 7, 2, 0, 2, 5, 12, 3, 2, 0, 1, 3, 1, 0, 1, 0, 1, 2]#[0, 1, 0, 2, 5, 17, 7, 3, 3, 16, 3, 0, 2, 18, 17, 2, 2, 13, 7, 7, 0, 1, 0, 0, 1, 1]
#vec = my_utils.catVec_to_optVec(vec)
#print(vec)
#opts = my_utils.optionsFromVector(vec)
#print(inp, tc_code, tc_name, init_input_sz)

#print(opts)

tc.autotune(tc_code, tc_name, *inp, starting_options=opts, cache_filename="savedopt.txt", store_to_cache=True)
