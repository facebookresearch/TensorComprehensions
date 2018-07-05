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

(tc_code, tc_name, inp, init_input_sz) = my_utils.get_convolution_example()

#config = tc.autotuner_settings
#config["pop_size"]=50
#config["generations"]=1
opts = tc.CudaMappingOptions("naive")
tc.autotune(tc_code, tc_name, inp, starting_options=opts)
