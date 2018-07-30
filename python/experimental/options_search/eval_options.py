import numpy as np
import time
import utils
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tensor_comprehensions as tc

set_options = [
[1, 1, 0, 1, 2, 1, 3, 6, 8, 0, 3, 0, 2, 11, 9, 8, 2, 0, 0, 3, 0, 0, 1, 0, 1, 2]
]

exptuner_config = utils.ExpTunerConfig()
exptuner_config.set_convolution_tc()

for i in range(len(set_options)):
    opts = np.array(set_options[i])
    time = utils.evalTime(opts, exptuner_config)
    print(time)
