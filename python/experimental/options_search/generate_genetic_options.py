import numpy as np
import torch
import tensor_comprehensions as tc

import utils

exptuner_config = utils.ExpTunerConfig()
exptuner_config.set_convolution_tc()
tc_code, tc_name, inp = exptuner_config.tc_code, exptuner_config.tc_name, exptuner_config.inp
#config = tc.autotuner_settings
#config["pop_size"]=50
#config["generations"]=1
opts = tc.MappingOptions("naive")
print(opts)

tc.autotune(tc_code, tc_name, *inp, starting_options=opts, cache_filename="genetic_savedopt_conv_default.txt", store_to_cache=True)
