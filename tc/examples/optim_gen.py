import numpy as np
import ipdb
import torch
import tensor_comprehensions as tc

import my_utils

tc_code, tc_name, inp, init_input_sz = my_utils.get_default_convolution_example()
my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)
#config = tc.autotuner_settings
#config["pop_size"]=50
#config["generations"]=1
opts = tc.MappingOptions("naive")
print(opts)

tc.autotune(tc_code, tc_name, *inp, starting_options=opts, cache_filename="genetic_savedopt_conv_default.txt", store_to_cache=True)
