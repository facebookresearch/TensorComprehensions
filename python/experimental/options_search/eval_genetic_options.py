import tensor_comprehensions as tc
import tensor_comprehensions.tclib as tclib
import utils

cache = tc.MappingOptionsCache("genetic_savedopt_conv_default.txt")

exptuner_config = utils.ExpTunerConfig()
exptuner_config.set_convolution_tc()
tc_code, tc_name, inp = exptuner_config.tc_code, exptuner_config.tc_name, exptuner_config.inp

print("divs : " + str(utils.getAllDivs(inp)))
tup = cache.load(tc_code, tc_name, inp, 1)
if(tup == []):
    exit()
best_options, = tup
best_options = best_options.getDict()
optsVect = utils.getRawVectorFromTcOpt(best_options)
opts = utils.optionsFromVector(optsVect)
print(opts)

time = utils.evalTime(opts, exptuner_config, estimator="median")
print(time)

