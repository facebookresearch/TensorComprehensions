import tensor_comprehensions as tc
import tensor_comprehensions.tclib as tclib
import ipdb
cache = tc.MappingOptionsCache("genetic_savedopt_conv_default.txt")
import my_utils
tc_code, tc_name, inp, init_input_sz = my_utils.get_convolution_example()
my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)
print("divs : " +str(my_utils.getAllDivs(inp)))
best_options, = cache.load(tc_code, tc_name, inp, 1)
best_options = best_options.getDict()
optsVect = my_utils.getRawVectorFromTcOpt(best_options)
#optsVect = my_utils.catVec_to_optVec(optsVect)
opts = my_utils.optionsFromVector(optsVect)
print(opts)
tc_prog = tc.compile(tc_code, tc_name, opts, *inp)
time = my_utils.evalTime(opts, estimator="median")
print(time)
tt=0.
for i in range(20):
    ct = tc_prog.executor.profile_kernel(inp)
    if i >= 10:
        tt+=ct
print(tt/10.)
