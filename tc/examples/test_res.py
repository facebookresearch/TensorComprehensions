import tensor_comprehensions as tc
import tensor_comprehensions.tclib as tclib
import ipdb
cache = tc.MappingOptionsCache("savedopt_ns.txt")
import my_utils
tc_code, tc_name, inp, init_input_sz = my_utils.get_convolution_example(already_set=True, inp_sz_list=[8,2,28,28,8,1,1])
my_utils.computeCat(inp)
my_utils.set_tc(tc_code, tc_name)
print("divs : " +str(my_utils.getAllDivs(inp)))
#best_options, = cache.load(tc_code, tc_name, inp, 1)
#ipdb.set_trace()
#best_options = best_options.getDict()
#optsVect = my_utils.getRawVectorFromTcOpt(best_options)
optsVect = [1, 1, 0, 1, 2, 1, 3, 6, 8, 0, 3, 0, 2, 11, 9, 8, 2, 0, 0, 3, 0, 0, 1, 0, 1, 2]
#[1, 1, 0, 4, 3, 8, 2, 0, 8, 10, 7, 0, 2, 5, 10, 7, 1, 3, 8, 11, 1, 0, 0, 0, 1, 0]
optsVect = my_utils.catVec_to_optVec(optsVect)
opts = my_utils.optionsFromVector(optsVect)
print(opts)
tc_prog = tc.compile(tc_code, tc_name, opts, *inp)
tt=0.
for i in range(20):
    ct = tc_prog.executor.profile_kernel(inp)
    if i >= 10:
        tt+=ct
print(tt/10.)
