# Building

These examples are automatically built when ```WITH_CAFFE2=ON``` is passed.
If you have been following the instructions given [here](https://github.com/facebookresearch/TensorComprehensions/blob/release/docs/source/installation.rst), you can use the command:

```
BUILD_TYPE=Release PYTHON=$(which python3) WITH_CAFFE2=ON CLANG_PREFIX=$HOME/clang+llvm-tapir5.0 ./build.sh --all
```

# Reproducibility
To reproduce results in our accompanying paper, we ensure cuda 8.0, nvrtc8.0 and cudnn6.0 are available. To disable those checks please use the ```--disable_version_checks=true``` flag.

For instance:
```
./build/examples/example_batchmatmul --disable_version_checks=true
```

# Running the autotuner manually
By default a full evolutionary search is run with 25 generations and 100 candidates per generation. This will take some time for some of the kernels. This setting be changed by using the proper gflags options: ```--tuner_gen_generations``` and ```--tuner_gen_pop_size```.

For instance, a more reasonable search could iterate as such:
```
./build/examples/example_batchmatmul --disable_version_checks=true --autotune=true --tuner_gen_generations=10 --tuner_gen_pop_size=20
```

Note
Running maually may trigger a reload and restart from the last saved best options, the autotuner would then print:
```
Loading proto from: /tmp/batchmatmul_cache_B_500_K_26_M_72_N_26.options and /tmp/batchmatmul_cache_B_500_K_26_M_72_N_26.cuda
```


Also when running manually, the number of CPU compilation threads and GPUs used for evaluation can be controlled via gflags
```--tuner_threads``` and ```--tuner_devices```

For instance, on a 4 GPU system with 20 threads:
```
./build/examples/example_batchmatmul --disable_version_checks=true --autotune=true --tuner_gen_generations=10 --tuner_gen_pop_size=10 --tuner_threads=20 --tuner_devices="0,1,2,3"
```

# Running the autotuner with provided scripts
These examples are run as part of ```test.sh``` but can also be run with a full autotuning run

If you are the lucky owner of a supercomputer with ```slurm``` and ```sbatch``` you can run:
```
sbatch --array=1-14 ./examples/scripts/autotuner_parallel.sh
```

Otherwise the following is valid too:
```
for f in $(seq 1 14); do SLURM_ARRAY_JOB_ID=local SLURM_ARRAY_TASK_ID=$f ./examples/scripts/autotuner_parallel.sh ; done
```

Results and logs will show in the subdir ```examples/results_xxx```, one can tail the ```*.INFO``` to obtain the best performance found by the autuner.

For instance:
```
for f in $(find examples/results* -name "*INFO"); do echo " " && cat $(dirname $f)/COMMAND && grep Generation $f | tail -n 8; done | less
```

To control the CPU compilation threads and the GPUs used for evaluation, please use the environment variables ```TUNER_THREADS``` and ```TUNER_GPUS```.
For instance, on a 4 GPU machine:
```
for f in $(seq 1 14); do TUNER_THREADS=20 TUNER_GPUS="0,1,2,3" SLURM_ARRAY_JOB_ID=local SLURM_ARRAY_TASK_ID=$f ./examples/scripts/autotuner_parallel.sh ; done
```
