# Building

These benchmarks are automatically built when ```WITH_CAFFE2=ON``` is passed.
If you have been following the instructions given [here](https://facebookresearch.github.io/TensorComprehensions/installation.html), you can use the command:

```
BUILD_TYPE=Release WITH_CAFFE2=ON CLANG_PREFIX=$(${CONDA_PREFIX}/bin/llvm-config --prefix) ./build.sh
```

# Running the autotuner manually
By default a full evolutionary search is run with 25 generations and 100 candidates per generation. This will take some time for some of the kernels. This setting be changed by using the proper gflags options: ```--tuner_gen_generations``` and ```--tuner_gen_pop_size```.

For instance, a shorter tuning search could iterate as such:
```
./build/tc/benchmarks/benchmark_batchmatmul --autotune=true --tuner_gen_generations=10 --tuner_gen_pop_size=20
```

When running manually, the number of CPU compilation threads and GPUs used for evaluation can be controlled via gflags
```--tuner_threads``` and ```--tuner_devices```

For instance, on a 4 GPU system with 20 threads:
```
./build/tc/benchmarks/benchmark_batchmatmul --autotune=true --tuner_gen_generations=10 --tuner_gen_pop_size=10 --tuner_threads=20 --tuner_devices="0,1,2,3"
```

# Running the autotuner with provided scripts
These examples are run as part of ```test.sh``` but can also be run with a full autotuning run

If you are the lucky owner of a supercomputer with ```slurm``` and ```sbatch``` you can run:
```
sbatch --array=1-40 ./tc/benchmarks/scripts/autotuner_parallel.sh
```

Results and logs will show in the subdir ```tc/benchmarks/results_xxx```, one can tail the ```*.INFO``` to obtain the best performance found by the autuner.

To control the CPU compilation threads and the GPUs used for evaluation, please use the environment variables ```TUNER_THREADS``` and ```TUNER_GPUS```.
For instance, on a 4 GPU machine:
```
for f in $(seq 1 14); do TUNER_THREADS=20 TUNER_GPUS="0,1,2,3" SLURM_ARRAY_JOB_ID=local SLURM_ARRAY_TASK_ID=$f ./tc/benchmarks/scripts/autotuner_parallel.sh ; done
```
