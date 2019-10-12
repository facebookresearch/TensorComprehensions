# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#!/bin/sh

#SBATCH -J TensorComprehensions # A single job name for the array
#SBATCH -n 8 # Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH --mem 40000 # Memory request (4Gb)
#SBATCH -t 0-2:00 # Maximum execution time (D-HH:MM)
#SBATCH --gres=gpu:2
#SBATCH --partition=priority,uninterrupted,learnfair,scavenge

module load cuda/9.0
. ${HOME}/anaconda/bin/activate
conda activate tc_build
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

export TUNER_THREADS=${TUNER_THREADS:=8}
export TUNER_DEVICES=${TUNER_DEVICES:="0,1"}
export DEVICE_NAME=$(nvidia-smi -L | head -n 1 | cut -d'(' -f 1 | cut -d':' -f 2 | sed "s/ //g")

export TC_PREFIX=$(git rev-parse --show-toplevel)
export PREFIX=${TC_PREFIX}/tc/benchmarks/results_$(date +%m%d%y)/${DEVICE_NAME}
export LOG_DIR=${TC_PREFIX}/tc/benchmarks/results_$(date +%m%d%y)/${DEVICE_NAME}/logs/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}

mkdir -p ${LOG_DIR}
chmod -R 777 ${LOG_DIR}

cat ${TC_PREFIX}/tc/benchmarks/scripts/AUTOTUNER_COMMANDS | grep -v "\#" | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1 | xargs -i echo {} > ${LOG_DIR}/COMMAND
cat ${TC_PREFIX}/tc/benchmarks/scripts/AUTOTUNER_COMMANDS | grep -v "\#" | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1 | xargs -i bash -c "{}"

# Run with:
# export NUM_TO_RUN=$(cat ${TC_PREFIX}/tc/benchmarks/scripts/AUTOTUNER_COMMANDS | grep -v "\#" | wc -l)
# sbatch --array=1-${NUM_TO_RUN} -C volta ./tc/benchmarks/scripts/autotuner_parallel.sh
