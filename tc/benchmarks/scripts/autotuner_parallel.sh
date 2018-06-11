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
# sbatch --array=1-40 -C volta ./tc/benchmarks/scripts/autotuner_parallel.sh
