#!/bin/sh

#SBATCH -J TensorComprehensions # A single job name for the array
#SBATCH -n 20 # Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH --mem 40000 # Memory request (4Gb)
#SBATCH -t 0-3:00 # Maximum execution time (D-HH:MM)
#SBATCH --gres=gpu:2
#SBATCH --partition=learnfair-2g

export TUNER_THREADS=${TUNER_THREADS:=20}
export TUNER_GPUS=${TUNER_GPUS:="0,1"}
export GPU_NAME=$(nvidia-smi -L | head -n 1 | cut -d'(' -f 1 | cut -d':' -f 2 | sed "s/ //g")

export TC_PREFIX=$(git rev-parse --show-toplevel)
export PREFIX=${TC_PREFIX}/examples/results_$(date +%m%d%y)/${GPU_NAME}
export LOG_DIR=${TC_PREFIX}/examples/results_$(date +%m%d%y)/${GPU_NAME}/logs/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}

mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/autotuner
chmod -R 777 ${LOG_DIR}

cat ${TC_PREFIX}/examples/scripts/AUTOTUNER_COMMANDS | grep -v "\#" | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1 | xargs -i echo {} > ${LOG_DIR}/COMMAND
cat ${TC_PREFIX}/examples/scripts/AUTOTUNER_COMMANDS | grep -v "\#" | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1 | xargs -i bash -c "{}"

# Run with:
# sbatch --array=1-14 ./examples/scripts/autotuner_parallel.sh
