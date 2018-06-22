#!/bin/bash

# NB: define this function before set -x, so that we don't
# pollute the log with a premature EXITED_USER_LAND ;)
function cleanup {
  # Note that if you've exited user land, then CI will conclude that
  # any failure is the CI's fault.  So we MUST only output this
  # string
  retcode=$?
  set +x
  if [ $retcode -eq 0 ]; then
    echo "EXITED_USER_LAND"
  fi
}

set -ex

source /etc/lsb-release

# note: printf is used instead of echo to avoid backslash
# processing and to properly handle values that begin with a '-'.
echo "ENTERED_USER_LAND"
log() { printf '%s\n' "$*"; }
error() { log "ERROR: $*" >&2; }
fatal() { error "$@"; exit 1; }

# appends a command to a trap
#
# - 1st arg:  code to add
# - remaining args:  names of traps to modify
#
trap_add() {
    trap_add_cmd=$1; shift || fatal "${FUNCNAME} usage error"
    for trap_add_name in "$@"; do
        trap -- "$(
            # helper fn to get existing trap command from output
            # of trap -p
            extract_trap_cmd() { printf '%s\n' "$3"; }
            # print existing trap command with newline
            eval "extract_trap_cmd $(trap -p "${trap_add_name}")"
            # print the new trap command
            printf '%s\n' "${trap_add_cmd}"
        )" "${trap_add_name}" \
            || fatal "unable to add to trap ${trap_add_name}"
    done
}
# set the trace attribute for the above function.  this is
# required to modify DEBUG or RETURN traps because functions don't
# inherit them unless the trace attribute is set
declare -f -t trap_add

trap_add cleanup EXIT

# Check we indeed have GPUs and list them in the log file
nvidia-smi

# Just install missing conda dependencies, build and run tests
cd /var/lib/jenkins/workspace
. /opt/conda/anaconda/bin/activate
git submodule update --init --recursive

source activate tc_build
conda install -y -c nicolasvasilache llvm-tapir50 halide
conda install -y -c conda-forge eigen
conda install -y -c nicolasvasilache caffe2

WITH_CAFFE2=ON CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda CLANG_PREFIX=$(${CONDA_PREFIX}/bin/llvm-config --prefix) BUILD_TYPE=Release ./build.sh

python setup.py install
./test_python/run_test.sh

for f in $(find ./python/examples -name "*.py"); do
    python $f -v
done

FILTER_OUT="benchmark_MLP_model benchmark_kronecker" ./test.sh
# 2LUT can OOM on smaller Maxwells on our CI machines
./build/tc/benchmarks/benchmark_MLP_model --gtest_filter=-*2LUT*
# Kronecker xxxAsMatMul can OOM
./build/tc/benchmarks/benchmark_kronecker --gtest_filter=-*AsMatMul*
