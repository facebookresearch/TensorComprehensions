#! /bin/bash

set -e

PYTHON=${PYTHON:="`which python3`"}

function run_test {
    ACC=""
    for f in $(echo ${TEST_REGEX}); do
        for ff in $(find ${TEST_PATH} -name $f -type f -executable); do
            ACC=$(echo ${ACC} ${ff})
        done
    done
    NPAR=${NPAR:=16}
    echo ${ACC} | tr '\n' ' ' | tr ' ' '\n' | xargs -n 1 -P ${NPAR} -i bash -c 'echo Running {}; ./{}|| exit 255' \
    || (echo "$(tput setaf 1)Some tests are broken $(tput sgr 0)" && exit 1)
}

TEST_PATH="./build/test"
TEST_REGEX="test_basic test_core test_inference test_isl_scheduler test_lang test_mapper* test_tc2halide"
run_test

# Python tests - we basically only want to test the libs are built correctly and
# import fine
echo "Running Python tests"

echo "Setting PYTHONPATH only"
export TC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${TC_DIR}/build/tensor_comprehensions/pybinds:${PYTHONPATH}
echo "PYTHONPATH: ${PYTHONPATH}"
${PYTHON} -c 'import tc'
${PYTHON} -c 'import mapping_options'
${PYTHON} -c 'import autotuner'

echo SUCCESS
