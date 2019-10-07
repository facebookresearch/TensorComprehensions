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
#! /bin/bash

set -e

PYTHON=${PYTHON:="`which python3`"}
WITH_CUDA=${WITH_CUDA:="ON"}

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
TEST_REGEX="test_basic test_core test_inference test_isl_scheduler test_lang test_mapper* test_tc2halide test_cuda_mapper test_cuda_mapper_memory_promotion"
run_test

echo SUCCESS
