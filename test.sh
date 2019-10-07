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

NGPU=$(nvidia-smi -L | wc -l)
let DEFAULT_PAR="${NGPU}"
NPAR=${NPAR:=${DEFAULT_PAR}}

FILTER_OUT=${FILTER_OUT:=some-weird-name-that-we-never-hit}

TESTS=
filter() {
    for f in $@; do
        if ! test -z $(echo ${FILTER_OUT} | grep $(basename $f)); then
            echo Skip $f
            continue
        fi
        TESTS="$TESTS $f"
    done
}

filter $(find build/test -type f -executable | grep -v "\.")
# Examples run the full search on a single thread, punt for now
# filter $(find build/tc/examples -type f -executable | grep -v "\.")
filter $(find build/tc/benchmarks -type f -executable | grep -v "\.")
for f in $TESTS; do echo $f; done | xargs -n 1 -P ${NPAR} -i bash -c 'echo Running {}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=$((RANDOM % '"${NGPU}"'))} ./{} "'\
"$@"'" || exit 255' || (echo "$(tput setaf 1)Some tests are broken $(tput sgr 0)" && exit 1)

echo SUCCESS

