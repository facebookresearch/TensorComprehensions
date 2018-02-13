#! /bin/bash

set -e

NGPU=$(nvidia-smi -L | wc -l)
let DEFAULT_PAR="${NGPU}"
NPAR=${NPAR:=${DEFAULT_PAR}}

FILTER_OUT=${FILTER_OUT:=some-weird-name-that-we-never-hit}

find build -name "test_*" -type f -executable -name "*${FILTER}*" | grep -v ${FILTER_OUT} | xargs -n 1 -P ${NPAR} -i bash -c 'echo Running {}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=$((RANDOM % '"${NGPU}"'))} ./{} "'"$@"'" || exit 255' \
    || (echo "$(tput setaf 1)Some tests are broken $(tput sgr 0)" && exit 1)

find build -name "example_*" -type f -executable -name "*${FILTER}*" | grep -v ${FILTER_OUT} | xargs -n 1 -P ${NPAR} -i bash -c 'echo Running {}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=$((RANDOM % '"${NGPU}"'))} ./{} --benchmark_warmup=0 --benchmark_iterations=1"'"$@"'" --disable_reproducibility_checks || exit 255' \
    || (echo "$(tput setaf 1)Some tests are broken $(tput sgr 0)" && exit 1)

echo SUCCESS
