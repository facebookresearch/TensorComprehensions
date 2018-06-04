#! /bin/bash

[ -z "$BACKEND" ] && echo "Need to set BACKEND (P100 or V100)" && exit 1;

TC_DIR=$(git rev-parse --show-toplevel)

BENCHMARKS=$(find ${TC_DIR}/build/tc/benchmarks/benchmark*)

for b in ${BENCHMARKS}; do
    $b --gtest_filter="*${BACKEND}*";
done
