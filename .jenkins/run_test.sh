#!/bin/bash

set -ex

source /etc/lsb-release

# condition: if 14.04 and conda, also run python tests
# condition: if 16.04 and conda, run only python tests
# condition: if any and non-conda, run test.sh only

if [[ "$DISTRIB_RELEASE" == 14.04 ]]; then
  echo "Running TC backend tests"
  ./test.sh
  if which conda &> /dev/null; then
    source activate
    echo "Running TC PyTorch tests"
    ./test_python/run_test.sh
  fi
fi

if [[ "$DISTRIB_RELEASE" == 16.04 ]]; then
  if which conda &> /dev/null; then
    echo "Running TC PyTorch tests"
    source activate
    ./test_python/run_test.sh
  else
    echo "Running TC backend tests"
    ./test.sh
  fi
fi
