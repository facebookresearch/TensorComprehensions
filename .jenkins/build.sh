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

# condition: if 14.04 and conda, conda install pytorch and build
# condition: if 16.04 and conda, conda install pytorch and build
# condition: if any and non-conda, simply build TC from scratch

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

if which ccache > /dev/null; then
  # Report ccache stats for easier debugging
  ccache --zero-stats
  ccache --show-stats
  function ccache_epilogue() {
    ccache --show-stats
  }
  trap_add ccache_epilogue EXIT
fi

if [[ "$DISTRIB_RELEASE" == 14.04 ]]; then
  if [[ $(conda --version | wc -c) -ne 0 ]]; then
    echo "Building TC in conda env"
    conda create -y --name tc-env python=3.6
    source activate tc-env
    conda install -y -c pytorch pytorch
    conda install -y pyyaml
    WITH_PYTHON_C2=OFF CORES=$(nproc) CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
  else
    echo "Building TC in non-conda env"
    WITH_PYTHON_C2=OFF CORES=$(nproc) CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
  fi
fi

if [[ "$DISTRIB_RELEASE" == 16.04 ]]; then
  if [[ $(conda --version | wc -c) -ne 0 ]]; then
    echo "Building TC in conda env"
    conda create -y --name tc-env python=3.6
    source activate tc-env
    conda install -y pytorch cuda90 -c pytorch
    conda install -y pyyaml
    WITH_PYTHON_C2=OFF CORES=$(nproc) CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
  else
    echo "Building TC in non-conda env"
    WITH_PYTHON_C2=OFF CORES=$(nproc) CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
  fi
fi
