#!/usr/bin/env bash
set -e

PYTHON=${PYTHON:="`which python3`"}

pushd "$(dirname "$0")"

###############################################################################
# Some basic TC tests
###############################################################################
echo "Running TC imports test"
$PYTHON test_tc_imports.py -v

echo "Running Mapping options test"
$PYTHON test_mapping_options.py -v

echo "Running normal TC test"
$PYTHON test_tc.py -v

echo "Running debug init test"
$PYTHON test_debug_init.py -v

###############################################################################
# PyTorch TcBuilder
###############################################################################
echo "Running PyTorch TcBuilder example"
$PYTHON pytorch_example.py -v

###############################################################################
# PyTorch testing all features
###############################################################################
echo "Running all PyTorch tests"
$PYTHON test_tc_torch.py -v

###############################################################################
# PyTorch layer tests
###############################################################################
echo "Running all PyTorch layers tests"
$PYTHON -m pytest -v --full-trace --junit-xml="/tmp/tensorcomp/python/result.xml" layers/

echo "All PyTorch layer tests have finished"

popd
