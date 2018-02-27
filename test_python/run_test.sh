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
# PyTorch testing all features
###############################################################################
echo "Running all PyTorch tests"
$PYTHON test_tc_torch.py -v

###############################################################################
# PyTorch layer tests
###############################################################################
echo "Running all PyTorch layers tests"
$PYTHON layers/test_absolute.py -v
$PYTHON layers/test_autotuner.py -v
$PYTHON layers/test_avgpool.py -v
$PYTHON layers/test_avgpool_autotune.py -v
$PYTHON layers/test_batchmatmul.py -v
$PYTHON layers/test_batchnorm.py -v
$PYTHON layers/test_layernorm.py -v
$PYTHON layers/test_cast.py -v
$PYTHON layers/test_concat.py -v
$PYTHON layers/test_convolution.py -v
$PYTHON layers/test_convolution_strided.py -v
$PYTHON layers/test_convolution_reorder.py -v
$PYTHON layers/test_convolution_strided_autotune.py -v
$PYTHON layers/test_convolution_train.py -v
$PYTHON layers/test_copy.py -v
$PYTHON layers/test_cos.py -v
$PYTHON layers/test_cosine_similarity.py -v
$PYTHON layers/test_dump_cuda.py -v
$PYTHON layers/test_external_cuda_injection.py -v
$PYTHON layers/test_fc.py -v
$PYTHON layers/test_fusion_fcrelu.py -v
$PYTHON layers/test_group_convolution.py -v
$PYTHON layers/test_group_convolution_strided.py -v
$PYTHON layers/test_indexing.py -v
$PYTHON layers/test_lookup_table.py -v
$PYTHON layers/test_matmul.py -v
$PYTHON layers/test_matmul_reuse_outputs.py -v
$PYTHON layers/test_maxpool.py -v
$PYTHON layers/test_relu.py -v
$PYTHON layers/test_scale.py -v
$PYTHON layers/test_sigmoid.py -v
$PYTHON layers/test_small_mobilenet.py -v
$PYTHON layers/test_softmax.py -v
$PYTHON layers/test_tanh.py -v
$PYTHON layers/test_tensordot.py -v
$PYTHON layers/test_transpose.py -v

echo "All PyTorch layer tests have finished"

popd
