Performance of TC
=================

TC can generate competitive code in a variety of cases thanks to its
Autotuner (see our companion paper: `arXiv <https://arxiv.org/abs/1802.04730>`_).
We will provide a set of benchmarks to illustrate the cases in
which it is recommended to use TC.

As a general rule of thumb, TC is a good candidate to rapidly prototype new
ML layers and integrate them without writing a single line of CUDA code.
For existing, computation bound layers, it should be expected that TC
performance will not beat libraries such as CUBLAS and CUDNN except in very
specific corner cases, described in our paper.

For the cases where efficient library implementations exist (e.g. matmul,
convolutions), it is usually recommended to use existing libraries, for now.
