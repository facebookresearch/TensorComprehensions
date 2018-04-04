Using TC with Caffe2
====================

**Author**: `Priya Goyal <https://github.com/prigoyal>`_

We provide *basic* integration of Tensor Comprehensions (TC) with Caffe2 for
*inference* purpose only. Using TC with PyTorch, you can express an
operator using Einstein notation and get the fast CUDA code for that layer with
just a few lines of code (examples below).

A **few cases** where TC can be useful:

* if you want to specialize your layer for input tensor sizes like (27, 23, 5, 3) unlike some specific sizes/architectures that have been heavily optimized.

* you are interested in fusing layers like group convolution, ReLU, FC.

* if you have a different new layer, let's call it :code:`hconv` (a variant of convolution), for which you wish you had an efficient kernel available.

TC makes its very trivial to get CUDA code for such cases and many more. By providing
TC integration with PyTorch, we hope to make it further easy for PyTorch users
to express their operations and bridge the gap between research and engineering.

Installation
------------

In order to use TC with Caffe2, we provide Caffe2 conda package that is compatible
with TC. We also provide the conda package for all TC dependencies using which
you can build TC from source with Caffe2 integration. For installation with
Caffe2 support, please see the instructions here: :ref:`installation_caffe2_integration`.

How it works
------------

For caffe2, TC provides a generic operator named :code:`TcOp` which accepts
a TC language, input variables name list, outputs variable list and executes the operator.
TC provides a dynamic library :code:`libtc_c2.so` that has this operator. This operator
can be loaded with Caffe2 using :code:`caffe2.python.dyndep` and we can use this operator
with Caffe2 then. See an example below:

Example
-------

For demonstration purpose, we will pick a simple example for :code:`matmul` layer


.. code-block:: python

    from caffe2.proto import caffe2_pb2
    from caffe2.python import core, workspace, dyndep
    dyndep.InitOpsLibrary(os.path.join(os.environ.get("CONDA_PREFIX"), "lib/libtc_c2.so"))

    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    mat1, mat2 = np.random.rand(100, 400), np.random.rand(400, 500)
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
        workspace.FeedBlob('mat1', mat1.astype(np.float32))
        workspace.FeedBlob('mat2', mat2.astype(np.float32))
        matmul = core.CreateOperator(
            "TcOp", ["mat1", "mat2"], ["out"], lang=lang, tcName="matmul"
        )
    workspace.RunOperatorOnce(matmul)
    out = workspace.FetchBlob("out")

Future
------

The integration with Caffe2 is very basic at the moment. We do not provide autotuner
support for Caffe2 at the moment and welcome contributions from the community.
