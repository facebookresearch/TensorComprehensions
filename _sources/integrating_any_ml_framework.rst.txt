.. _integrating_ml_frameworks:

Integrating TC with ML framework
================================

TC backend is agnostic to frameworks and is designed in a way such that the integration
with any ML framework can be easy. For our first release version, we provided
integration with PyTorch and basic integration with Caffe2 framework however, we
welcome the community to integrate TC with other frameworks.

TC backend is based on DLPack tensors which is very lightweight header library
for describing the tensor. A DLPack tensor is a simple struct which has information
like data pointer, tensor size, tensor strides, tensor type etc. DLPack doesn't do
any memory allocations and rather provides the meta information about the tensor.
Hence converting a tensor for example torch tensor to DLPack tensor doesn't involve
any copies and is very cheap.

Step 1: DLpack support in framework
-----------------------------------
In order to integrate a new framework, minimal DLPack support is needed. Two functions
are needed:

* :code:`toDlpack`: create the DLPack tensor struct from the tensor.

* :code:`fromDlpack`: create the tensor from the DLPack struct.

As an example, `ATen <https://github.com/zdevito/ATen/>`_ (a C++ tensor library)
has these functions which can be found `here <https://github.com/zdevito/ATen/blob/master/src/ATen/DLConvertor.h>`_.


Step 2: Integrating TC
----------------------
Once the DLPack support is available in the framework, integration of TC is easy.
This can be achieved by writing a lightweight C++ code which uses the DLPack tensor
conversion calls to convert tensors so that they can be passed to TC backend.

Further, since TC itself doesn't do any data allocations, it infers the output tensor
shapes for a given TC and input sizes. The output tensors shapes are sent back after the
compilation and framework needs to allocate storage for output tensors using that
information. This is all that is needed for integrating an ML framework with TC.
Concretely, following functions need to be defined:

* :code:`define`: This is simply a wrapper which takes the TC lang input and dispatches call to the TC backend. Nothing else is needed at this step.

* :code:`toDlpackTensors`: This should take the vector of input tensors (framework) and use the dlpack tensor conversions API defined by framework to convert input tensors to dlpack tensors.

* :code:`compile`: This takes the dlpack tensors converted in previous step and dispatches compilation call to TC backend on those input dlpack tensors.

* :code:`prepareOutputs`: TC backend send back the output tensors infor (strides, shapes, type etc.) and framework should allocate the outputs storage.

* :code:`run`: This simply dispatches the output tensor pointers to the TC backend and returns the outputs received.

As an example, we provide ATen tensor library integration in TC and the implementation
for above functions can be found in TC codebase. See `this <https://github.com/facebookresearch/TensorComprehensions/tree/master/src/aten>`_.
