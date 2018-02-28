Autograd with TC
================

We provide the TC intergation with PyTorch `autograd` so that it is easy to write
a training layer with TC and be able to run backwards as well if the layer is part
of a network. In order to write a training layer with TC, you need to follow the
steps below:

1. Define your TC language that has two definitions: one for the forward layer and the other for the backward layer and pass it to :code:`tc.define` call. In addition, also pass :code:`training=True` and the name of the backward TC :code:`backward`.

2. Create the Input Variables and Parameters. For example, weights should be marked as Parameters and the inputs tensors as Variables.

3. Run the layer and get the output of forward pass.

4. To see that the backward call works fine, you can call backward on the outputs.

Let's see one example to demonstrate the steps above:

Examples
--------

.. code-block:: python

     import tensor_comprehensions as tc
     import torch
     from torch.autograd import Variable
     from torch.nn.parameter import Parameter
     CONV_LANG = """
     def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {{
        O(n, m, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(m, c, kh, kw)
     }}
     def convolution_grad(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(N,M,H,W) O_grad) -> (I_grad, W1_grad) {{
        I_grad(n, c, h, w) +=! O_grad(n, m, {sh} * h - kh, {sw} * w - kw) * W1(m, c, kh, kw)
        W1_grad(m, c, kh, kw) +=! O_grad(n, m, {sh} * h - kh, {sw} * w - kw) * I(n, c, h, w)
     }}
     """
     N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
     convolution = tc.define(CONV_LANG, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
     I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
     W = Parameter(torch.randn(O, C, kH, kW).cuda())
     out = convolution(I, W)
     out[0].sum().backward()


Specifying Mapping Options
--------------------------

We highly recommend passing the mapping options when running the kernel.
See :ref:`must_pass_options` for more details. When running the training layer,
you can pass the options for forward and backward layer separately or you can
pass the same options for them. In case you want to pass different options for
them, the example for that would be:

.. code-block:: python

     import tensor_comprehensions as tc
     import torch
     from torch.autograd import Variable
     from torch.nn.parameter import Parameter
     CONV_LANG = """
     def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {{
        O(n, m, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(m, c, kh, kw)
     }}
     def convolution_grad(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(N,M,H,W) O_grad) -> (I_grad, W1_grad) {{
        I_grad(n, c, h, w) +=! O_grad(n, m, {sh} * h - kh, {sw} * w - kw) * W1(m, c, kh, kw)
        W1_grad(m, c, kh, kw) +=! O_grad(n, m, {sh} * h - kh, {sw} * w - kw) * I(n, c, h, w)
     }}
     """
     N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
     convolution = tc.define(CONV_LANG, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
     I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
     W = Parameter(torch.randn(O, C, kH, kW).cuda())
     out = convolution(I, W, options=[tc.Options("conv"), tc.Options("group_conv")])
     out[0].sum().backward()

In order to obtain options via autotuning for backward and forward layer, keep reading further.


Autotuning training layer
-------------------------

You can autotune a training layer easily. The forward and backward layers will
be tuned separately in order to ensure maximal performance. Please read :ref:`pytorch_autotune_layers`
for how to set autotuner parameters. We will see how to autotune a training
layer, save cache and run the layer with help of examples:

You can either cache to default options or to a file (also see :ref:`autotuner_cache_choices`).
Let's see how to cache options to file when we tune a training layer.

.. code-block:: python

     import tensor_comprehensions as tc
     import torch
     CONV_LANG = """
     def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {{
        O(n, m, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(m, c, kh, kw)
     }}
     def convolution_grad(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(N,M,H,W) O_grad) -> (I_grad, W1_grad) {{
        I_grad(n, c, h, w) +=! O_grad(n, m, {sh} * h - kh, {sw} * w - kw) * W1(m, c, kh, kw)
        W1_grad(m, c, kh, kw) +=! O_grad(n, m, {sh} * h - kh, {sw} * w - kw) * I(n, c, h, w)
     }}
     """
     N, C, H, W, O, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
     convolution = tc.define(CONV_LANG, training=True, name="convolution", backward="convolution_grad", constants={"sh":sH, "sw":sW})
     I, W1 = torch.randn(N, C, H, W).cuda(), torch.randn(O, C, kH, kW).cuda()
     convolution.autotune(I, W, cache="convolution_train.tc")
     out = convolution(I, W)
     out[0].sum().backward()

You will find two cache files created: :code:`convolution_train.cuda/options` has
options for the forward layer and :code:`convolution_train_backward.cuda/options` file
has options for the grad layer.

Reordering grad outputs
-----------------------

In the backward pass, TC uses the list of input tensors in the forward pass and appends
the output tensors list to it. This is treated as the input to the backward TC definition.
However, sometimes, the forward layer TC might have some temporary variable for which we don't
need gradient in the backward TC. In such cases, users can use :code:`reorder_function`. See
the example below for how to use it:

.. code-block:: python

     import tensor_comprehensions as tc
     import torch
     LANG = """
     def convolution(float(N, C, H, W) I, float(M, C, KH, KW) W1, float(M) B) -> (tmp, O) {
       tmp(n, m, h, w) +=! I(n, c, h + kh, w + kw) * W1(m, c, kh, kw)
       O(n, m, h, w) = tmp(n, m, h, w) + B(m)
     }
     def convolution_grad(float(N, C, H, W) I, float(M, C, KH, KW) W1, float(M) B, float(N, M, H, W) O_grad)
     -> (I_grad, W1_grad, B_grad) {
       I_grad(n, c, h, w) +=! O_grad(n, m, h - kh, w - kw) * W1(m, c, kh, kw)
       W1_grad(m, c, kh, kw) +=! O_grad(n, m,  h - kh, w - kw) * I(n, c, h, w)
       B_grad(m) +=! O_grad(n, m, h, w)
     }
     """

     # since the forward layer produces two outputs, one is temporary which is
     # not needed in the forward pass, we can reorder the grad_outputs as we want.
     # So, here we return the output grad that we actually use in backwards TC.
     def reorder():
         def reorder_function(grad_outputs):
             return [grad_outputs[1]]
         return reorder_function

     N, C, H, W, M, kH, kW, sH, sW = 32, 4, 56, 56, 16, 1, 1, 1, 1
     convolution = tc.define(LANG, training=True, name="convolution", backward="convolution_grad")
     I = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
     W = Parameter(torch.randn(M, C, kH, kW).cuda())
     B = Parameter(torch.randn(M).cuda())
     out = convolution(I, W, B, reorder_function=reorder())
     out[0].sum().backward()
