# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import torch, pdb
from torch.autograd import Variable, Function


def get_tensors(inputs):
    tensor_inps = []
    for inp in inputs:
        if isinstance(inp, tuple):
            tensor_inps.append(torch.randn(inp).cuda())
        elif isinstance(inp, Variable):
            tensor_inps.append(inp.data)
        elif torch.is_tensor(inp):
            tensor_inps.append(inp)
        else:
            raise RuntimeError("Unsupported input type: ", type(inputs).__name__)
    return tensor_inps


def wrap_variable(inputs):
    if torch.is_tensor(inputs):
        return Variable(inputs)
    elif isinstance(inputs, tuple):
        return tuple(wrap_variable(v) for v in inputs)
    elif isinstance(inputs, list):
        return [wrap_variable(v) for v in inputs]
    else:
        raise RuntimeError("Unsupported input type: ", type(inputs).__name__)


def unpack_variables(inputs):
    if isinstance(inputs, Variable):
        return inputs.data
    elif torch.is_tensor(inputs):
        return inputs
    elif isinstance(inputs, tuple):
        return tuple(unpack_variables(v) for v in inputs)
    elif isinstance(inputs, list):
        return [unpack_variables(v) for v in inputs]
    else:
        raise RuntimeError("Unsupported input type: ", type(inputs).__name__)


# TC doesn't support strided tensors yet, so we have to make inputs contiguous
def make_contiguous(inputs):
    if isinstance(inputs, Variable) or torch.is_tensor(inputs):
        return inputs.contiguous()
    elif isinstance(inputs, tuple):
        return tuple(make_contiguous(v) for v in inputs)
    elif isinstance(inputs, list):
        return [make_contiguous(v) for v in inputs]
    else:
        raise RuntimeError("Unsupported input type: ", type(inputs).__name__)


class TCFunction(Function):

    @staticmethod
    def forward(ctx, tc_unit, tc_info, kwargs, *inputs):
        ctx.tc_unit, ctx.tc_info, ctx.kwargs = tc_unit, tc_info, kwargs
        ctx.save_for_backward(*inputs)
        outputs = unpack_variables(tc_info["outputs"]) if "outputs" in tc_info else None
        outputs = tc_unit.run(
            tc_info["forward_name"],
            make_contiguous(list(inputs)), outputs=outputs
        )
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        real_inputs, kwargs = ctx.saved_variables, ctx.kwargs
        tc_unit, tc_info = ctx.tc_unit, ctx.tc_info
        kwargs["type"] = "backward"
        reorder_function = kwargs["reorder_function"] if "reorder_function" in kwargs else None
        rearranged_grad_outputs = list(grad_outputs)
        if reorder_function is not None:
            rearranged_grad_outputs = reorder_function(list(grad_outputs))
        inputs = make_contiguous(unpack_variables(list(real_inputs) + list(rearranged_grad_outputs)))

        # if backwards hasn't been compiled before, we compile it  again
        if "compiled_backward" not in tc_info:
            tc_unit.compile(tc_info["backward_name"], inputs, **kwargs)
            tc_info["compiled_backward"] = True
        grad_inputs = tc_unit.run(tc_info["backward_name"], inputs)
        return (None, None, None,) + tuple(wrap_variable(grad_inputs))
