import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils.cpp_extension import load

# import group_linear
# group_linear = load(name="group_linear", sources=["src/group_linear/group_linear.cpp"], verbose=True)


# class GroupLinearFunction(torch.autograd.Function):
#     """ GroupLinear <Python wrapper for C++ extension>
#     """
#     @staticmethod
#     def forward(ctx, input, weight, bias=None):
#         output = group_linear.forward(input, weight, bias)
#         ctx.save_for_backward(input, weight, bias)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = group_linear.backward(
#             grad_output.contiguous(), *ctx.saved_tensors)
#         grad_input, grad_weight, grad_bias = output
#         return grad_input, grad_weight, grad_bias

class GroupLinearFunction(torch.autograd.Function):
    """ GroupLinear <Python>
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.bmm(weight.transpose(-1, -2))  # use `bmm` instead of `mm` to parallelize a group of MLPs
        if bias is not None:
            output += bias.unsqueeze(1).expand_as(output)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.bmm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(1,2).bmm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(1)
        return grad_input, grad_weight, grad_bias

class GroupLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 num_group=1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_group = num_group
        self.weight = Parameter(torch.empty((num_group, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(num_group, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return GroupLinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'num_group={}, in_features={}, out_features={}, bias={}, dtype={}'.format(
            self.num_group, self.in_features, self.out_features, self.bias is not None, self.weight.dtype
        )


# class LinearFunction(torch.autograd.Function):
#     """ Linear <Python wrapper for C++ extension>
#     """
#     @staticmethod
#     def forward(ctx, input, weight, bias=None):
#         output = group_linear.forward(input, weight, bias)
#         ctx.save_for_backward(input, weight, bias)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = group_linear.backward(
#             grad_output.contiguous(), *ctx.saved_tensors)
#         grad_input, grad_weight, grad_bias = output
#         return grad_input, grad_weight, grad_bias

class LinearFunction(torch.autograd.Function):
    """ Linear <Python>
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return GroupLinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == '__main__':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    
    input = (torch.randn(2,10,3,dtype=torch.double,requires_grad=True), torch.randn(2,64,3,dtype=torch.double,requires_grad=True), torch.randn(2,64,dtype=torch.double,requires_grad=True))
    test = gradcheck(GroupLinearFunction.apply, input, eps=1e-6, atol=1e-4)
    print('GroupLinear', test)

    input = (torch.randn(10,3,dtype=torch.double,requires_grad=True), torch.randn(64,3,dtype=torch.double,requires_grad=True))
    test = gradcheck(LinearFunction.apply, input, eps=1e-6, atol=1e-4)
    print('Linear:', test)

    