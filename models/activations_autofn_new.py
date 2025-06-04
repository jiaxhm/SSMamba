import torch
from torch import nn as nn
from torch.nn import functional as F

__all__ = ["swish_auto", "SwishAuto", "mish_auto", "MishAuto", "LELUAuto"]


class SwishAutoFn(torch.autograd.Function):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    Memory efficient variant from:
     https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
    """

    @staticmethod
    def forward(ctx, x):
        result = x.mul(torch.sigmoid(x))
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        return grad_output.mul(x_sigmoid * (1 + x * (1 - x_sigmoid)))


def swish_auto(x, inplace=False):
    # inplace ignored
    return SwishAutoFn.apply(x)


class SwishAuto(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishAutoFn.apply(x)


class MishAutoFn(torch.autograd.Function):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    Experimental memory-efficient variant
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        x_tanh_sp = F.softplus(x).tanh()
        return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


def mish_auto(x, inplace=False):
    # inplace ignored
    return MishAutoFn.apply(x)


class MishAuto(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishAutoFn.apply(x)


#--------------------------2-------------------------
class LELUAutoFn(torch.autograd.Function):
    """LogExp: A custom activation function defined as
    f(x) = (x * log(1 + exp(x))) / (log(1 + exp(-x)) + log(1 + exp(x)))
    """

    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x, a, b)
        # Compute the numerator and denominator
        u = x.mul(F.softplus(a*x+b))  # x * log(1 + exp(x))
        v = F.softplus(-(a*x)+b) + F.softplus(a*x+b) # log(1 + exp(-x)) + log(1 + exp(x))####
        y = u / v
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b = ctx.saved_tensors
        # Compute the numerator and denominator
        u = x.mul(F.softplus(a*x+b))
        v = F.softplus(-a*x+b) + F.softplus(a*x+b)
        # Derivatives
        du_dx = F.softplus(a*x+b) + x.mul(torch.sigmoid(a*x+b)) * a  # Derivative of u
        dv_dx = -a * torch.sigmoid(-(a * x) + b) + torch.sigmoid(a*x+b) * a  # Derivative of v
        du_da = x.pow(2) * torch.sigmoid(a * x + b)
        dv_da = x.mul(torch.sigmoid(a * x + b) - torch.sigmoid(-a * x + b))
        du_db = x.mul(torch.sigmoid(a * x + b))
        dv_db = torch.sigmoid(-(a * x) + b) + torch.sigmoid(a * x + b)
        # Using the quotient rule for derivatives
        grad_input = grad_output.mul((du_dx * v - dv_dx * u) / (v ** 2))
        grad_a = grad_output.mul((du_da * v - dv_da * u) / (v ** 2))
        grad_b = grad_output.mul((du_db * v - dv_db * u) / (v ** 2))

        return grad_input, grad_a, grad_b


class LELUAuto(nn.Module):
    def __init__(self, inplace: bool = False):
        super(LELUAuto, self).__init__()
        self.inplace = inplace
        self.a = nn.Parameter(torch.Tensor([1.0]))
        self.b = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):

        return LELUAutoFn.apply(x, self.a, self.b)