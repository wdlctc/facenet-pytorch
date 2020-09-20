import torch
import copy
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize

def shift(x, max_entry=None, fl_mode="ceil"):
    """ right shift tensor
    """
    if max_entry is None:
        max_entry = x.abs().max()

    if fl_mode == "ceil":
        return x / (2.0 ** torch.ceil(torch.log2(max_entry))), max_entry
    else:
        return x / (2.0 ** torch.round(torch.log2(max_entry))), max_entry

def rebase(x, max_entry, fl_mode="ceil"):
    """ left shift tensor
    """
    if fl_mode == "ceil":
        return x * (2.0 ** torch.ceil(torch.log2(max_entry)))
    else:
        return x * (2.0 ** torch.round(torch.log2(max_entry)))

def QG(x, max_entry=None, bits_G=8, fl_mode="ceil", mode="nearest"):
    x, max_entry = shift(x, max_entry)
    norm = fixed_point_quantize(
        x, wl=bits_G, fl=bits_G - 1, clamp=True, symmetric=True, rounding=mode
    )
    output = rebase(norm, max_entry)
    return output

def quantizer():
    class Rounding(torch.autograd.Function):
        @staticmethod
        def forward(self, x, max_entry):

            out = QG(x,max_entry)

            return out

        @staticmethod
        def backward(self, grad_output):
            return grad_output
    return Rounding.apply

class Quantizer(nn.Module):
    def __init__(
        self,
        device = None
    ):
        super(Quantizer, self).__init__()
        self.quantize = quantizer()
        self.flag = False
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)
        self.max_entry = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))

    def forward(self, x):
        if self.flag is True:
            return self.quantize(x, self.max_entry)
        else:
            self.max_entry.data = torch.max(self.max_entry.data, x.abs().max())
            return x

    def set_flag(self):
        self.flag = True

    def reset_flag(self):
        self.flag = False

    def get_max_entry(self):
        return self.max_entry

    def get_fl(self):
        return torch.ceil(torch.log2(elf.max_entry))


def _set_quant_func(quant):
    def _set_LP_layer(module):
        if type(module) is type(quant):
            module.set_flag()
        else:
            return

    return _set_LP_layer

def lower(
    model
):
    quant = Quantizer()
    lower_func = _set_quant_func(quant)
    model.apply(lower_func)

def _show_quant_func(quant):
    def _set_LP_layer(module):
        if type(module) is type(quant):
            print(module)
            print(module.flag)
            print(module.max_entry)
        else:
            return

    return _set_LP_layer

def show(
    model
):
    quant = Quantizer()
    show_func = _show_quant_func(quant)
    model.apply(show_func)
