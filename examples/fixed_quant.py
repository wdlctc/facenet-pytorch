import torch
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Function
from qtorch.quant import fixed_point_quantize

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
        def forward(self, x):

            out = QG(x)

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
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        return self.quantize(x)
