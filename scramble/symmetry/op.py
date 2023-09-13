import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np


class Op(torch.nn.Module):
    """
    Pytorch-ified gemmi.Op
    """
    def __init__(self, op, device=None):
        super().__init__()
        dtype = torch.float32
        if isinstance(op, str):
            op = gemmi.Op(op)

        self.gemmi_op = op

        self.rot = torch.nn.Parameter(
            torch.tensor(self.gemmi_op.rot, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.den = torch.nn.Parameter(
            torch.tensor(self.gemmi_op.DEN, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.identity = self.gemmi_op == 'x,y,z'

    def __str__(self):
        return f"Op({self.gemmi_op.triplet()})"

    def forward(self, hkl):
        if self.identity:
            return hkl
        dtype = hkl.dtype
        hkl = hkl.type(torch.float32)
        hkl = torch.floor_divide(hkl @ self.rot, self.den) 
        hkl = hkl.type(dtype)
        return hkl

