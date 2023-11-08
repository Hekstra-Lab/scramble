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



if __name__=='__main__':
    cell = gemmi.UnitCell(70., 70., 40., 90., 90., 120.)
    sg = gemmi.SpaceGroup("P 63")
    hkl = rs.utils.generate_reciprocal_cell(cell, 2.)

    import gemmi
    ops = [gemmi.Op("x,y,z")] + \
        gemmi.find_twin_laws(cell, sg, 1e-3, False)


    op = ops[-1]
    gemmi_hkl = np.array(list(map(op.apply_to_hkl, hkl)))

    test_hkl = Op(op)(torch.tensor(hkl)).detach().cpu().numpy()

    assert (test_hkl == gemmi_hkl).all()
