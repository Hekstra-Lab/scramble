import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np

class ReciprocalASU(torch.nn.Module):
    def __init__(self, cell, spacegroup, dmin, anomalous=False, dtype=None, device=None):
        super().__init__()
        self.Hasu = rs.utils.generate_reciprocal_asu(cell, spacegroup, dmin, anomalous)
        self.asu_size = len(self.Hasu)
        self.hmax = cell.get_hkl_limits(dmin)
        h,k,l = self.hmax
        miller_id = -np.ones((2*h + 1, 2*k + 1, 2*l + 1), dtype='int')
        self.spacegroup = spacegroup
        self.centric = torch.nn.Parameter(
            torch.tensor(
                rs.utils.is_centric(self.Hasu, self.spacegroup),
                dtype=torch.bool,
                device=device,
            ),
            requires_grad=False,
        )
        self.multiplicity = torch.nn.Parameter(
            torch.tensor(
                rs.utils.compute_structurefactor_multiplicity(self.Hasu, self.spacegroup),
                dtype=torch.float32,
                device=device
            ),
            requires_grad=False,
        )

        for op in spacegroup.operations():
            Hop = rs.utils.apply_to_hkl(self.Hasu, op)
            h,k,l = Hop.T
            miller_id[h, k, l] = np.arange(len(self.Hasu))

            #Now do Friedel
            if not anomalous:
                h,k,l = -Hop.T
                miller_id[h, k, l] = np.arange(len(self.Hasu))

        self.miller_id = torch.nn.Parameter(
            torch.tensor(miller_id, dtype=torch.int, device=device),
            requires_grad=False,
        )

    def forward(self, hkl):
        h,k,l = hkl[...,0],hkl[...,1],hkl[...,2]
        return self.miller_id[h, k, l]



