import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np
from scramble.symmetry import Op

class ReciprocalASUCollection(torch.nn.Module):
    def __init__(self, reciprocal_asus, dtype=None, device=None):
        super().__init__()
        self.asu_size = 0
        self.hmax = 0
        self.reciprocal_asus = torch.nn.ModuleList(reciprocal_asus)
        for rasu in reciprocal_asus:
            self.asu_size += rasu.asu_size
            self.hmax = np.maximum(self.hmax, rasu.hmax).tolist()

        miller_id = []
        dHKL = []
        asu_id = []
        asu_start = 0
        for rasu in reciprocal_asus:
            asu_miller_id = torch.arange(len(rasu.Hasu), device=device, dtype=torch.int)
            asu_miller_id = asu_miller_id + asu_start
            miller_id.append(
                rasu.to_voxel_grid(asu_miller_id, self.hmax)
            )

            asu_dHKL = rs.utils.compute_dHKL(rasu.Hasu, rasu.cell)
            dHKL.append(
                rasu.to_voxel_grid(
                    torch.tensor(asu_dHKL, dtype=dtype), 
                    self.hmax
                )
            )
            asu_start = asu_start + rasu.asu_size

        miller_id = torch.stack(miller_id)
        dHKL = torch.stack(dHKL)
        self.register_buffer('miller_id', miller_id)
        self.register_buffer('dHKL', dHKL)
        self.register_buffer(
            'seen',
            torch.zeros((self.asu_size,), dtype=torch.bool),
        )

    @property
    def centric(self):
        return torch.concat([rasu.centric for rasu in self.reciprocal_asus])

    @property
    def multiplicity(self):
        return torch.concat([rasu.multiplicity for rasu in self.reciprocal_asus])

    @property
    def dmin(self):
        out = [rasu.dmin for rasu in self.reciprocal_asus]
        device = out[0].device
        dtype = out[0].dtype
        out = torch.tensor(out, device=device, dtype=dtype)
        return out

    @property
    def Hasu(self):
        out = torch.concat([rasu.Hasu for rasu in self], axis=0)
        return out

    @property
    def asu_ids(self):
        device = self.miller_id.device
        dtype = self.miller_id.dtype
        out = torch.concat(
            [asu_id*torch.ones((rasu.asu_size,), dtype=dtype, device=device) for asu_id,rasu in enumerate(self)], 
        )
        return out

    @property
    def Hasu(self):
        out = torch.concat([rasu.Hasu for rasu in self], axis=0)
        return out

    def __iter__(self):
        return self.reciprocal_asus.__iter__()

    def __next__(self):
        return self.reciprocal_asus.__next__()

    def __len__(self):
        return len(self.reciprocal_asus)

    def forward(self, asu_id, hkl):
        h,k,l = hkl[...,0], hkl[...,1], hkl[...,2]
        asu_id = asu_id.squeeze(-1)
        out = self.miller_id[asu_id, h, k, l]
        self.seen[out] = True
        return out

    def compute_dHKL(self, asu_id, hkl):
        h,k,l = hkl[...,0], hkl[...,1], hkl[...,2]
        asu_id = asu_id.squeeze(-1)
        return self.dHKL[asu_id, h, k, l][...,None]


class ReciprocalASU(torch.nn.Module):
    @rs.decorators.cellify
    @rs.decorators.spacegroupify
    def __init__(self, cell, spacegroup, dmin, anomalous=False, dtype=None):
        super().__init__()
        hmax = cell.get_hkl_limits(dmin)
        Hasu = rs.utils.generate_reciprocal_asu(cell, spacegroup, dmin, anomalous)
        self.register_buffer(
            'hmax',
            torch.tensor(hmax, dtype=torch.int),
        )
        self.register_buffer(
            '_cell',
            torch.tensor([cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma], dtype=dtype),
        )
        self.register_buffer(
            '_spacegroup',
            torch.tensor(list(map(ord, spacegroup.xhm()))),
        )
        self.anomalous = anomalous
        self.register_buffer(
            'dmin',
            torch.tensor(dmin),
        )
        self.register_buffer(
            'Hasu',
            torch.tensor(
                Hasu, 
                dtype=torch.int,
            ),
        )
        self.register_buffer(
            'centric',
            torch.tensor(
                rs.utils.is_centric(self.Hasu, self.spacegroup),
                dtype=torch.bool,
            ),
        )
        self.register_buffer(
            'multiplicity',
            torch.tensor(
                rs.utils.compute_structurefactor_multiplicity(self.Hasu, self.spacegroup),
                dtype=torch.float32,
            ),
        )
        miller_id = self.to_voxel_grid(torch.arange(len(self.Hasu), dtype=torch.int))
        self.register_buffer(
            'miller_id',
            miller_id,
        )
        dHKL = rs.utils.compute_dHKL(Hasu, cell)
        dHKL = self.to_voxel_grid(torch.tensor(dHKL, dtype=dtype))
        self.register_buffer(
            'dHKL',
            dHKL,
        )
        self.register_buffer(
            'seen',
            torch.zeros((self.asu_size,), dtype=torch.bool),
        )

    @property
    def asu_size(self):
        return len(self.Hasu)

    @property
    def ops(self):
        return [Op(op) for op in self.spacegroup.operations()]

    @property
    def cell(self):
        return gemmi.UnitCell(*self._cell)

    @property
    def spacegroup(self):
        return gemmi.SpaceGroup(''.join(map(chr, self._spacegroup.detach().cpu().numpy())))

    def is_absent(self, hkl):
        self.spacegroup.operations().systematic_absences

    def to_voxel_grid(self, asu_values, hkl_limits=None, fill_value=-1):
        dtype = asu_values.dtype
        device = asu_values.device
        if hkl_limits is None:
            hkl_limits = self.hmax
        h,k,l = hkl_limits
        voxel_grid = fill_value * torch.ones(
            (2*h + 1, 2*k + 1, 2*l + 1), 
            dtype=dtype, device=device,
        )
        for op in self.ops:
            Hop = op(self.Hasu)
            h,k,l = Hop.T
            voxel_grid[h, k, l] = asu_values

            #Now do Friedel
            if not self.anomalous:
                h,k,l = -Hop.T
                voxel_grid[h, k, l] = asu_values
        return voxel_grid

    def to_refl_id(self, hkl):
        h,k,l = hkl[...,0],hkl[...,1],hkl[...,2]
        out = self.miller_id[h, k, l]
        return out

    def forward(self, hkl):
        out = self.to_refl_id(hkl)
        assert (out >= 0).all()
        #TODO: use appropriate context manager to only update during training
        self.seen[out] = True
        return out

    def compute_dHKL(self, hkl):
        h,k,l = hkl[...,0],hkl[...,1],hkl[...,2]
        out = self.dHKL[h, k, l]
        return out


if __name__=="__main__":
    cell = [34., 45., 98., 90., 90., 90.]
    sg = 19
    dmin = 1.8
    anomalous = False
    rasu = ReciprocalASU(cell, sg, dmin, anomalous=anomalous)

    r1 = ReciprocalASU(cell, sg, 5., anomalous=anomalous)
    r2 = ReciprocalASU(cell, sg, 2., anomalous=anomalous)
    rac = ReciprocalASUCollection((r1, r2))
    from IPython import embed
    embed(colors='linux')
