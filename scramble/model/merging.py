import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np
from scramble.model.mlp import Linear,MLP


def weighted_pearson(x, y, w):
    z = torch.reciprocal(w.sum(-1))

    mx = z * (w * x).sum(-1)
    my = z * (w * y).sum(-1)

    dx = x - mx[...,None]
    dy = y - my[...,None]

    cxy = z * (w * dx * dy).sum(-1)
    cx = z * (w * dx * dx).sum(-1)
    cy = z * (w * dy * dy).sum(-1)

    r = cxy / torch.sqrt(cx * cy)
    return r

class MergingModel(torch.nn.Module):
    def __init__(self, surrogate_posterior, scaling_model, likelihood, expand_harmonics=None, kl_weight=1.):
        super().__init__()
        self.expand_harmonics = expand_harmonics
        self.kl_weight = kl_weight
        self.scaling_model = scaling_model
        self.surrogate_posterior = surrogate_posterior
        self.likelihood = likelihood

    def forward(
            self, 
            asu_id,
            hkl, 
            I, 
            SigI, 
            metadata, 
            wavelength, 
            mc_samples=32, 
            return_op=True, 
            return_cc=True
        ):

        q = self.surrogate_posterior.distribution()
        z = q.rsample((mc_samples,))
        p = self.surrogate_posterior.prior()
        kl_div = torch.distributions.kl_divergence(q, p)
        kl_div = kl_div.mean()

        Imodel = torch.concat((
            q.mean[...,None],
            q.stddev[...,None],
        ), axis=-1)

        Iscale = torch.concat((
            I,
            SigI,
        ), axis=-1)

        ll = []
        Ipred = []

        for op in self.surrogate_posterior.reindexing_ops:
            with torch.no_grad():
                _hkl = op(hkl)
                _hkl,_wavelength,_dHKL,refl_id = self.expand_harmonics(asu_id, _hkl, wavelength)
                mask = refl_id >= 0

            d_inv = torch.zeros_like(_dHKL)
            pos = d_inv > 0.
            d_inv[pos] = torch.reciprocal(torch.square(_dHKL[pos]))
            _metadata = torch.concat((
                metadata[...,None,:] * torch.ones_like(_dHKL),
                d_inv,
                _wavelength,
            ), axis=-1)

            scale = self.scaling_model(
                Imodel[refl_id.squeeze(-1)],
                Iscale[...,None,:] * torch.ones_like(_dHKL), 
                _metadata,
                mask,
                sample_size=mc_samples
            )

            _Ipred = torch.zeros(
                scale.shape[:-1] + (mc_samples,),
                dtype=scale.dtype,
                device=scale.device,
            )
            _Ipred[mask.squeeze(-1)] = z.T[refl_id[mask]] * scale[mask.squeeze(-1)]
            _Ipred = _Ipred.sum(-2) #This unassuming line is harmonic deconvolution
            mask = mask.any(-2).squeeze(-1)

            _ll = torch.zeros_like(_Ipred)
            _ll[mask] = self.likelihood(_Ipred[mask], I[mask], SigI[mask])
            _ll = _ll.mean(-1) #Expected log likelihood averages over samples
            _ll =  _ll.sum(-1) / mask.sum(-1) #Average per image
            _Ipred = _Ipred.mean(-1)

            ll.append(_ll)
            Ipred.append(_Ipred)

        ll = torch.dstack(ll)
        ll,op_idx = ll.max(-1)
        op_idx = op_idx.squeeze(0)
        ll = ll.mean()


        elbo = -ll + self.kl_weight * kl_div 

        metrics = {
            'ELBO' : float(elbo),
            'D_KL' : float(kl_div),
        }

        if return_cc:
            with torch.no_grad():
                Ipred = torch.dstack(Ipred)
                batch_idx = torch.arange(I.shape[0], dtype=op_idx.dtype, device=op_idx.device)
                Ipred = Ipred[batch_idx, :, op_idx]

                w = torch.zeros_like(SigI)
                w[mask] = torch.reciprocal(torch.square(SigI[mask]))
                cc = weighted_pearson(
                    I.flatten(),
                    Ipred.flatten(),
                    w.flatten(),
                )
                metrics['CCpred'] = float(cc)

        out = (elbo, metrics)
        if return_op:
            out = out + (op_idx,)

        return out


if __name__=="__main__":
    sg = gemmi.SpaceGroup("P 31 2 1")
    cell = gemmi.UnitCell(76.030, 76.030, 76.140, 90.00, 90.00, 120.00)
    ops = [gemmi.Op("x,y,z")] 
    ops.extend(gemmi.find_twin_laws(cell, sg, 1e-3, False))
    print(ops)

