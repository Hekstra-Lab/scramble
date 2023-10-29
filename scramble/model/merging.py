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
    @rs.decorators.cellify
    @rs.decorators.spacegroupify
    def __init__(self, surrogate_posterior, scaling_model, likelihood, expand_harmonics=None, kl_weight=1.):
        super().__init__()
        self.expand_harmonics = expand_harmonics
        self.kl_weight = kl_weight
        self.scaling_model = scaling_model
        self.surrogate_posterior = surrogate_posterior
        self.likelihood = likelihood

    def forward(
            self, 
            hkl, 
            I, 
            SigI, 
            image_id, 
            metadata, 
            wavelength=None, 
            dHKL=None, 
            mc_samples=32, 
            return_op=True, 
            return_cc=True
        ):
        _,image_id = torch.unique(image_id, return_inverse=True)
        harmonic_id = None

        Iscale = torch.concat((
            I,
            SigI,
        ), axis=-1)

        #from IPython import embed
        #embed(colors='linux')
        #XX
        harmonic_image_id = image_id
        if self.expand_harmonics is not None:
            convolved_metadata = [image_id, I, SigI]
            harmonic_metadata = [image_id, metadata, Iscale]
            (
                hkl, dHKL, wavelength, 
                (image_id, I, SigI),
                (harmonic_image_id, metadata, Iscale),
            ), harmonic_id = self.expand_harmonics(hkl, dHKL, wavelength, convolved_metadata, harmonic_metadata, rasu=self.surrogate_posterior.reciprocal_asu)

        metadata = torch.concat([metadata, wavelength, torch.reciprocal(torch.square(dHKL))], -1)

        q = self.surrogate_posterior.distribution()
        p = self.surrogate_posterior.prior()
        kl_div = torch.distributions.kl_divergence(q, p)
        kl_div = kl_div.mean()

        Imodel = torch.concat((
            q.mean[...,None],
            q.stddev[...,None],
        ), axis=-1)

        ll = []

        z = q.rsample((mc_samples,))
        Ipred = []
        for op in self.surrogate_posterior.reindexing_ops:
            _hkl = op(hkl)
            refl_id = self.surrogate_posterior.reciprocal_asu(_hkl)
            _Imodel = Imodel[refl_id]
            scale = self.scaling_model(
                _Imodel, 
                Iscale, 
                harmonic_image_id, 
                metadata, 
                sample_size=mc_samples
            )

            _Ipred = z.T[refl_id] * scale
            if self.expand_harmonics is not None:
                _Ipred = self.expand_harmonics.convolve_harmonics(
                    _Ipred, harmonic_id,
                )

            _ll = self.likelihood(_Ipred, I, SigI)
            _ll = self.scaling_model.image_model.sum_images(_ll, image_id) 
            _ll = _ll.mean(-1, keepdims=True)

            ll.append(_ll)
            Ipred.append(_Ipred.mean(-1, keepdims=True))

        ll = torch.concat(ll, axis=-1)
        #_,op_idx = ll.max(-1)
        #ll = torch.softmax(ll, axis=-1) * ll
        ll,op_idx = ll.max(-1)
        ll = ll.mean()
        Ipred = torch.concat(Ipred, axis=-1)



        elbo = -ll + self.kl_weight * kl_div 

        metrics = {
            'ELBO' : f'{elbo:0.2e}',
            'D_KL' : f'{kl_div:0.2e}',
        }

        if return_cc:
            Ipred = Ipred[torch.arange(len(Ipred)), op_idx[image_id].squeeze(-1)]

            w = torch.reciprocal(torch.square(SigI))
            cc = float(weighted_pearson(
                I.flatten(),
                Ipred.flatten(),
                w.flatten(),
            ))
            metrics['CCpred'] = f'{cc:0.2f}'

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

