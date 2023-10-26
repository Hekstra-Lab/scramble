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


class ImageModel(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.mlp = MLP(width, depth)

    def sample_idx(self, image_id, sample_size):
        """
        Make a batch of samples for the image representation model.
        This method doesn't guarantee an exact number of samples per image.
        Sample_size only represents the average number of samples per image.
        """
        p = torch.ones_like(image_id, dtype=torch.float32)
        counts = torch.bincount(image_id)
        num_images = image_id.max() + 1
        p = p / counts[image_id] / num_images
        sample_idx = torch.multinomial(p, num_images * sample_size)
        return sample_idx

    def sum_images(self, tensor, image_id, num_images=None):
        device = tensor.device
        float_dtype = tensor.dtype
        d = tensor.shape[-1]
        if num_images is None:
            num_images = image_id.max() + 1
        result = torch.zeros(
            (num_images, d),
            device=device,
            dtype=float_dtype,
        )
        idx = torch.tile(image_id, (1, d))
        result = result.scatter_add(
            0,
            idx,
            tensor,
        )
        return result 

    def forward(self, data, image_id, sample_size=32):
        idx = self.sample_idx(image_id.squeeze(-1), sample_size)
        samples = data[idx]
        out = self.mlp(samples)
        num_images = image_id.max() + 1
        out = self.sum_images(
            out, image_id[idx], num_images=num_images
        ) / sample_size
        return out


class ScalingModel(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.image_model = ImageModel(width, depth)
        self.linear_in = Linear(width)
        self.mlp = MLP(width, depth, input_layer=False)
        self.linear_out = Linear(1)

    def forward(self, Imodel, Iobs, image_id, metadata, sample_size=32):
        image_data = torch.concat((
            Imodel,
            Iobs,
            metadata,
        ), axis=-1)
        image_reps = self.image_model(image_data, image_id, sample_size)

        scale = self.linear_in(metadata)
        scale = scale + image_reps[image_id.squeeze(-1)]
        scale = self.mlp(scale)
        scale = self.linear_out(scale)
        return scale

class NormalLikelihood(torch.nn.Module):
    def forward(self, Ipred, Iobs, SigIobs):
        p = torch.distributions.Normal(Iobs, SigIobs)
        ll = p.log_prob(Ipred)
        return ll

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

    def forward(self, hkl, I, SigI, image_id, metadata, wavelength=None, dHKL=None, mc_samples=32, return_op=True, return_cc=True):
        _,image_id = torch.unique(image_id, return_inverse=True)
        q = self.surrogate_posterior.distribution()
        p = self.surrogate_posterior.prior()
        kl_div = torch.distributions.kl_divergence(q, p)
        kl_div = kl_div.mean()
        harmonic_id = None

        Iscale = torch.concat((
            I,
            SigI,
        ), axis=-1)

        ll_idx = None
        if self.expand_harmonics is not None:
            (hkl, dHKL, wavelength, metadata, harmonic_image_id, Iscale), harmonic_id = self.expand_harmonics(hkl, dHKL, wavelength, metadata, image_id, Iscale)
            ll_idx = torch.ones(I.shape, device=I.device, dtype=torch.bool)
            ll_idx[harmonic_id] = False
        else:
            harmonic_image_id = image_id
        metadata = torch.concat([metadata, wavelength, torch.reciprocal(torch.square(dHKL))], -1)

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
            scale = self.scaling_model(
                Imodel[refl_id], 
                Iscale, 
                harmonic_image_id, 
                metadata, 
                sample_size=mc_samples
            )

            _Ipred = z.T[refl_id] * scale
            if self.expand_harmonics is not None:
                _Ipred = self.expand_harmonics.convolve_harmonics(
                    _Ipred, harmonic_id, len(I),
                )

            _ll = self.likelihood(_Ipred, I, SigI)
            if ll_idx is not None:
                _ll[ll_idx.squeeze(-1)] = 0.
            _ll = self.scaling_model.image_model.sum_images(_ll, image_id) 
            _ll = _ll.mean(-1, keepdims=True)

            ll.append(_ll)
            Ipred.append(_Ipred.mean(-1, keepdims=True))

        ll = torch.concat(ll, axis=-1)
        ll,op_idx = ll.max(-1)
        ll = ll.mean()
        Ipred = torch.concat(Ipred, axis=-1)

        Ipred = Ipred[torch.arange(len(Ipred)), op_idx[image_id].squeeze(-1)]

        w = torch.reciprocal(torch.square(SigI))
        if ll_idx is not None:
            w[ll_idx.squeeze(-1)] = 0.
        cc = float(weighted_pearson(
            I.flatten(),
            Ipred.flatten(),
            w.flatten(),
        ))


        elbo = -ll + self.kl_weight * kl_div 
        out = (elbo,)

        metrics = {
            'ELBO' : f'{elbo:0.2e}',
            'D_KL' : f'{kl_div:0.2e}',
            'CCpred' : f'{cc:0.2f}',
        }
        if return_cc:
            out = out + (metrics,)
        if return_op:
            out = out + (op_idx,)

        return out

if __name__=="__main__":
    sg = gemmi.SpaceGroup("P 31 2 1")
    cell = gemmi.UnitCell(76.030, 76.030, 76.140, 90.00, 90.00, 120.00)
    ops = [gemmi.Op("x,y,z")] 
    ops.extend(gemmi.find_twin_laws(cell, sg, 1e-3, False))
    print(ops)

