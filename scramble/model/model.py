import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np
import math
from scramble.model.mlp import Linear,MLP


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
            (num_images, tensor.shape[-1]),
            device=device,
            dtype=float_dtype,
        )
        result = result.scatter_add(
            -2,
            torch.tile(image_id[:,None], (1, d)),
            tensor,
        )
        return result 

    def forward(self, data, image_id, sample_size=32):
        idx = self.sample_idx(image_id, sample_size)
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
        scale = scale + image_reps[image_id]
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
    def __init__(self, surrogate_posterior, scaling_model, likelihood, dtype=None, device=None, kl_weight=1.):
        super().__init__()
        self.kl_weight = kl_weight
        self.scaling_model = scaling_model
        self.surrogate_posterior = surrogate_posterior
        self.likelihood = likelihood

    def convolve(self, tensor, harmonic_id):
        device = tensor.device
        float_dtype = tensor.dtype
        harmonic_size = harmonic_id.max() + 1
        harmonic_id = harmonic_id[...,None] * torch.ones_like(tensor, dtype=torch.int)
        d = tensor.shape[-1]
        result = torch.zeros(
            (harmonic_size, d),
            device=device,
            dtype=float_dtype,
        )
        result = result.scatter_add(
            -2,
            harmonic_id,
            tensor,
        )
        return result

    def forward(self, hkl, I, SigI, image_id, metadata, harmonic_id=None, mc_samples=32, return_op=True):
        q = self.surrogate_posterior.distribution()
        p = self.surrogate_posterior.prior()
        kl_div = torch.distributions.kl_divergence(q, p)
        kl_div = kl_div.mean()
        device = I.device
        float_dtype = I.dtype

        Imodel = torch.concat((
            q.mean[...,None],
            q.stddev[...,None],
        ), axis=-1)
        Iscale = torch.concat((
            I[...,None],
            SigI[...,None],
        ), axis=-1)
        if harmonic_id is not None:
            Iscale = Iscale[harmonic_id]

        ll = []
        z = q.rsample((mc_samples,))
        for op in self.surrogate_posterior.reindexing_ops:
            _hkl = op(hkl)
            refl_id = self.surrogate_posterior.reciprocal_asu(_hkl)
            scale = self.scaling_model(Imodel[refl_id], Iscale, image_id, metadata, sample_size=mc_samples)

            _Ipred = z.T[refl_id] * scale
            if harmonic_id is not None:
                _Ipred = self.convolve(_Ipred, harmonic_id)

            _ll = self.likelihood(_Ipred, I[:,None], SigI[:,None])
            if harmonic_id is not None:
                _image_id = torch.ones_like(_ll[:,0], dtype=harmonic_id.dtype)
                _image_id[harmonic_id] = image_id
            else:
                _image_id = image_id
            _ll = self.scaling_model.image_model.sum_images(_ll, _image_id) / mc_samples
            _ll = _ll.mean(-1, keepdims=True)

            ll.append(_ll)
        ll = torch.concat(ll, axis=-1)
        ll,op_idx = ll.max(-1)
        ll = ll.mean()

        elbo = -ll + self.kl_weight * kl_div 
        if return_op:
            return elbo, op_idx
        return elbo

if __name__=="__main__":
    sg = gemmi.SpaceGroup("P 31 2 1")
    cell = gemmi.UnitCell(76.030, 76.030, 76.140, 90.00, 90.00, 120.00)
    ops = [gemmi.Op("x,y,z")] 
    ops.extend(gemmi.find_twin_laws(cell, sg, 1e-3, False))
    print(ops)

