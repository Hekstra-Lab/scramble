import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np
from scramble.model.mlp import Linear,MLP

class ScalingModel(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.image_mlp = MLP(width, depth)
        self.linear_in = Linear(width)
        self.mlp = MLP(width, depth, input_layer=False)
        self.linear_out = Linear(1)

    def forward(self, Imodel, Iobs, metadata, mask, sample_size=32):
        batch_size = Iobs.shape[0]
        p = mask.any(-2).squeeze(-1)
        p = p / p.sum(-1, keepdims=True)
        j = torch.multinomial(p, sample_size)
        i = torch.arange(batch_size, dtype=j.dtype, device=j.device)[:,None] * torch.ones_like(j)

        image_data = torch.concat((
            Imodel[i,j],
            Iobs[i,j],
            metadata[i,j],
        ), axis=-1)

        image_mask = mask[i,j]
        image_rep = self.image_mlp(image_data)
        image_rep = (image_mask * image_rep).sum([1, 2], keepdims=True)
        image_rep = image_rep / image_mask.sum([1, 2], keepdims=True)

        scale = self.linear_in(metadata)
        scale = scale + image_rep
        scale = self.mlp(scale)
        scale = self.linear_out(scale)

        scale = torch.where(mask, scale, 0.)
        return scale


