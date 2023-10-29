import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np
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


