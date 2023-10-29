import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np
from scramble.model.mlp import Linear,MLP


class NormalLikelihood(torch.nn.Module):
    def forward(self, Ipred, Iobs, SigIobs):
        p = torch.distributions.Normal(Iobs, SigIobs)
        ll = p.log_prob(Ipred)
        return ll

