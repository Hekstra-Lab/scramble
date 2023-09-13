import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np

class SurrogatePosterior(torch.nn.Module):
    def __init__(self, reciprocal_asu, reindexing_ops=None, dtype=None, device=None, epsilon=1e-6):
        super().__init__()
        if reindexing_ops is None:
            from scramble.symmetry import Op
            reindexing_ops = [Op("x,y,z")]
        else:
            reindexing_ops = reindexing_ops
        self.reindexing_ops = torch.nn.ModuleList(reindexing_ops)
        self.reciprocal_asu = reciprocal_asu
        asu_size = self.reciprocal_asu.asu_size
        self._concentration = torch.nn.Parameter(
            torch.ones(asu_size, dtype=dtype, device=device)
        )
        self._rate = torch.nn.Parameter(
            torch.ones(asu_size, dtype=dtype, device=device)
        )
        self.epsilon = epsilon
        self.concentration_min = 1.

    @property
    def concentration(self):
        return torch.exp(self._concentration) + self.epsilon + self.concentration_min

    @property
    def rate(self):
        return torch.exp(self._rate) + self.epsilon

    def distribution(self):
        return torch.distributions.Gamma(self.concentration, self.rate)

    def prior(self):
        centric,multiplicity = self.reciprocal_asu.centric,self.reciprocal_asu.multiplicity
        concentration = torch.where(centric, 0.5, 1.)
        rate = torch.where(centric, 0.5 / multiplicity, 1. / multiplicity)
        return torch.distributions.Gamma(concentration, rate)

    def get_f_sigf(self):
        """
        X ~ Gamma(k, 1 / theta) => sqrt(X) ~ GenGamma(p=2, d=2k, a=sqrt(theta))
        GenGamma(x|p,d,a) =(1/Z) * x**(d-1) * exp(-(x/a)**p) 
        """
        k,b = self.concentration,self.rate
        theta = torch.reciprocal(b)
        p = 2
        d = 2 * k
        a = torch.sqrt(theta)

        from scipy.stats import gengamma
        """
        In scipy, 
        GenGamma(x|a,c) = (1/Z) * x**(ca-1) * exp(-(x)**c) 
        we have, 
         - scale = a
         - ca = d
         - c = p
        therefore,
         - a = d / c
        """
        scale = a
        c = a
        a = d / c
        F = gengamma.mean(a, c, scale=a)
        SigF = gengamma.std(a, c, scale=a)

    def get_f_sigf(self):
        """
        X ~ Gamma(k, 1 / theta) => sqrt(X) ~ GenGamma(p=2, d=2k, a=sqrt(theta))
        GenGamma(x|p,d,s) =(1/Z) * x**(d-1) * exp(-(x/s)**p) 
        """
        k,b = self.concentration,self.rate
        theta = torch.reciprocal(b)
        p = 2
        d = 2 * k
        s = torch.sqrt(theta)

        """
        In scipy, 
        GenGamma(x|a,c) = (1/Z) * x**(ca-1) * exp(-(x)**c) 
        we have, 
         - scale = s
         - ca = d
         - c = p
        therefore,
         - a = d / c
        """
        from scipy.stats import gengamma
        scale = s
        c = p
        a = d / c
        a = a.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()

        F = gengamma.mean(a, c, scale=scale)
        SigF = gengamma.std(a, c, scale=scale)

        return F,SigF
