import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np

class SurrogatePosterior(torch.nn.Module):
    def __init__(self, reciprocal_asu_collection, reindexing_ops=None, dtype=None, device=None, epsilon=1e-6):
        super().__init__()
        if reindexing_ops is None:
            from scramble.symmetry import Op
            reindexing_ops = [Op("x,y,z")]
        else:
            reindexing_ops = reindexing_ops
        self.reindexing_ops = torch.nn.ModuleList(reindexing_ops)
        self.reciprocal_asu_collection = reciprocal_asu_collection
        asu_size = self.reciprocal_asu_collection.asu_size
        self._log_concentration = torch.nn.Parameter(
            torch.ones(asu_size, dtype=dtype, device=device)
        )
        self._log_rate = torch.nn.Parameter(
            torch.ones(asu_size, dtype=dtype, device=device)
        )
        self.register_buffer(
            'epsilon',
            torch.tensor(epsilon, dtype=torch.float32),
        )
        self.register_buffer(
            'concentration_min',
            torch.tensor(1., dtype=torch.float32),
        )

    @property
    def rac(self):
        return self.reciprocal_asu_collection

    @property
    def concentration(self):
        return torch.exp(self._log_concentration) + self.epsilon + self.concentration_min

    @property
    def rate(self):
        return torch.exp(self._log_rate) + self.epsilon

    def distribution(self):
        return torch.distributions.Gamma(self.concentration, self.rate)

    def prior(self):
        centric,multiplicity = self.reciprocal_asu_collection.centric,self.reciprocal_asu_collection.multiplicity
        concentration = torch.where(centric, 0.5, 1.)
        rate = torch.where(centric, 0.5 / multiplicity, 1. / multiplicity)
        return torch.distributions.Gamma(concentration, rate)

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

    def to_datasets(self, seen=True):
        h,k,l = self.reciprocal_asu_collection.Hasu.detach().cpu().numpy().T
        q = self.distribution()
        I = q.mean.detach().cpu().numpy()
        SigI = q.stddev.detach().cpu().numpy()
        F,SigF = self.get_f_sigf()

        for asu_id,rasu in enumerate(self.reciprocal_asu_collection):
            idx = self.reciprocal_asu_collection.asu_ids == asu_id
            if seen:
                idx &= self.reciprocal_asu_collection.seen

            idx = idx.detach().cpu().numpy()

            cell = rasu.cell
            spacegroup = rasu.spacegroup
            q = self.distribution()
            out = rs.DataSet(
                {
                    'H' : rs.DataSeries(h[idx], dtype='H'),
                    'K' : rs.DataSeries(k[idx], dtype='H'),
                    'L' : rs.DataSeries(l[idx], dtype='H'),
                    'I' : rs.DataSeries(I[idx], dtype='J'),
                    'SIGI' : rs.DataSeries(SigI[idx], dtype='Q'),
                    'F' : rs.DataSeries(F[idx], dtype='F'),
                    'SIGF' : rs.DataSeries(SigF[idx], dtype='Q'),
                }, 
                merged=True, 
                cell=rasu.cell, 
                spacegroup=rasu.spacegroup,
            )
            out = out.set_index(["H", "K", "L"])
            if rasu.anomalous:
                anom_keys = [
                    'F(+)', 'SIGF(+)', 'F(-)', 'SIGF(-)', 
                    'I(+)', 'SIGI(+)', 'I(-)', 'SIGI(-)', 
                ]
                out.unstack_anomalous()[anom_keys]
            yield out

    def save_mtzs(self, prefix):
        for asu_id,ds in enumerate(self.to_datasets()):
            mtz_file = f"{prefix}_{asu_id}.mtz"
            ds.write_mtz(mtz_file)

