import numpy as np
import reciprocalspaceship as rs
import torch

class ExpandHarmonics(torch.nn.Module):
    def __init__(self, rac, wavelength_min=0., wavelength_max=torch.inf, max_multiplicity=5):
        super().__init__()
        self.rac = rac
        self.register_buffer(
            'wavelength_min',
            torch.tensor(
                wavelength_min,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            'wavelength_max',
            torch.tensor(
                wavelength_max,
                dtype=torch.float32,
            ),
        )
        self.register_buffer('max_multiplicity', torch.tensor(max_multiplicity))

    def calculate_harmonics(self, hkl):
        h,k,l = hkl[...,0],hkl[...,1],hkl[...,2]
        n = torch.gcd(torch.gcd(h, k), l)
        return n[...,None]

    def _flat_expand_harmonics(self, asu_id, hkl, wavelength):
        n = self.calculate_harmonics(hkl)

        dmin = self.rac.dmin[asu_id]

        hkl_0 = hkl // n
        wavelength_0 = wavelength * n
        d_0 = self.rac.compute_dHKL(asu_id, hkl_0)

        # we want { n * hkl | d_0 / n >= dmin , lam_min <= lam_0 / n <= lam_max}
        # this gives 3 inequalities:
        #   n <= d_0 / dmin
        #   n <= lam_0 / lam_min
        #   n >= lam_0 / lam_max
        n_max = torch.minimum(d_0 // dmin, wavelength_0 // self.wavelength_min)
        n_min = wavelength_0 // self.wavelength_max + 1 #add one because floor div rounds down

        # This is the number of refls contributing to each observation
        multiplicity = n_max - n_min + 1
        # This can be negative but that just means it should be filtered 
        multiplicity = torch.maximum(torch.zeros_like(multiplicity), multiplicity)
        # For consistency we need to cap the max multiplicity
        multiplicity = torch.minimum(multiplicity, self.max_multiplicity)

        n_all = n_min + torch.arange(self.max_multiplicity, device=asu_id.device)
        n_all[n_all > n_max] = 0.

        hkl_all = hkl_0[...,None,:] * n_all[...,:,None].to(torch.int)
        refl_id = self.rac(asu_id[...,None], hkl_all)
        absent = refl_id < 0

        hkl_all[absent] = 0 #TODO: is this line necessary? 
        n_all[absent] = 0 #what about this one?
        present = ~absent
        n_inv = n_all.clone()
        n_inv[present] = torch.reciprocal(n_inv[present])
        d_all = d_0 * n_inv
        wavelength_all = wavelength_0 * n_inv

        return hkl_all, wavelength_all[...,None], d_all[...,None], refl_id[...,None]



    def _zeros_like_add_harmonic_dim(self, tensor, dtype=None, device=None):
        shape = tensor.shape[:-1] + (5,) + tensor.shape[-1:]
        if dtype is None:
            dtype = tensor.dtype
        if device is None:
            device = tensor.device
        return torch.zeros(shape, dtype=dtype, device=device)

    def forward(self, asu_id, hkl, wavelength):
        """
        Expand the harmonics of hkl updating dHKL and wavelength accordingly. 
        This does not support any leading batch dimensions. 

        Parameters
        ----------
        asu_id : tensor
            ... x 1 tensor of asu indices.
        hkl : tensor
            ... x 3 tensor of miller indices. Note these should reflect the
            miller indices as indexed not after mapping to the reciprocal
            asymmetric unit. 
        wavelength : tensor
            ... x 1 tensor of wavelengths. 

        Returns
        -------
        hkl_all : tensor
            ... x h x 3 tensor of miller indices with a new harmonic dimension
            with size given by max_multiplicity
        wavelength_all : tensor
            ... x h x 1 tensor of wavelengths with a new harmonic dimension
            with size given by max_multiplicity
        d_all : tensor
            ... x h x 1 tensor of resolutions with a new harmonic dimension
            with size given by max_multiplicity
        refl_id : tensor
            ... x h x 1 tensor of reflection indices consistent with the reciprocal 
            ASU colleciotn
        """
        mask = (hkl != 0).any(-1)

        hkl_all = self._zeros_like_add_harmonic_dim(hkl)
        wavelength_all = self._zeros_like_add_harmonic_dim(wavelength)
        d_all = self._zeros_like_add_harmonic_dim(wavelength)
        refl_id = self._zeros_like_add_harmonic_dim(asu_id, dtype=torch.int)

        retval = self._flat_expand_harmonics(
                asu_id[mask], hkl[mask], wavelength[mask])
        hkl_all[mask], wavelength_all[mask], d_all[mask], refl_id[mask] = self._flat_expand_harmonics(asu_id[mask], hkl[mask], wavelength[mask])

        return hkl_all, wavelength_all, d_all, refl_id


