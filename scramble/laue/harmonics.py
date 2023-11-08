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

    def convolve_harmonics(self, tensor, harmonic_id, size=None):
        if size is None:
            size = harmonic_id.max() + 1
        d = tensor.shape[-1]
        dtype = tensor.dtype
        device = tensor.device
        result = torch.zeros(
            (size,d), dtype=dtype, device=device
        )
        idx = torch.tile(harmonic_id, (1, d))
        result = result.scatter_add(
            -2, 
            idx, 
            tensor
        )
        return result

    def calculate_harmonics(self, hkl):
        h,k,l = hkl[...,0],hkl[...,1],hkl[...,2]
        n = torch.gcd(torch.gcd(h, k), l)
        return n[...,None]

    def forward(self, asu_id, hkl, wavelength):
        """
        Expand the harmonics of hkl updating dHKL and wavelength accordingly. 
        This does not support any leading batch dimensions. 

        Parameters
        ----------
        hkl : tensor
            n x 3 tensor of miller indices. Note these should reflect the
            miller indices as indexed not after mapping to the reciprocal
            asymmetric unit. 
        dHKL : tensor
            n x 1 tensor of resolutions. 
        wavelength : tensor
            n x 1 tensor of wavelengths. 
        metadata : tensors (optional)
            Optionally specify additional n x d medata vectors which will be
            tiled to match the updated miller indices. 
        """
        n = self.calculate_harmonics(hkl)
        mask = (hkl == 0).all(-1, keepdims=True)
        dmin = self.rac.dmin[asu_id]

        hkl_0 = hkl // torch.where(mask, 1, n)
        wavelength_0 = wavelength * n

        d_0 = self.rac.compute_dHKL(asu_id, hkl_0)
        n_max = torch.minimum(
            torch.where(mask, 0, d_0 // dmin),
            torch.where(mask, 0, wavelength_0 // self.wavelength_min)
        )
        n_min = torch.where(mask, 0, wavelength_0 // self.wavelength_max + 1)
        multiplicity = n_max - n_min + 1
        multiplicity = torch.maximum(torch.zeros_like(multiplicity), multiplicity)
        multiplicity = torch.where(mask, 0, multiplicity)
        multiplicity = torch.minimum(multiplicity, self.max_multiplicity)

        n_all = n_min + torch.arange(self.max_multiplicity, device=asu_id.device)
        n_all = torch.where(n_all <= n_max, n_all, 0)

        hkl_all = hkl_0[...,None,:] * n_all[...,:,None].to(torch.int)
        refl_id = self.rac(asu_id[...,None], hkl_all)
        absent = refl_id < 0
        hkl_all = torch.where(absent[...,None], 0, hkl_all)
        n_all = torch.where(absent, 0, n_all)

        idx = n_all == 0.
        d_all = torch.where(idx, 0., d_0) / torch.where(idx, 1., n_all)
        wavelength_all = torch.where(idx, 0., wavelength_0) / torch.where(idx, 1., n_all)

        return hkl_all, wavelength_all[...,None], d_all[...,None], refl_id[...,None]

def calculate_harmonics(H):
    n = np.gcd.reduce(H, axis=-1)
    return n

@rs.decorators.range_indexed
def expand_harmonics(ds, dmin=None,  wavelength_key='Wavelength'):
    """
    Expand reflection observations to include all contributing harmonics. All 
    contributing reflections will be included out to a resolution cutoff 
    irrespective of peak wavelength.

    Parameters
    ----------
    ds : rs.DataSet
        Laue data without multiples. These should be unmerged data with 
        miller indices that have not been mapped to the ASU (P1 Miller Indices).
    dmin : float
        Highest resolution in Å to which harmonics will be predicted. If not 
        supplied, the highest resolution reflection in ds will set dmin.

    Returns
    -------
    ds : rs.DataSet
        DataSet with all reflection observations expanded to include their 
        constituent reflections. New columns 'H_0', 'K_0', 'L_0' will be added 
        to each reflection to store the Miller indices of the innermost 
        reflection on each central ray. 
    """
    if ds.merged:
        raise ValueError("Expected unmerged data, but ds.merged is True")

    ds = ds.copy()

    #Here's where we get the metadata for Laue harmonic deconvolution
    #This is the HKL of the closest refl on each central ray
    if 'dHKL' not in ds:
        ds.compute_dHKL(inplace=True)
    if dmin is None:
        dmin = ds['dHKL'].min() - 1e-12

    Hobs = ds.get_hkls()

    #Calculated the harmonic as indexed
    nobs = calculate_harmonics(Hobs)

    #Add primary harmonic miller index, wavelength, and resolution
    # H = H_n / n
    # lambda = lambda_n * n
    # d = d_n * n
    H_0 = (Hobs/nobs[:,None]).astype(np.int32)
    d_0 = ds['dHKL'].to_numpy() * nobs
    Wavelength_0 = ds[wavelength_key].to_numpy() * nobs

    #This is the largest harmonic that should be
    #included for each observation in order to
    #respect the resolution cutoff
    n_max =  np.floor_divide(d_0, dmin).astype(int)

    #This is where we make the indices to expand
    #each harmonic the appropriate number of times given dmin
    n = np.arange(1, n_max.max() + 2)
    idx,n = np.where(n <= n_max[:,None])
    n = n + 1
    #idx are the indices for expansion and n is the corresponding
    #set of harmonic integers

    ds = ds.iloc[idx]
    ds['H_0'],ds['K_0'],ds['L_0'] = H_0[idx].T
    ds['dHKL'] = (d_0[idx] / n)
    ds[wavelength_key] = (Wavelength_0[idx] / n)
    ds['H'],ds['K'],ds['L'] = (n[:,None] * H_0[idx]).T

    return ds

if __name__=="__main__":
    ds = rs.read_mtz("../../../careless-examples/pyp/off.mtz")
    ds = ds[ds.BATCH==1]
    ds['Hobs'],ds['Kobs'],ds['Lobs'] = ds.get_hkls().T
    dmin = 1.5

    metadata = ds[['X', 'Y']].to_numpy()
    hkl = ds.get_hkls()
    dHKL = ds.compute_dHKL().dHKL.to_numpy()[:,None]
    I = ds['I'].to_numpy()[:,None]
    SigI = ds['I'].to_numpy()[:,None]
    wavelength = ds['Wavelength'].to_numpy()[:,None]

    hkl = torch.tensor(hkl, dtype=torch.long)
    metadata = torch.tensor(metadata, dtype=torch.float32)
    dHKL = torch.tensor(dHKL, dtype=torch.float32)
    I = torch.tensor(I, dtype=torch.float32)
    SigI = torch.tensor(SigI, dtype=torch.float32)
    wavelength = torch.tensor(wavelength, dtype=torch.float32)

    ds_with_harmonics = expand_harmonics(ds, dmin)
    eh = ExpandHarmonics(dmin)
    (hkl, dHKL, wavelength, metadata, eI, eSigI), harmonic_id = eh(hkl, dHKL, wavelength, metadata, I, SigI)

    n_obs = eh.calculate_harmonics(hkl)

    assert np.all(ds_with_harmonics.get_hkls() == hkl.numpy())
    assert np.isclose(
        ds_with_harmonics.Wavelength.to_numpy(),
        wavelength.numpy().squeeze(),
    ).all()
    assert np.isclose(
        ds_with_harmonics.dHKL.to_numpy(),
        dHKL.numpy().squeeze(),
    ).all()
    assert len(ds) == len(ds_with_harmonics.groupby(['H_0', 'K_0', 'L_0']))
    from IPython import embed
    embed(colors='linux')

    metadata = ds[['X', 'Y']].to_numpy()
    hkl = ds.get_hkls()
    dHKL = ds.compute_dHKL().dHKL.to_numpy()[:,None]
    I = ds['I'].to_numpy()[:,None]
    SigI = ds['I'].to_numpy()[:,None]
    wavelength = ds['Wavelength'].to_numpy()[:,None]

    hkl = torch.tensor(hkl, dtype=torch.long)
    metadata = torch.tensor(metadata, dtype=torch.float32)
    dHKL = torch.tensor(dHKL, dtype=torch.float32)
    I = torch.tensor(I, dtype=torch.float32)
    SigI = torch.tensor(SigI, dtype=torch.float32)
    wavelength = torch.tensor(wavelength, dtype=torch.float32)
    
    eh = ExpandHarmonics(dmin, 1., 1.2)
    (hkl, dHKL, wavelength, metadata, eI, eSigI), harmonic_id = eh(hkl, dHKL, wavelength, metadata, I, SigI)

    from IPython import embed
    embed(colors='linux')


