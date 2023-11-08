import torch
from torch.utils.data import Dataset
import reciprocalspaceship as rs
from .util import get_first_key_of_type



class MTZDataset(Dataset):
    def __init__(
            self, 
            mtz_file, 
            metadata_keys,
            asu_id=0,
            wavelength_key='Wavelength', 
            intensity_key=None, 
            sigma_key=None,
            batch_key=None,
            int_dtype=torch.int64,
            float_dtype=torch.float32,
            max_refls=4096,
            reindexing_op=None,
        ):
        self.mtz_file = mtz_file
        ds = rs.read_mtz(mtz_file).reset_index()
        self.reindexing_op = reindexing_op

        if batch_key is None:
            batch_key = get_first_key_of_type(ds, 'B')
        if intensity_key is None:
            intensity_key = get_first_key_of_type(ds, 'J')
        if sigma_key is None:
            sigma_key = get_first_key_of_type(ds, 'Q')
        self.metadata_keys = metadata_keys
        self.wavelength_key = wavelength_key

        self.batch_key = batch_key
        self.sigma_key = sigma_key
        self.intensity_key = intensity_key
        self.asu_id=asu_id
        self.int_dtype=int_dtype
        self.float_dtype=float_dtype
        self.max_refls=max_refls

        ds['image_id'] = ds.groupby(batch_key).ngroup() 
        self.ds = ds.set_index('image_id')

    @classmethod
    def from_multiple_mtzs(cls, mtz_files, metadata_keys, asu_ids, **kwargs):
        if isinstance(asu_ids, int):
            asu_ids = [asu_ids] * len(mtz_files)

        data = []
        for asu_id, mtz_file in zip(asu_ids, mtz_files):
            ds = MTZDataset(
                mtz_file, 
                metadata_keys, 
                asu_id, 
                **kwargs,
            )
            data.append(ds)
        out = torch.utils.data.ConcatDataset(data)
        return out

    def __len__(self):
        return self.ds.index.max()

    def _pad(self, tensor, pad_size, value=None):
        #This is flat and numbered backwards with two values per dim starting with the last dimension
        #is this not sheer madness? why???
        pads = (0, 0, 0, pad_size) 
        out = torch.nn.functional.pad(tensor, pads, value=value)
        return out

    def __getitem__(self, idx):
        ds = self.ds.loc[idx]
        pad_size = None
        if len(ds) > self.max_refls:
            ds = ds.sample(self.max_refls, replace=False)
        elif len(ds) < self.max_refls:
            pad_size = self.max_refls - len(ds)

        asu_id = self.asu_id * torch.ones((len(ds), 1), dtype=self.int_dtype)
        hkl = torch.tensor(
            ds.get_hkls(), 
            dtype=self.int_dtype, 
        )
        Iobs = torch.tensor(
            ds[self.intensity_key].to_numpy('float32'),
            dtype=self.float_dtype,
        )[:,None]
        SigIobs = torch.tensor(
            ds[self.sigma_key].to_numpy('float32'),
            dtype=self.float_dtype,
        )[:,None]
        metadata = torch.tensor(
            ds[self.metadata_keys].to_numpy('float32'),
            dtype=self.float_dtype,
        )
        wavelength = torch.tensor(
            ds[self.wavelength_key].to_numpy('float32'),
            dtype=self.float_dtype,
        )[:,None]

        if self.reindexing_op is not None:
            hkl = self.reindexing_op(hkl)

        data = [asu_id,  hkl, Iobs, SigIobs,  metadata, wavelength]
        if pad_size is not None:
            data = [
                self._pad(asu_id, pad_size),
                self._pad(hkl, pad_size),
                self._pad(Iobs, pad_size),
                self._pad(SigIobs, pad_size, value=1.), #Pad this with a positive number to avoid issues with the likelihood
                self._pad(metadata, pad_size),
                self._pad(wavelength, pad_size),
            ]

        return data


