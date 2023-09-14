"""
Solve the indexing ambiguity problem for serial
Laue data. 
"""

from argparse import ArgumentParser
import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np
from matplotlib import pyplot as plt
from scramble.laue import expand_harmonics
from scramble.model import MergingModel,ScalingModel,SurrogatePosterior,NormalLikelihood
from os.path import dirname

parser = ArgumentParser(description=__doc__)

parser.add_argument("metadata_keys", type=str, help='Comma separated list of keys to be used as metadata during scaling.')
parser.add_argument("mtz", nargs="+")
parser.add_argument("-s", "--spacegroup", default=None, type=str, help="The spacegroup to use for determining reindexing operations. By default the group from the first input mtz file will be used.")
parser.add_argument("-c", "--cell", nargs=6, default=None, type=float, help="The cell to use for determining reindexing operations. By default the cell from the first input mtz file will be used.")
parser.add_argument("-d", "--dmin", type=float, default=None, help="Resolution cutoff.")
parser.add_argument("-b", "--batch-key", type=str, default=None, help="The name of the batch (image number) key. By default this will be assigned to the first batch key of the first mtz file. All mtzs must use the same batch key.")
parser.add_argument("-i", "--intensity-key", type=str, default=None, help="The name of the intensity key. By default this will be assigned to the first intensityh key of the first mtz file. All mtzs must use the same intensity key.")
parser.add_argument("-q", "--sigma-key", type=str, default=None, help="The name of the sigma (uncertainty) key. By default this will be assigned to the first sigma key of the first mtz file. All mtzs must use the same sigma key.")
parser.add_argument("-w", "--wavelength-key", type=str, default='Wavelength', help="The name of the wavelength key. The default is 'Wavelength'.")
parser.add_argument('--max-obliq', type=float, default=1e-3, help='Default value is 1e-3. See the "max_obliq" argument for gemmi.find_twin_laws.')
parser.add_argument('--double-precision', action='store_true', help='Use double precision floating point math instead of single.')
parser.add_argument('--plot', action='store_true', help='Produce a bar plot of reindexing ops.')
parser.add_argument('--use-cuda', action='store_true', help='Use a (single) GPU.')
parser.add_argument('--anomalous', action='store_true', help='Keep the two halves of reciprocal space separate in the results.')
parser.add_argument('--mlp-width', type=float, default=32, help='Width of the multilayer perceptron with default 32.')
parser.add_argument('--mlp-depth', type=float, default=10, help='Depth of the multilayer perceptron with default 10.')
parser.add_argument('--steps', type=int, default=10_000, help='Number of optimization steps.')
parser.add_argument('--mtz-out', type=str, default=None)
parser.add_argument('--mtz-suffix', type=str, default='_scrambled.mtz')
parser = parser.parse_args()



def get_first_key_of_type(ds, dtype):
    idx = ds.dtypes == dtype
    keys = ds.dtypes.keys()[idx]
    key = keys[0]
    return key

def main():
    data = []
    spacegroup = parser.spacegroup
    cell = parser.cell
    batch_key = parser.batch_key
    metadata_keys = parser.metadata_keys.split(',')
    intensity_key = parser.intensity_key
    wavelength_key = parser.wavelength_key
    sigma_key = parser.sigma_key
    float_dtype = torch.float32
    int_dtype = torch.int64 #int32 is not supported by scatter_add
    device = None
    anomalous = parser.anomalous
    if parser.double_precision:
        float_dtype=torch.double
    if parser.use_cuda:
        device = 'cuda'

    batch_start = 0
    for i,mtz in enumerate(parser.mtz):
        ds = rs.read_mtz(mtz)
        if cell is None:
            cell = ds.cell
        if spacegroup is None:
            spacegroup = ds.spacegroup
        if batch_key is None:
            batch_key = get_first_key_of_type(ds, 'B')
        if intensity_key is None:
            intensity_key = get_first_key_of_type(ds, 'J')
        if sigma_key is None:
            sigma_key = get_first_key_of_type(ds, 'Q')

        ds['file_id'] = i

        ds['image_id'] = ds.groupby(batch_key).ngroup() + batch_start
        batch_start = ds['image_id'].max() + 1
        data.append(ds)

    ds = rs.concat(data, check_isomorphous=False)
    ds.cell = cell
    ds.spacegroup = spacegroup
    ds.compute_dHKL(inplace=True)

    dmin = parser.dmin
    if dmin is None:
        dmin = ds.dHKL.min()

    ds = ds[ds.dHKL >= dmin]

    reindexing_ops = [gemmi.Op("x,y,z")] 
    reindexing_ops.extend(gemmi.find_twin_laws(cell, spacegroup, parser.max_obliq, False))


    Hasu = rs.utils.generate_reciprocal_asu(cell, spacegroup, dmin)
    ds['Hobs'], ds['Kobs'], ds['Lobs'] = ds.get_hkls().T
    wavelength_min,wavelength_max = ds[wavelength_key].min(),ds[wavelength_key].max()
    ds = expand_harmonics(ds)
    idx = (ds[wavelength_key] >= wavelength_min) & (ds[wavelength_key] <= wavelength_max)
    ds = ds[idx]
    ds['harmonic_id'] = ds.groupby(['H_0', 'K_0', 'L_0', batch_key, 'file_id']).ngroup()

    Iobs,SigIobs = ds[['harmonic_id', intensity_key, sigma_key]].groupby('harmonic_id').first().to_numpy().T

    hkl = torch.tensor(
        ds.get_hkls(), 
        dtype=int_dtype, 
        device=device
    )
    image_id = torch.tensor(
        ds.image_id.to_numpy(),
        device=device,
        dtype=int_dtype,
    )
    harmonic_id = torch.tensor(
        ds.harmonic_id.to_numpy(),
        device=device,
        dtype=int_dtype,
    )
    Iobs = torch.tensor(
        Iobs,
        device=device,
        dtype=float_dtype,
    )
    SigIobs = torch.tensor(
        SigIobs,
        device=device,
        dtype=float_dtype,
    )
    metadata = torch.tensor(
        ds[metadata_keys].to_numpy(),
        device=device,
        dtype=float_dtype,
    )
    from scramble.symmetry import ReciprocalASU,Op

    rasu = ReciprocalASU(cell, spacegroup, dmin, anomalous)
    reindexing_ops = [Op(op) for op in reindexing_ops]

    surrogate_posterior = SurrogatePosterior(rasu, reindexing_ops)

    scaling_model = ScalingModel(parser.mlp_width, parser.mlp_depth)
    likelihood = NormalLikelihood()
    merging_model = MergingModel(surrogate_posterior, scaling_model, likelihood)

    opt = torch.optim.Adam(merging_model.parameters())

    data = [hkl, Iobs, SigIobs, image_id, metadata, harmonic_id]
    if parser.use_cuda:
        merging_model = merging_model.cuda()
        data = [datum.cuda() for datum in data]

    save_frequency = 100
    from tqdm import trange
    bar = trange(parser.steps)
    loss = []
    op_ids = []
    for i in bar:
        opt.zero_grad()
        elbo,op_id = merging_model(*data)
        elbo.backward()
        opt.step()
        elbo = float(elbo)
        loss.append(elbo)
        bar.set_postfix({'ELBO' : f'{elbo:0.2e}'})
        op_ids.append(op_id.detach().cpu().numpy())


    if parser.mtz_out is not None:
        F, SigF = surrogate_posterior.get_f_sigf()
        q = surrogate_posterior.distribution()
        h,k,l = surrogate_posterior.reciprocal_asu.Hasu.T
        out = rs.DataSet({
            'H' : rs.DataSeries(h, dtype='H'),
            'K' : rs.DataSeries(k, dtype='H'),
            'L' : rs.DataSeries(l, dtype='H'),
            'I' : rs.DataSeries(q.mean.detach().cpu().numpy(), dtype='J'),
            'SIGI' : rs.DataSeries(q.stddev.detach().cpu().numpy(), dtype='Q'),
            'F' : rs.DataSeries(F, dtype='F'),
            'SIGF' : rs.DataSeries(SigF, dtype='Q'),
        }, merged=True, cell=cell, spacegroup=spacegroup).set_index(["H", "K", "L"])
        out.write_mtz(parser.mtz_out)

    csv = 'id,file,' + ','.join([f'"{op.gemmi_op.triplet()}"' for op in reindexing_ops]) + '\n'
    op_id = op_id.detach().cpu().numpy()
    batch_start = 0

    for i,mtz in enumerate(parser.mtz):
        ds = rs.read_mtz(mtz).reset_index()
        ds['image_id'] = ds.groupby(batch_key).ngroup() + batch_start
        mtz_op_id = op_id[ds.image_id.to_numpy()]
        for j,op in enumerate(reindexing_ops):
            idx = mtz_op_id == j
            ds[idx] = ds[idx].apply_symop(op.gemmi_op)

        batch_end = ds['image_id'].max() + 1
        out = mtz[:-4] + parser.mtz_suffix
        del(ds['image_id'])
        ds = ds.set_index(['H', 'K', 'L'])
        ds.write_mtz(out)

        image_op_id = op_id[batch_start:batch_end]
        counts = np.bincount(image_op_id, minlength=len(reindexing_ops))
        line = f"{i+1},{mtz}," + ','.join(map(str, counts)) + '\n'
        csv = csv + line
        batch_start = batch_end

    counts = np.bincount(op_id, minlength=len(reindexing_ops))
    line = f"{i+1},Total," + ','.join(map(str, counts)) + '\n'
    csv = csv + line

    print("Reindexing operation counts:")
    print(csv)
    out = dirname(parser.mtz[0]) + '/scramble.log'
    with open(out, 'w') as f:
        f.write(csv)

    counts = np.bincount(op_id)
    x = np.arange(len(counts))
    plt.bar(x, counts, color='k')
    plt.xticks(
        x,
        [op.gemmi_op.triplet() for op in reindexing_ops],
        ha='right', 
        rotation=45,
        rotation_mode='anchor',
    )
    plt.xlabel('Reindexing Operation')
    plt.ylabel('Images')
    plt.title('Reindexing Results')
    if parser.plot:
        plt.show()

    out = dirname(parser.mtz[0]) + '/scramble.png'
    plt.savefig(out)


if __name__=="__main__":
    main()

