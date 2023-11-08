"""
Solve the indexing ambiguity problem for serial
Laue data. 
"""

from argparse import ArgumentParser
import pandas as pd
import gemmi
import reciprocalspaceship as rs
import torch
import numpy as np
from matplotlib import pyplot as plt
from scramble.laue import ExpandHarmonics
from scramble.model import MergingModel,ScalingModel,SurrogatePosterior,NormalLikelihood
from scramble.symmetry import ReciprocalASU,Op
from os.path import dirname,abspath

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
parser.add_argument("--wavelength-range", default=None, type=float, nargs=2, help="Optionally specify the wavelength range to be used for harmonic deconvolution. By default, the empirical range will be used.")
parser.add_argument('--max-obliq', type=float, default=1e-3, help='Default value is 1e-3. See the "max_obliq" argument for gemmi.find_twin_laws.')
parser.add_argument('--double-precision', action='store_true', help='Use double precision floating point math instead of single.')
parser.add_argument('--use-debug-scale', action='store_true', help='Use a simpler scaling model for debugging.')
parser.add_argument('--disable-index-disambiguation', action='store_true', help='Disable indexing disambiguation. This option is suitable for single crystal data.')
parser.add_argument('--disable-harmonic-deconvolution', action='store_true', help='Disable harmonic deconvolution. This option is primarily for debugging purposes.')
parser.add_argument('--plot', action='store_true', help='Produce a bar plot of reindexing ops.')
parser.add_argument('--use-cuda', action='store_true', help='Use a (single) GPU.')
parser.add_argument('--anomalous', action='store_true', help='Keep the two halves of reciprocal space separate in the results.')
parser.add_argument('--mlp-width', type=float, default=32, help='Width of the multilayer perceptron with default 32.')
parser.add_argument('--mlp-depth', type=float, default=10, help='Depth of the multilayer perceptron with default 10.')
parser.add_argument('--steps', type=int, default=10_000, help='Number of optimization steps.')
parser.add_argument('--save-frequency', type=int, default=1_000, help='How often to write output. Defaults to every 1000 steps.')
parser.add_argument('--mtz-out', type=str, default=None)
parser.add_argument('--mtz-suffix', type=str, default='_scrambled.mtz')
parser.add_argument('-k', '--kl-weight', type=float, default=1e-3)
parser.add_argument('--mc-samples', type=int, default=32)
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
    mc_samples = parser.mc_samples

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

    kl_weight = parser.kl_weight
    dmin = parser.dmin
    if dmin is None:
        dmin = ds.dHKL.min()
    if parser.wavelength_range is None:
        wavelength_min = ds[wavelength_key].min()
        wavelength_max = ds[wavelength_key].max()
    else:
        wavelength_min,wavelength_max = parser.wavelength_range

    reindexing_ops = [gemmi.Op("x,y,z")] 
    if not parser.disable_index_disambiguation:
        reindexing_ops.extend(gemmi.find_twin_laws(cell, spacegroup, parser.max_obliq, False))
    reindexing_ops = [Op(op) for op in reindexing_ops]

    rasu = ReciprocalASU(cell, spacegroup, dmin, anomalous)
    expand_harmonics = None
    if not parser.disable_harmonic_deconvolution:
        expand_harmonics = ExpandHarmonics(rasu, wavelength_min, wavelength_max)
        ds = ds[ds.dHKL >= dmin]

    surrogate_posterior = SurrogatePosterior(rasu, reindexing_ops)

    # Use this simple scaling model for debugging:
    if parser.use_debug_scale:
        from scramble.model.scaling import MLPScalingModel 
        scaling_model = MLPScalingModel(parser.mlp_width, parser.mlp_depth)
    else:
        scaling_model = ScalingModel(parser.mlp_width, parser.mlp_depth)

    likelihood = NormalLikelihood()
    merging_model = MergingModel(surrogate_posterior, scaling_model, likelihood, expand_harmonics=expand_harmonics, kl_weight=kl_weight)

    opt = torch.optim.Adam(merging_model.parameters())

    Iobs,SigIobs = ds[[intensity_key, sigma_key]].to_numpy().T

    hkl = torch.tensor(
        ds.get_hkls(), 
        dtype=int_dtype, 
        device=device
    )
    image_id = torch.tensor(
        ds.image_id.to_numpy(),
        device=device,
        dtype=int_dtype,
        )[:,None]
    Iobs = torch.tensor(
        Iobs,
        device=device,
        dtype=float_dtype,
    )[:,None]
    SigIobs = torch.tensor(
        SigIobs,
        device=device,
        dtype=float_dtype,
    )[:,None]
    metadata = torch.tensor(
        ds[metadata_keys].to_numpy('float32'),
        device=device,
        dtype=float_dtype,
    )
    #metadata = (metadata - metadata.mean(0, keepdims=True)) / metadata.std(0, keepdims=True)
    wavelength = torch.tensor(
        ds[wavelength_key].to_numpy(),
        device=device,
        dtype=float_dtype,
    )[:,None]
    dHKL = torch.tensor(
        ds.dHKL.to_numpy(),
        device=device,
        dtype=float_dtype,
    )[:,None]



    data = [hkl, Iobs, SigIobs, image_id, metadata, wavelength, dHKL]
    if parser.use_cuda:
        merging_model = merging_model.cuda()
        data = [datum.cuda() for datum in data]

    def chkpt(op_id):
        if parser.mtz_out is not None:
            out = surrogate_posterior.to_dataset()
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
        out = dirname(abspath(parser.mtz[0])) + '/scramble.log'
        with open(out, 'w') as f:
            f.write(csv)

        counts = np.bincount(op_id, minlength=len(reindexing_ops))
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

        out = dirname(abspath(parser.mtz[0])) + '/scramble.png'
        plt.savefig(out)

    from tqdm import trange
    bar = trange(1, parser.steps + 1)
    loss = []
    op_ids = []
    history = None
    for i in bar:
        opt.zero_grad()
        elbo,metrics,op_id = merging_model(*data, mc_samples=mc_samples, return_op=True, return_cc=True)
        elbo.backward()
        opt.step()
        loss.append(metrics['ELBO'])
        bar.set_postfix(metrics)
        op_ids.append(op_id.detach().cpu().numpy())
        if history is None:
            history = {k:[v] for k,v in metrics.items()}
            history['step'] = []
        else:
            for k,v in metrics.items():
                history[k].append(v)
        history['step'].append(i)

        if i % parser.save_frequency == 0:
            chkpt(op_id)
            history_file = dirname(abspath(parser.mtz[0])) + '/scramble_history.csv'
            pd.DataFrame(history).to_csv(history_file, index=False)

    chkpt(op_id)


if __name__=="__main__":
    main()

