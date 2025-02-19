import numpy as np
import pandas as pd
from siren import utils
import mrcfile
import os 
import glob
import pickle 

def binarize_vol_array(vols_list, vols_num, voxels_num, bin_thr, filter_bin = False):
    vol_array = np.zeros((vols_num, voxels_num))
    if type(bin_thr) == float:
        vols_added = None
        for i, vol in enumerate(vols_list):
            vol_data,_ = load_vol(vol)
            assert vol_data.shape[0]==vol_data.shape[1]==vol_data.shape[2], 'Input maps must be cubic'
            vol_array[i] = utils.load_vol(vol)[0].flatten()
        binned_array = np.where(vol_array > bin_thr, 1, 0)
    elif type(bin_thr) == str:
        vols_added = [] 
        bin_thr = pd.read_csv(bin_thr, index_col = 0)
        bin_thr = bin_thr.set_index('vol_id')
        avg = np.mean(bin_thr['denormalized_predictions'])
        std=np.std(bin_thr['denormalized_predictions'])
        for i, vol in enumerate(vols_list):
            vol_num = int(vol.split('_')[-1].split('.mrc')[0])
            vol_thr = bin_thr.loc[vol_num, 'denormalized_predictions']
            if filter_bin:
                if (vol_thr <= avg + 2*std) and (vol_thr >= avg - 2*std):
                    data = utils.load_vol(vol)[0].flatten()
                    data = np.where(data > vol_thr, 1, 0)
                    vol_array[i] = data
                    vols_added.append(vol_num)
                else:
                    print(f'vol_{vol_num} omitted')
            else:
                data = utils.load_vol(vol)[0].flatten()
                data = np.where(data > vol_thr, 1, 0)
                vol_array[i] = data
        vol_array = pd.DataFrame(vol_array)
        binned_array = vol_array.loc[~(vol_array==0).all(axis=1)].values
    union_vox = np.where(np.sum(binned_array, axis = 0) > vols_num/100)[0]
    binned_array = binned_array[:, union_vox]
    binned_array = binned_array.astype('int')
    return binned_array, union_vox, vols_added

def check_dir(dirname, make = False):
    if not dirname.endswith('/'):
        dirname = dirname + '/'
    if make:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    return dirname

def find_cutoffs(freqs):
    freq1, freq2, n, p, q, v = freqs
    x = np.zeros(v)
    y = np.zeros(v)
    x[0:freq1] = 1
    y[0:freq2] = 1
    tilex = np.zeros((n, v))
    tiley = np.zeros((n, v))
    
    counter = 0
    while counter < n:
        new_x = np.random.choice(x, size = (1, v), replace = True)
        new_y = np.random.choice(y, size = (1, v), replace = True)
        tilex[counter, :] = new_x
        tiley[counter, :] = new_y
        counter += 1
    summed = tilex + tiley
    both_pos = np.array([len(np.where(summed[i, :] == 2)[0]) for i in range(0, n)])
    both_neg = np.array([len(np.where(summed[i, :] == 0)[0]) for i in range(0, n)])
    pos_cutoff = np.percentile(both_pos, p*100)
    neg_cutoff = np.percentile(both_neg, q*100)
    
    return (freq1, freq2, pos_cutoff, neg_cutoff)

def write_vol(sel, dictionary, out_dir, vollist, voxels, boxsize, apix):
    outfile = f'{out_dir}/block_{str(sel)}.mrc'
    with mrcfile.open(vollist[0], 'r', permissive = True) as mrc:
        #data = mrc.data
        head = mrc.header
    corrgroup = voxels[dictionary[sel]]
    new_data = np.zeros((boxsize, boxsize, boxsize))
    new_data = new_data.flatten()
    new_data[corrgroup] = 1
    with mrcfile.new(outfile, overwrite = True) as mrc:
        mrc.set_data(new_data.reshape(boxsize, boxsize, boxsize).astype('float32'))
        mrc.set_extended_header(head)
        mrc.voxel_size = apix
    return outfile

def read_blocks(block_dir, union_vox):
    blocks_dict = {}
    for i in glob.glob(block_dir + '*.mrc'):
        block_num = int(i.split('_')[-1].split('.mrc')[0])
        block = utils.load_vol(i)[0].flatten()
        block_vox = np.where(block == 1)[0]
        blocks_dict[block_num] = np.where(np.isin(union_vox, block_vox))[0].tolist()
    return blocks_dict

def calc_pval(freqs):
    vox1, vox2, freq1, freq2, obs_pos, obs_neg, num_vols = freqs
    p = binomtest(obs_pos, num_vols, p=freq1/num_vols*freq2/num_vols).pvalue
    q = binomtest(obs_neg, num_vols, p=(1-freq1/num_vols)*(1-freq2/num_vols)).pvalue
    pdir = 0
    qdir = 0
    if obs_pos > freq1*freq2/num_vols:
        pdir = 1
    if obs_neg > (1-freq1)*(1-freq2)/num_vols:
        qdir = 1
    return [vox1, vox2, p, q, pdir, qdir]

def create_mapping(vols_list, union_vox):
    mapping = utils.load_vol(vols_list[0])[0].astype('str')
    for i1, i2 in enumerate(mapping):
        for j1, j2 in enumerate(i2):
            for k1, k2 in enumerate(j2):
                mapping[i1,j1,k1] = ''.join([str(m).zfill(2) for m in [i1, j1, k1]]) 
    mapping = mapping.flatten()[union_vox]
    return mapping
