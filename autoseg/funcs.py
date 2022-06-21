import numpy as np
from cryodrgn import mrc
import os 
import glob

def binarize_vol_array(vols_list, vols_num, voxels_num, bin_thr):
    vol_array = np.zeros((vols_num, voxels_num))
    for i, vol in enumerate(vols_list):
        vol_array[i] = mrc.parse_mrc(vol)[0].flatten()
    binned_array = np.where(vol_array > bin_thr, 1, 0)
    union_vox = np.where(np.sum(binned_array, axis = 0) > 5)[0]
    binned_array = binned_array[:, union_vox]
    binned_array = binned_array.astype('int')
    return binned_array, union_vox

def check_dir(dirname, make = False):
    if not dirname.endswith('/'):
        dirname = dirname + '/'
    if make:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    return dirname

def find_cutoffs(freqs):
    freq1, freq2, n, p, q = freqs
    x = np.zeros(500)
    y = np.zeros(500)
    x[0:freq1] = 1
    y[0:freq2] = 1
    tilex = np.zeros((n, 500))
    tiley = np.zeros((n, 500))
    
    counter = 0
    while counter < n:
        new_x = np.random.choice(x, size = (1, 500), replace = True)
        new_y = np.random.choice(y, size = (1, 500), replace = True)
        tilex[counter, :] = new_x
        tiley[counter, :] = new_y
        counter += 1
    summed = tilex + tiley
    both_pos = np.array([len(np.where(summed[i, :] == 2)[0]) for i in range(0, n)])
    both_neg = np.array([len(np.where(summed[i, :] == 0)[0]) for i in range(0, n)])
    pos_cutoff = np.percentile(both_pos, p*100)
    neg_cutoff = np.percentile(both_neg, q*100)
    
    return (freq1, freq2, pos_cutoff, neg_cutoff)

def write_vol(sel, dictionary, out_dir, vollist, voxels):
    outfile = f'{out_dir}/block_{str(sel)}.mrc'
    data, head = mrc.parse_mrc(vollist[0])
    corrgroup = voxels[dictionary[sel]]
    data[:] = 0
    data = data.flatten()
    data[corrgroup] = 1
    mrc.write(outfile, data.reshape(64, 64, 64), header = head)
    return outfile

def read_blocks(block_dir, union_vox):
    blocks_dict = {}
    for i in glob.glob(block_dir + '*.mrc'):
        block_num = int(i.split('_')[-1].split('.mrc')[0])
        block = mrc.parse_mrc(i)[0].flatten()
        block_vox = np.where(block == 1)[0]
        blocks_dict[block_num] = np.where(np.isin(union_vox, block_vox))[0].tolist()
    return blocks_dict

def calc_pval(freqs):
    vox1, vox2, freq1, freq2, obs_pos, obs_neg, num_vols = freqs
    p = binomtest(obs_pos, num_vols, p=freq1/num_vols*freq2/num_vols).pvalue
    q = binomtest(obs_neg, num_vols, p=(1-freq1/num_vols)*(1-freq2/num_vols)).pvalue
    pdir = 0
    qdir = 0
    if obs_pos > freq1*freq2/500:
        pdir = 1
    if obs_neg > (1-freq1)*(1-freq2)/500:
        qdir = 1
    return [vox1, vox2, p, q, pdir, qdir]


