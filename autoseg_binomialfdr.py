import argparse
import pandas as pd
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from cryodrgn import mrc
from cryodrgn import utils
from scipy.special import comb
from scipy.stats import binomtest
import networkx as nx

def add_args(parser):
    parser.add_argument('--voldir', type = str, required = True, help = 'Directory where (downsampled) volumes are stored')
    parser.add_argument('--threads', type = int, required = True, help = 'Number of threads for multiprocessing')
    parser.add_argument('--bin', type = float, default = None, help = 'Threshold at which to binarize maps')
    parser.add_argument('--outdir', type = str, default = './', help = 'Directory where outputs will be stored')
    parser.add_argument('--alpha_pos', type = float, default = 0.01, help = 'P value threshold for positive co-occupancy')
    parser.add_argument('--exp_frac', type = float, default = 0.25, help = '')
    return parser

def check_dir(dirname, make = False):
    if not dirname.endswith('/'):
        dirname = dirname + '/'
    if make:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    return dirname

def calc_pval(freqs):
    vox1, vox2, freq1, freq2, obs, num_vols = freqs
    p = binomtest(obs, num_vols, p=freq1*freq2)
    return [vox1, vox2, p]

def consolidate_blocks(voxdict):
    intersects = {(i,j): len(set(voxdict[i]) & set(voxdict[j]))/len(set(voxdict[i]) | set(voxdict[j])) for i,j in combinations(voxdict.keys(), 2)}
    consolidated = {}
    consolidate = [i for i in intersects if intersects[i] > 0.4]
    for j in voxdict:
        g = np.unique([i for i in consolidate if j in i])
        if len(g) == 0:
            g = [j]
        voxsets = [set(voxdict[i]) for i in g]
        grouped = list(set().union(*voxsets))
        counter = 0
        for i in consolidated:
            if grouped == consolidated[i]:
                counter += 1
        if counter == 0:
            consolidated[j] = grouped
    return consolidated

def write_vol(sel, dictionary, out_dir, vollist, voxels):
    outfile = f'{out_dir}/blocks_dict_expand_{str(sel)}.mrc'
    data, head = mrc.parse_mrc(vollist[0])
    corrgroup = voxels[dictionary[sel]]
    data[:] = 0
    data = data.flatten()
    data[corrgroup] = 1
    mrc.write(outfile, data.reshape(64, 64, 64), header = head)
    return

def main(args):
    t0 = time.time()
    
    outdir = check_dir(args.outdir, make = True)
    voldir = check_dir(args.voldir)
    vol_list = np.sort(glob.glob(voldir + '*.mrc'))
    num_vols = len(vol_list)
    boxsize = mrc.parse_mrc(vol_list[0])[0].shape[0]
    num_voxels = boxsize**3

    print(f'reading in data from {voldir}')
    vol_array = np.zeros((num_vols, num_voxels))

    for i, vol in enumerate(vol_list):
        vol_array[i] = mrc.parse_mrc(vol)[0].flatten()
    
    print(f'binarizing data at threshold {args.bin}')
    binned = np.where(vol_array > args.bin, 1, 0)
    union_voxels = np.where(np.sum(binned, axis = 0) > 5)[0]
    binned = binned[:, union_voxels]
    binned = binned.astype('int')
    vals = range(0, len(union_voxels))
    
    print('calculating p-values')
    pvals_pos = []
    comb_list = np.array(list(combinations(vals, 2)))
    mapdata_pos = []
    for i in comb_list:
        vox1, vox2 = i 
        x = binned[:, vox1].astype('float')
        y = binned[:, vox2].astype('float')
        f1 = np.sum(x)/len(x)
        f2 = np.sum(y)/len(y)   
        summed = x + y
        obs_pos = len(np.where(summed == 2)[0])
        mapdata_pos.append((vox1, vox2, f1, f2, obs_pos, num_vols))

    pool = Pool(args.threads)
    mapresults_pos = pool.map(calc_pval, mapdata_pos)
    combs_pos = np.array([i[0:2] for i in mapresults_pos])
    pvals_pos = np.array([i[2] for i in mapresults_pos])
    sorted_pos = np.sort(pvals_pos)
    threshold_pos = max([i for j,i in enumerate(sorted_pos) if i < j/len(sorted_pos)*args.alpha_pos])


    print('sketching communities')
    corr_graph = nx.Graph() 
    pass_pos = combs_pos[np.where(pvals_pos <= threshold_pos)[0]]
    corr_graph.add_edges_from(pass_pos)

    blocks_dict = {}
    for i, j in enumerate(nx.algorithms.community.label_propagation.label_propagation_communities(corr_graph)):
        blocks_dict[i] = list(j)
    
    utils.save_pkl(blocks_dict, outdir + 'blocks_dict.pkl')
    
    t_final = time.time() - t0
    print(f'total run time: {t_final}s')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())

        
    


