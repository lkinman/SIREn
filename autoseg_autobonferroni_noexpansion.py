import argparse
import pandas as pd
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
import copy
import random
from cryodrgn import mrc
from cryodrgn import utils
from scipy.special import comb
import networkx as nx

def add_args(parser):
    parser.add_argument('--voldir', type = str, required = True, help = 'Directory where (downsampled) volumes are stored')
    parser.add_argument('--threads', type = int, required = True, help = 'Number of threads for multiprocessing')
    parser.add_argument('--bin', type = float, default = None, help = 'Threshold at which to binarize maps')
    parser.add_argument('--outdir', type = str, default = './', help = 'Directory where outputs will be stored')
    parser.add_argument('--posp', type = float, default = 0.01, help = 'P value threshold before Bonferroni correction for positive co-occupancy')
    parser.add_argument('--negp', type = float, default = 0.05, help = 'P value threshold before Bonferroni correction for negative co-occupancy')
    parser.add_argument('--exp_frac', type = float, default = 0.25, help = '')
    return parser

def check_dir(dirname, make = False):
    if not dirname.endswith('/'):
        dirname = dirname + '/'
    if make:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    return dirname

def find_cutoffs(freqs):
    freq1, freq2, n, p, q = freqs
    p = p*100
    q = q*100
    x = np.zeros(500)
    y = np.zeros(500)
    x[0:freq1] = 1
    y[0:freq2] = 1
    tilex = np.zeros((n, 500))
    tiley = np.zeros((n, 500))
    
    counter = 0
    while counter < n:
        np.random.shuffle(x)
        np.random.shuffle(y)
        tilex[counter, :] = x
        tiley[counter, :] = y
        counter += 1
    summed = tilex + tiley
    both_pos = np.array([len(np.where(summed[i, :] == 2)[0]) for i in range(0, n)])
    both_neg = np.array([len(np.where(summed[i, :] == 0)[0]) for i in range(0, n)])
    pos_cutoff = np.percentile(both_pos, p)
    neg_cutoff = np.percentile(both_neg, q)
    
    return (freq1, freq2, pos_cutoff, neg_cutoff)

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
    totals = np.sum(binned, axis = 0)
    vals = range(0, len(union_voxels))
    sample_num = int(len(union_voxels)/10)
    rands = random.sample(vals, sample_num)
    print(sample_num)

    print('Calculating corrected p-values')
    corr_factor = comb(sample_num, 2)
    posp_corr = 1-(args.posp/corr_factor)
    negp_corr = 1-(args.negp/corr_factor)
    num_trials = 1000

    print('bootstrapping p-values')
    comb_list = []
    for i in combinations(np.unique(totals), 2):
        f1, f2 = i
        comb_list.append((f1, f2, num_trials, posp_corr, negp_corr))
    for i in np.unique(totals):
        comb_list.append((i, i, num_trials, posp_corr, negp_corr))
    pool = Pool(args.threads)
    cutoffs = pool.map(find_cutoffs, comb_list)
    
    cutoffs_dict = {}
    for i in cutoffs:
        val1, val2, pos, neg = i
        minval = min(val1, val2)
        maxval = max(val1, val2)
        cutoffs_dict[(minval, maxval)] = (pos, neg)
    
    utils.save_pkl(cutoffs_dict, outdir + 'cutoffs_dict.pkl')
    print('using random subset to sketch communities')
    rands = random.sample(vals, sample_num)

    corr_graph = nx.Graph()

    for j,i in enumerate(combinations(rands, 2)):
        vox1, vox2 = i 
        x = binned[:, vox1].astype('float')
        y = binned[:, vox2].astype('float')
        summed = x + y
        minvox = min(totals[[vox1, vox2]])
        maxvox = max(totals[[vox1, vox2]])
        pos_cutoff, neg_cutoff = cutoffs_dict[(minvox, maxvox)]
        if (len(np.where(summed == 2)[0]) > 2*pos_cutoff) and (len(np.where(summed == 0)[0]) > neg_cutoff): 
            corr_graph.add_edge(vox1, vox2)
            
    blocks_dict = {}
    for i, j in enumerate(nx.algorithms.community.label_propagation.label_propagation_communities(corr_graph)):
        blocks_dict[i] = list(j)
    utils.save_pkl(blocks_dict, outdir + 'blocks_dict.pkl')
    for i in blocks_dict.keys():
        write_vol(i, blocks_dict, outdir, vol_list, union_voxels)
    '''
    print('Bootstrapping p-values for community expansion')
    block_lens = [len(blocks_dict[i]) for i in blocks_dict.keys()]
    corr_factor_exp = len(vals)*np.sum(block_lens)
    posp_corr_exp = args.posp/corr_factor_exp
    negp_corr_exp = args.negp/corr_factor_exp

    comb_list = []
    for i in combinations(np.unique(totals), 2):
        f1, f2 = i
        comb_list.append((f1, f2, num_trials, posp_corr_exp, negp_corr_exp))
    for i in np.unique(totals):
        comb_list.append((i, i, num_trials, posp_corr_exp, negp_corr_exp))
    cutoffs_exp = pool.map(find_cutoffs, comb_list)
    
    cutoffs_dict_exp = {}
    for i in cutoffs_exp:
        val1, val2, pos, neg = i
        minval = min(val1, val2)
        maxval = max(val1, val2)
        cutoffs_dict_exp[(minval, maxval)] = (pos, neg)

    print('expanding communities with full voxel list')
    blocks_dict_expand = copy.deepcopy(blocks_dict)
    unadded_list = []

    for i in vals:
        unadded = True
        x = binned[:, i].astype('float')
        for j in blocks_dict.keys():
            counter = 0
            for k in blocks_dict[j]:
                y = binned[:, k].astype('float')
                summed = x + y
                minvox = min(totals[[i, k]])
                maxvox = max(totals[[i, k]])
                pos_cutoff, neg_cutoff = cutoffs_dict_exp[(minvox, maxvox)]
                if (len(np.where(summed == 2)[0]) > pos_cutoff) and (len(np.where(summed == 0)[0]) > neg_cutoff): 
                    counter += 1
            if counter/len(blocks_dict[j]) >= args.exp_frac:
                blocks_dict_expand[j].append(i) 
                unadded = False
        if unadded:
            unadded_list.append(i)
    
    print('writing volumes and saving data')
    for i in blocks_dict_expand.keys():
        write_vol(i, blocks_dict_expand, outdir, vol_list, union_voxels)
    utils.save_pkl(blocks_dict_expand, outdir + 'blocks_dict_expand.pkl')
    
    t_final = time.time() - t0
    print(f'total run time: {t_final}s')
    '''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())

        
    


