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
from autoseg import funcs

def add_args(parser):
    parser.add_argument('--config', type = str,  required = True, help = 'Path to sketch_communities.py config file')
    parser.add_argument('--blockdir', type = str, required = True, help = 'Path to directory where segmented blocks are stored')
    parser.add_argument('--threads', type = int,  required = True, help = 'Number of threads for multiprocessing')
    parser.add_argument('--exp_frac', type = float, default = 0.25, help = '')
    parser.add_argument('--posp', type = float, default = 0.01, help = 'P value threshold before Bonferroni correction for positive co-occupancy')
    parser.add_argument('--posp_factor', type = float, default = 2, help = 'Factor by which to multiply Bonferroni-corrected p-value for positive co-occupancy')
    parser.add_argument('--negp', type = float, default = 0.05, help = 'P value threshold before Bonferroni correction for negative co-occupancy')
    parser.add_argument('--negp_factor', type = float, default = 1, help = 'Factor by which to multiply Bonferroni-corrected p-value for negative co-occupancy')
    return parser

def main(args):
    t0 = time.time()
    configs_dict = utils.load_pkl(args.config)
    outdir = funcs.check_dir(configs_dict['outdir'])
    expand_outdir = funcs.check_dir(outdir + '02_expand', make = True)
    voldir = funcs.check_dir(configs_dict['voldir'])
    blockdir = funcs.check_dir(args.blockdir)
    vol_list = np.sort(glob.glob(voldir + '*.mrc'))
    num_vols = len(vol_list)
    boxsize = mrc.parse_mrc(vol_list[0])[0].shape[0]
    num_voxels = boxsize**3
    binned, union_voxels = funcs.binarize_vol_array(vol_list, num_vols, num_voxels, configs_dict['bin'])
    totals = np.sum(binned, axis = 0)
    vals = range(0, len(union_voxels))

    print('read in segmentation-masked blocks')
    blocks_dict = funcs.read_blocks(blockdir, union_voxels)

    print('Bootstrapping p-values for community expansion')
    block_lens = [len(blocks_dict[i]) for i in blocks_dict.keys()]
    corr_factor = len(vals)*np.sum(block_lens)
    posp_corr_exp = 1-(args.posp/corr_factor)
    negp_corr_exp = 1-(args.negp/corr_factor)

    comb_list = []
    num_trials = 1000
    pool = Pool(args.threads)
    for i in combinations(np.unique(totals), 2):
        f1, f2 = i
        comb_list.append((f1, f2, num_trials, posp_corr_exp, negp_corr_exp))
    for i in np.unique(totals):
        comb_list.append((i, i, num_trials, posp_corr_exp, negp_corr_exp))
    cutoffs = pool.map(funcs.find_cutoffs, comb_list)
    
    cutoffs_dict = {}
    for i in cutoffs:
        val1, val2, pos, neg = i
        minval = min(val1, val2)
        maxval = max(val1, val2)
        cutoffs_dict[(minval, maxval)] = (pos, neg)

    utils.save_pkl(cutoffs_dict, expand_outdir + 'cutoffs_dict.pkl')

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
                pos_cutoff, neg_cutoff = cutoffs_dict[(minvox, maxvox)]
                if (len(np.where(summed == 2)[0]) > pos_cutoff) and (len(np.where(summed == 0)[0]) > neg_cutoff): 
                    counter += 1
            if counter/len(blocks_dict[j]) >= args.exp_frac:
                blocks_dict_expand[j].append(i) 
                unadded = False
        if unadded:
            unadded_list.append(i)
    
    print('writing volumes and saving data')
    for i in blocks_dict_expand.keys():
        funcs.write_vol(i, blocks_dict_expand, expand_outdir, vol_list, union_voxels)
    utils.save_pkl(blocks_dict_expand, expand_outdir + 'blocks_dict_expand.pkl')
    utils.save_pkl(configs_dict, expand_outdir + 'config.pkl')
    t_final = time.time() - t0
    print(f'total run time: {t_final}s')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())

        
    


