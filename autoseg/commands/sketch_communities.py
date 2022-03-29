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
    parser.add_argument('--voldir', type = str, required = True, help = 'Directory where (downsampled) volumes are stored')
    parser.add_argument('--threads', type = int, required = True, help = 'Number of threads for multiprocessing')
    parser.add_argument('--bin', type = float, default = None, help = 'Threshold at which to binarize maps')
    parser.add_argument('--outdir', type = str, default = './', help = 'Directory where outputs will be stored')
    parser.add_argument('--posp', type = float, default = 0.01, help = 'P value threshold before Bonferroni correction for positive co-occupancy')
    parser.add_argument('--posp_factor', type = float, default = 2, help = 'Factor by which to multiply Bonferroni-corrected p-value for positive co-occupancy')
    parser.add_argument('--negp', type = float, default = 0.05, help = 'P value threshold before Bonferroni correction for negative co-occupancy')
    parser.add_argument('--negp_factor', type = float, default = 1, help = 'Factor by which to multiply Bonferroni-corrected p-value for negative co-occupancy')
    return parser

def main(args):
    t0 = time.time()
    
    configs_dict = {'voldir': args.voldir, 'threads': args.threads, 'bin': args.bin, 'outdir': args.outdir, 'posp': args.posp, 'negp': args.negp, 'posp_factor': args.posp_factor, 'negp_factor': args.negp_factor}

    sketch_outdir = funcs.check_dir(args.outdir + '00_sketch/', make = True)
    voldir = funcs.check_dir(args.voldir)
    vol_list = np.sort(glob.glob(voldir + '*.mrc'))
    num_vols = len(vol_list)
    boxsize = mrc.parse_mrc(vol_list[0])[0].shape[0]
    num_voxels = boxsize**3
    print(num_voxels)
    print(len(vol_list))
    binned, union_voxels = funcs.binarize_vol_array(vol_list, num_vols, num_voxels, args.bin)
    totals = np.sum(binned, axis = 0)
    vals = range(0, len(union_voxels))
    sample_num = int(len(union_voxels)/3)
    rands = random.sample(vals, sample_num)

    print('Calculating corrected p-values')
    corr_factor = comb(sample_num, 2)
    print(corr_factor)
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
    cutoffs = pool.map(funcs.find_cutoffs, comb_list)
    
    cutoffs_dict = {}
    for i in cutoffs:
        val1, val2, pos, neg = i
        minval = min(val1, val2)
        maxval = max(val1, val2)
        cutoffs_dict[(minval, maxval)] = (pos, neg)
    
    utils.save_pkl(cutoffs_dict, sketch_outdir + 'cutoffs_dict.pkl')

    print('using random subset to sketch communities')

    corr_graph = nx.Graph()

    for j,i in enumerate(combinations(rands, 2)):
        vox1, vox2 = i 
        x = binned[:, vox1].astype('float')
        y = binned[:, vox2].astype('float')
        summed = x + y
        minvox = min(totals[[vox1, vox2]])
        maxvox = max(totals[[vox1, vox2]])
        pos_cutoff, neg_cutoff = cutoffs_dict[(minvox, maxvox)]
        if (len(np.where(summed == 2)[0]) >args.posp_factor*pos_cutoff) and (len(np.where(summed == 0)[0]) > args.negp_factor*neg_cutoff): 
            corr_graph.add_edge(vox1, vox2)
            
    blocks_dict = {}
    for i, j in enumerate(nx.algorithms.community.label_propagation.label_propagation_communities(corr_graph)):
        if len(list(j)) > 10:
        blocks_dict[i] = list(j)
    
    print('writing volumes and saving data')
    for i in blocks_dict.keys():
        funcs.write_vol(i, blocks_dict, sketch_outdir, vol_list, union_voxels)
    utils.save_pkl(blocks_dict, sketch_outdir + 'blocks_dict.pkl')
    utils.save_pkl(configs_dict, sketch_outdir + 'config.pkl')
    
    t_final = time.time() - t0
    print(f'total run time: {t_final}s')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())

        
    


