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
from scipy.special import comb
import networkx as nx
from siren import funcs, utils
import re
import logging

logging.getLogger('matplotlib').disabled = True

def add_args(parser):
    parser.add_argument('--voldir', type = str, required = True, help = 'Directory where (downsampled) volumes are stored')
    parser.add_argument('--threads', type = int, required = True, help = 'Number of threads for multiprocessing')
    parser.add_argument('--apix', type = float, required = True, help = 'Angstroms per pixel of maps')
    parser.add_argument('--bin', type = float, default = None, help = 'Threshold at which to binarize maps')
    parser.add_argument('--outdir', type = str, default = './', help = 'Directory where outputs will be stored')
    parser.add_argument('--posp', type = float, default = 0.01, help = 'P value threshold before Bonferroni correction for positive co-occupancy')
    parser.add_argument('--negp', type = float, default = 0.05, help = 'P value threshold before Bonferroni correction for negative co-occupancy')
    parser.add_argument('--binfile', type = str, default = None, help = 'csv file containing predicted binarization thresholds')
    parser.add_argument('--filter', action = 'store_true', help = 'If predicted binarization file is supplied, this flage indicates that volumes should be filtered by avg +/- 2 std of predicted threshold')
    parser.add_argument('--subsample', type = int, default = 3, help = 'divisor for subsampling voxels')
    return parser

def main(args):
    t0 = time.time()
    if args.filter:
        filter_vols = True
    else:
        filter_vols = False
    if args.bin:
        bin_thr = args.bin
    else:
        bin_thr = args.binfile
    configs_dict = {'voldir': args.voldir, 'threads': args.threads, 'bin': bin_thr, 'outdir': args.outdir, 'posp': args.posp, 'negp': args.negp, 'apix': args.apix}
    outdir = funcs.check_dir(args.outdir, make = True)
    sketch_outdir = funcs.check_dir(outdir + '00_sketch/', make = True)
    voldir = funcs.check_dir(args.voldir)

    vol_list = glob.glob(voldir + '*.mrc')
    vol_list.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.mrc')[0]))
    num_vols = len(vol_list)
    boxsize = utils.load_vol(vol_list[0])[0].shape[0]
    num_voxels = boxsize**3
    binned, union_voxels, vols_added = funcs.binarize_vol_array(vol_list, num_vols, num_voxels, bin_thr, filter_bin = filter_vols)
    num_vols = len(binned)
    map_array = funcs.create_mapping(vol_list, union_voxels)
    totals = np.sum(binned, axis = 0)
    vals = range(0, len(union_voxels))
    sample_num = int(len(union_voxels)/args.subsample)
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
        comb_list.append((f1, f2, num_trials, posp_corr, negp_corr, num_vols))
    for i in np.unique(totals):
        comb_list.append((i, i, num_trials, posp_corr, negp_corr, num_vols))
    pool = Pool(args.threads)
    cutoffs = pool.map(funcs.find_cutoffs, comb_list)
    
    cutoffs_dict = {}
    for i in cutoffs:
        val1, val2, pos, neg = i
        minval = min(val1, val2)
        maxval = max(val1, val2)
        cutoffs_dict[(minval, maxval)] = (pos, neg)
    
    utils.dump_pkl(cutoffs_dict, sketch_outdir + 'cutoffs_dict.pkl')

    print('using random subset to sketch communities')

    corr_graph = nx.Graph()

    for j,i in enumerate(combinations(rands, 2)):
        vox1, vox2 = i 
        x = binned[:, vox1].astype('float')
        y = binned[:, vox2].astype('float')
        summed = x + y
        minvox = min(totals[[vox1, vox2]])
        maxvox = max(totals[[vox1, vox2]])
        vox_dist = np.linalg.norm(np.array(re.findall('..', map_array[vox1])).astype('int') - np.array(re.findall('..', map_array[vox2])).astype('int'))*args.apix
        if vox_dist < 40:
            posp_factor = max(1.25, 0.02*vox_dist+0.95)
        else:
            posp_factor = 1.75
        negp_factor = 1 
        pos_cutoff, neg_cutoff = cutoffs_dict[(minvox, maxvox)]
        if (len(np.where(summed == 2)[0]) > min(posp_factor*pos_cutoff, minvox)) and (len(np.where(summed == 0)[0]) > min(negp_factor*neg_cutoff, num_vols - maxvox)): 
            corr_graph.add_edge(vox1, vox2)
            
    blocks_dict = {}
    for i, j in enumerate(nx.algorithms.community.label_propagation.label_propagation_communities(corr_graph)):
        if len(list(j)) > 5:
            blocks_dict[i] = list(j)
    
    fig, ax = plt.subplots(1, figsize = (30, 10))
    nx.draw_networkx(corr_graph, pos = nx.spring_layout(corr_graph), node_size = 30, with_labels = False)
    fig.savefig(sketch_outdir + 'corr_graph.pdf')
    
    print('writing volumes and saving data')
    for i in blocks_dict.keys():
        funcs.write_vol(i, blocks_dict, sketch_outdir, vol_list, union_voxels, boxsize, args.apix)
    utils.dump_pkl(blocks_dict, sketch_outdir + 'blocks_dict.pkl')
    utils.dump_pkl(configs_dict, sketch_outdir + 'config.pkl')
    
    t_final = time.time() - t0
    print(f'total run time: {t_final}s')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())

        
    


