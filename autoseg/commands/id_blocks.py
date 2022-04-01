import argparse
import pandas as pd
import numpy as np
import os
import glob
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
    parser.add_argument('--p', type = float, default = 99, help = 'Percentile occupancy of blocks to set as high threshold')
    return parser

def main(args):
    configs_dict = utils.load_pkl(args.config)
    outdir = funcs.check_dir(configs_dict['outdir'])
    ided_outdir = funcs.check_dir(configs_dict['outdir'] + '03_blocks', make = True)
    voldir = funcs.check_dir(configs_dict['voldir'])
    blockdir = funcs.check_dir(args.blockdir)
    vol_list = np.sort(glob.glob(voldir + '*.mrc'))
    num_vols = len(vol_list)
    boxsize = mrc.parse_mrc(vol_list[0])[0].shape[0]
    num_voxels = boxsize**3
    binned, union_voxels = funcs.binarize_vol_array(vol_list, num_vols, num_voxels, configs_dict['bin'])
    blocks_dict = funcs.read_blocks(blockdir, union_voxels)
    for i in blocks_dict:
        occs = np.sum(binned[:, blocks_dict[i]], axis = 1)/len(blocks_dict[i])
        high_thr = np.percentile(np.sum(binned[:, blocks_dict[i]], axis = 1)/len(blocks_dict[i]), args.p)
        high_occs = np.where(occs > high_thr)[0]
        low_occs = np.where(occs == 0)[0]
        high_occs_union = np.where(np.sum(binned[high_occs, :], axis = 0) == len(high_occs), 1, 0)
        low_occs_intersect = np.where(np.sum(binned[low_occs, :], axis = 0) > 1, 1, 0)
        vox_sel = np.where(high_occs_union-low_occs_intersect == 1)[0]

        if len(vox_sel) > 30:
            outfile = outdir + f'blocks{i}.mrc'
            data, head = mrc.parse_mrc(vol_list[0])
            corrgroup = union_voxels[vox_sel]
            data[:] = 0
            data = data.flatten()
            data[corrgroup] = 1
            mrc.write(outfile, data.reshape(64, 64, 64), header = head)
            
        configs_dict['p'] = args.p
        utils.save_pkl(configs_dict, ided_outdir + 'configs_dict.pkl')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())

        
    


