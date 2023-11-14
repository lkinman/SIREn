## Written by Maria Carreira

## October 2023

import os
import numpy as np
import pandas as pd
import argparse
import glob
import warnings 
import logging
from siren import utils, funcs
import torch
torch.manual_seed(42)
from natsort import natsorted

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def add_args(parser):

    parser.add_argument("--voldir", type=os.path.abspath, required=True, help="Path to input volume (.mrc) or directory containing volumes")
    parser.add_argument("--labels", type=os.path.abspath, required=False, help="User-annotated labels for downsampled (non-normalized) volumes for normalization")
    parser.add_argument("--outdir", type=str, default = './', required=True, help="Path to output directory for normalized volumes")
    return parser


def main(args):
    voldir = args.voldir
    labels = args.labels
    outdir = funcs.check_dir(args.outdir + 'normalized/', make=True)
    outdir_downsampled = funcs.check_dir(outdir + 'downsampled/', make = True)

    upper_thr = 99.999
    lower_thr = 0.001
    vol_d_min_list = []
    vol_d_max_list = []
    vol_min_list = []
    vol_max_list = []
    downsampled_boxsize = 64

    if os.path.isdir(voldir):
        vol_list = natsorted(glob.glob(voldir + '/*.mrc'))  
    else:
        vol_list = [voldir,]
    

    logging.info("Loading data")
    for filename in vol_list:
        vol, pixel_size = utils.load_vol(filename)
        boxsize = utils.check_boxsize(vol)
        if boxsize != 64:
            logging.info(f"Downsampling: {filename}")
            vol_d = utils.downsample(vol, downsampled_boxsize, boxsize)
            vol_norm, vol_min, vol_max = utils.normalize(vol, upper_thr, lower_thr)
            basename = os.path.basename(filename)
            outname = os.path.join(outdir_downsampled, basename) 
            downsampled_angpix = pixel_size*boxsize/downsampled_boxsize
            utils.write(vol_d, outname, downsampled_angpix)
            vol_min_list.append(vol_min)
            vol_max_list.append(vol_max)

        else:
            vol_d = vol

        logging.info(f"Normalizing: {filename}")
        vol_d_norm, vol_d_min, vol_d_max = utils.normalize(vol_d, upper_thr, lower_thr)
        basename = os.path.basename(filename)
        outname = os.path.join(outdir, basename)
        downsampled_angpix = pixel_size
        utils.write(vol_d_norm, outname, downsampled_angpix)

        vol_d_min_list.append(vol_d_min)
        vol_d_max_list.append(vol_d_max)
    
    logging.info("Saving downsampled map statistics")
    df = pd.DataFrame()
    df['vol_min'] = vol_d_min_list   
    df['vol_max' ] = vol_d_max_list  
    df.index.name = 'vol_id'
    df = df.reset_index()                
    file_name = 'map_stats_downsampled.csv'
    outfile = os.path.join(outdir, file_name)
    df.to_csv(outfile) 

    if boxsize != 64:
        logging.info("Saving full box map statistics")
        df_full = pd.DataFrame()
        df_full['vol_min'] = vol_min_list   
        df_full['vol_max' ] = vol_max_list  
        df_full.index.name = 'vol_id'
        df_full = df_full.reset_index()                
        file_name = 'map_stats_raw.csv'
        outfile = os.path.join(outdir, file_name)
        df_full.to_csv(outfile) 

    if labels:
        logging.info("Normalizing labels")
        df1 = pd.read_csv(labels)
        for i in df1.index:
            label = df1.loc[i, 'contour_level'] 
            vol_id = df1.loc[i, 'vol_id']
            label_norm = 2*(label - df.loc[vol_id, 'vol_min'])/(df.loc[vol_id, 'vol_max'] - df.loc[vol_id, 'vol_min']) - 1  
            df1.loc[i, 'normalized_label'] = label_norm 
        
        merged_df = pd.merge(df1, df, on='vol_id', how='left')
        file_name = 'transformed_labels.csv'
        outfile = os.path.join(outdir, file_name)
        merged_df.to_csv(outfile) 


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())
