## Written by Maria Carreira

## October 2023


import os
import torch
import numpy as np
import glob
import pandas as pd
import argparse
torch.manual_seed(42)  
from modules import model_cnn
import matplotlib.pyplot as plt
import logging
from natsort import natsorted
from siren import utils
import warnings 

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def parse_args(parser):

    parser.add_argument("-vol_dir", type=os.path.abspath, required=True, help="Path to input volume (.mrc) or directory containing volumes")
    parser.add_argument("-normalize_csv", type=os.path.abspath, required=True, help="map_stats.csv (either downsampled or raw map stats)")
    parser.add_argument("-labels", type=os.path.abspath, required=False, help="User-annotated labels for downsampled (non-normalized) volumes for evaluating model performance")
    parser.add_argument("-weights_file", required=True, help="Path to model weights (weights.pth or fine_tuned_weights.pth)")
    parser.add_argument("-batch_size", type=int, required=False, default=4, help="Minibatch size")
    parser.add_argument("-outdir", type=os.path.abspath, required=True, help="Path to output directory")

    return parser


def main(args):

    vol_dir = args.vol_dir
    weights_file = args.weights_file
    batch_size = args.batch_size
    outdir = args.outdir
    labels = args.labels
    norm_df = args.normalize_csv


    if os.path.isdir(vol_dir):
        vol_list = natsorted(glob.glob(vol_dir + '/*.mrc'))  
    else:
        vol_list = [vol_dir,]
    
    vol_array = np.empty(shape=(len(vol_list),64,64,64), dtype=np.float32)
    for i in range(len(vol_list)):

        mrc_vol,_ = utils.load_vol(vol_list[i])
        vol_array[i] = mrc_vol 

    torch.set_default_dtype(torch.float32)
    logging.info("Loading data")
    data_generator = torch.utils.data.DataLoader(vol_array, batch_size = batch_size, shuffle=False)

    logging.info("Loading model")
    model = model_cnn.CNNModel()
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(weights_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    optimizer.load_state_dict(torch.load(weights_file)['optimizer_state_dict'])

    model = model.to(device)
 
    logging.info("Making predictions")
    pred_list = utils.model_pred(model, data_generator, device)

    logging.info("Saving dataframe with predictions")
    all_preds = np.concatenate(pred_list, axis = 0)
    norm_df = pd.read_csv(norm_df)
    norm_df['prediction'] = all_preds

    logging.info("Denormalize predictions")
    norm_df['int_predictions'] = (norm_df['prediction'] + 1)/2
    norm_df['denormalized_predictions'] = (norm_df['int_predictions'] * (norm_df['vol_max'] - norm_df['vol_min'])) + norm_df['vol_min']
    file_name = 'predictions.csv'
    outfile = os.path.join(outdir, file_name)
    norm_df.to_csv(outfile) 

    if labels:
        logging.info("Plotting")
        logging.getLogger('matplotlib.font_manager').disabled = True
        user_labels = pd.read_csv(labels, index_col = 'vol_id').sort_values('vol_id')
        user_labels = user_labels.sort_values(by='contour_level')
        plt.errorbar(range(len(user_labels)), user_labels['contour_level'], yerr = [user_labels['contour_level'] - user_labels['lower_bound'], user_labels['upper_bound'] - user_labels['contour_level']], fmt = '.', color = 'black')
        plt.scatter(range(len(user_labels)), norm_df.loc[user_labels.index, 'denormalized_predictions'])
        file_name = 'predictions_vs_labels.pdf'
        outfile = os.path.join(outdir, file_name)
        plt.ylabel('Labels')
        plt.xlabel('Volumes')
        plt.title('User-annotated labels vs. CNN predictions')
        plt.savefig(outfile)  
        plt.show()

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description=__doc__)
    args = parse_args(parser).parse_args()  
    main(args)


