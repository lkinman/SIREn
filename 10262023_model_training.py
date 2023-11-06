## Written by Maria Carreira

## October 2023

import os
import torch
torch.manual_seed(42)  
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import argparse
import logging
import pickle
from siren import model_cnn, data, utils
import warnings 

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.NOTSET)

def parse_args(parser):

    parser.add_argument("-vol_dir", type=os.path.abspath, required=True, help="Path to downsampled and normalized input volumes")
    parser.add_argument("-labels_csv", type=os.path.abspath, required=True, help="Path to .csv containing normalized labels")
    parser.add_argument("-batch_size", type=int, required=False, default=8, help="Minibatch size")
    parser.add_argument("-num_epochs", type=int, required=False, default=10, help="Number of epochs")
    parser.add_argument("-outdir", type=os.path.abspath, required=True, help="Path to output directory")

    return parser


def main(args):
    csv_path=args.labels_csv
    vol_dir=args.vol_dir
    batch_size=args.batch_size
    epochs = args.num_epochs
    outdir=args.outdir

    logging.info("Loading data")
    ds = data.CustomDataset(csv_path, vol_dir)
    train, test, val = random_split(ds, [0.8, 0.1, 0.1]) 
    torch.set_default_dtype(torch.float32)
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

    logging.info("Saving training data as .pkl")
    output_file = f'{args.outdir}/train.pkl'
    train_data = list(train_loader)

    with open(output_file, 'wb') as f:
        pickle.dump(train_data, f)
    
    logging.info("Loading model")
    model = model_cnn.CNNModel()
    error = nn.MSELoss() 
    learning_rate =  0.00001  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0) 
    scaler = torch.cuda.amp.GradScaler()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    logging.info("Start training")
    training_loss_avg_list = utils.train_cnn(model, train_loader, epochs, optimizer, device, error, scaler)

    logging.info("Saving weights")
    name = 'weights.pth'
    utils.save_model(epochs, model, optimizer, error, outdir, name)

    logging.info("Plotting")
    num_epoch=range(0, epochs)
    logging.getLogger('matplotlib.font_manager').disabled = True
    plt.plot(num_epoch, training_loss_avg_list, 'r', label='Training loss')   
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=10)
    file_name = 'training_model_loss.pdf'
    outfile = os.path.join(outdir, file_name)
    plt.savefig(outfile)  
    plt.show()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description=__doc__)
    args = parse_args(parser).parse_args()
    main(args)
