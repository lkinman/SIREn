## Written by Maria Carreira

## October 2023

import os
import torch
torch.manual_seed(42)  
import torch.nn as nn
from torch.optim import *
#import PyQt5
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import argparse
import logging
from siren import model_cnn, data, utils, funcs
import warnings 

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.NOTSET)


def add_args(parser):

    parser.add_argument("--voldir", type=os.path.abspath, required=True, help="Path to subset of downsampled and normalized input volumes")
    parser.add_argument("--labels", type=os.path.abspath, required=True, help="Path to .csv containing subset of normalized labels")
    parser.add_argument("--batch_size", type=int, required=False, default=4, help="Minibatch size")
    parser.add_argument("--num_epochs", type=int, required=False, default=5, help="Number of epochs")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--lr", type=float, default=1e-6, required=False, help="Learning rate for fine-tuning")
    parser.add_argument("--outdir", type=str, default = './', required=True, help="Path to output directory")

    return parser


def main(args):
    csv_path = args.labels
    voldir = args.voldir
    batch_size = args.batch_size
    epochs = args.num_epochs
    weights = args.weights
    learning_rate = args.lr
    outdir = funcs.check_dir(args.outdir, make=True)
    
    logging.info("Loading data")
    ds = data.CustomDataset(csv_path, voldir)
    train, test, val = random_split(ds, [0.8, 0.1, 0.1])  
    torch.set_default_dtype(torch.float32)
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    
    logging.info("Loading model")
    model = model_cnn.CNNModel()
    save_path = weights
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    error = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0) 
    scaler = torch.cuda.amp.GradScaler()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model= model.to(device)

    logging.info("Start training")
    training_loss_avg_list = utils.train_cnn(model, train_loader, epochs, optimizer, device, error, scaler)
    
    logging.info("Saving weights")
    name = 'fine_tuned_weights.pth'
    utils.save_model(epochs, model, optimizer, error, outdir, name)

    logging.info("Plotting")
    num_epoch=range(0,args.num_epochs)
    logging.getLogger('matplotlib.font_manager').disabled = True
    plt.plot(num_epoch, training_loss_avg_list, 'r', label='Training loss')   
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=10)
    file_name = 'model_loss.pdf'
    outfile = os.path.join(outdir, file_name)
    plt.savefig(outfile)  
    plt.show()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())
