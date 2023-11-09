## Written by Maria Carreira

## October 2023

import numpy as np
import mrcfile 
import torch
torch.manual_seed(42)  
import logging
logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.NOTSET)
import pickle

def dump_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    return

def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_vol(filename):
    with mrcfile.open(filename, 'r', permissive=True) as mrc:
        vol_data = mrc.data.copy()
        pixel_size = mrc.voxel_size.x
    return vol_data, pixel_size


def check_boxsize(vol):
    return vol.shape[0]


def downsample(vol, downsampled_boxsize, boxsize):  
    vol_ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol)))
    start = boxsize // 2 - downsampled_boxsize // 2
    stop = boxsize // 2 + downsampled_boxsize // 2
    vol_ft_d = vol_ft[start:stop, start:stop, start:stop]
    vol_d = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(vol_ft_d)))
    vol_d = vol_d.real
    vol_d*=((downsampled_boxsize/boxsize)**3)
    vol_d = vol_d.astype(np.float32)
    return vol_d


def normalize(vol_d, upper_thr, lower_thr):
    d_upper = np.percentile(vol_d, upper_thr)
    d_lower = np.percentile(vol_d, lower_thr)

    vol_d = np.where(vol_d > d_upper, d_upper, vol_d)
    vol_d = np.where(vol_d < d_lower, d_lower, vol_d)
    
    vol_d_norm = np.where(vol_d <= 0, 0, vol_d)
    vol_d_min = np.min(vol_d[vol_d > 0])
    vol_d_max = np.max(vol_d_norm)
    vol_d_norm = (vol_d_norm - vol_d_min)/(vol_d_max - vol_d_min)
    vol_d_norm = 2*vol_d_norm - 1
    return vol_d_norm, vol_d_min, vol_d_max


def write(vol, outname, downsampled_angpix):
    with mrcfile.new(outname, overwrite=True) as output_mrc:                     
        output_mrc.set_data(vol)
        output_mrc.update_header_from_data()
        output_mrc.voxel_size = downsampled_angpix
    return output_mrc


def save_model(epochs, model, optimizer, error, outdir, name):
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': error,
                }, f'{outdir}/{name}')
    

def train_cnn(model, train_loader, epochs, optimizer, device, error, scaler):
  
    training_loss_avg_list=[]

    for epoch in range(epochs):

        training_loss_list=[]

        model.train()
        for step, (x_batch_train, y_batch_train, idx) in enumerate(train_loader): 
            x_batch_train=x_batch_train.to(device)        
            y_batch_train=y_batch_train.to(device) 
            
            with torch.cuda.amp.autocast():
                with torch.autograd.set_detect_anomaly(True):
                    optimizer.zero_grad()
                    pred = model(x_batch_train)
                    pred = pred.to(device)
                    
                    #ref: https://stackoverflow.com/questions/71418817/is-there-any-way-to-include-a-countera-variable-that-count-something-in-a-loss
                    #https://discuss.pytorch.org/t/training-with-threshold-in-pytorch/145962
                    v1 = torch.sigmoid(10*(x_batch_train-pred.reshape(-1, 1, 1, 1)))
                    v2 = torch.sigmoid(10*(x_batch_train-y_batch_train.reshape(-1, 1, 1, 1)))
                    training_loss=error(v1, v2)

            if not torch.isnan(training_loss):
                scaler.scale(training_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            training_loss_list.append(training_loss.item())

        training_loss_avg=np.mean(training_loss_list)  
        training_loss_avg_list.append(training_loss_avg)
    
        logging.info("Epoch: {}".format(epoch))
        logging.info("Training loss: {}".format(training_loss_avg_list[epoch]))
        
    return training_loss_avg_list


def model_pred(model, data_generator, device):
    model.eval()

    pred_list = []

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for (idx, x) in enumerate(data_generator):
                x = x.to(device)
                prediction = model(x)
                prediction = (prediction.squeeze(dim=1))
                prediction = prediction.cpu().numpy()
                pred_list.append(prediction)

    return pred_list
