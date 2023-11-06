import os
import numpy as np
from cryodrgn import mrc
import pandas as pd
from torch.utils.data import Dataset
import torch
torch.manual_seed(42)  
from commands.modules import utils

class CustomDataset(Dataset):
    def __init__(self, csv_path, vol_dir):  
        
        df=pd.read_csv(csv_path)

        vol_array=np.empty(shape=(len(df),64,64,64), dtype=np.float32)

        for i, ind in enumerate(df.index):
            file = df.loc[ind, 'map_file']  
            full_file_path = os.path.join(vol_dir, file)
            mrc_vol,_ = utils.load_vol(full_file_path)
            vol_array[i] = mrc_vol 

        self.vol_array = vol_array
        self.labels = df['normalized_label'].to_numpy(dtype=np.float32)
        self.names = df.loc[ind, 'map_file'] 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vol_array[idx], self.labels[idx], idx