import torch.nn as nn

#Convolutional layers adapted from https://www.nature.com/articles/s41598-022-19212-6

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv3d(1, 32, 7, 1, padding_mode='replicate'),
                            nn.BatchNorm3d(32),
                            nn.ReLU(),
                            nn.Conv3d(32, 64, 5, 1, padding_mode='replicate'),
                            nn.BatchNorm3d(64),
                            nn.ReLU(),
                            nn.MaxPool3d(2),
                            nn.Conv3d(64,128, 5, 1, padding_mode='replicate'),
                            nn.BatchNorm3d(128),
                            nn.ReLU(),
                            nn.MaxPool3d(2),
                            nn.Conv3d(128, 256, 3, 1, padding_mode='replicate'),
                            nn.BatchNorm3d(256),
                            nn.ReLU(),
                            nn.MaxPool3d(2),
                            nn.Conv3d(256,384, 1, 1, padding_mode='replicate'),
                            nn.ReLU(),
                            nn.AvgPool3d(3, stride=2),
                            nn.Dropout(0.3))

        self.mlp = nn.Sequential(nn.Linear(in_features=384, out_features=64),
                    nn.ReLU(),
                    nn.Linear(in_features=64, out_features=1))


    def forward(self, x):
        x=x.view(-1, 1, 64, 64, 64)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x