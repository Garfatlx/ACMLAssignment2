import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


# Define the convolutional autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self,inputchannels, encoderlayers, decoderlayers, stride, padding):
        super(ConvAutoencoder, self).__init__()
        self.inputchannels = inputchannels
        self.encoderlayers = encoderlayers
        self.decoderlayers = decoderlayers
        self.stride = stride
        self.padding = padding

        # Define encoder
        self.encoder = nn.Sequential()
        in_channels = self.inputchannels  # Assuming input has 3 channels (e.g., RGB images)
        for i, out_channels in enumerate(encoderlayers):
            self.encoder.add_module(f'conv_{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding))
            self.encoder.add_module(f'relu_{i}', nn.ReLU())
            self.encoder.add_module(f'pool_{i}', nn.MaxPool2d(2, 2))
            in_channels = out_channels

        # Define decoder
        self.decoder = nn.Sequential()
        for i, out_channels in enumerate(decoderlayers):
            self.decoder.add_module(f'upsample_{i}', nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder.add_module(f'conv_{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding))
            if i == len(decoderlayers) - 1:
                self.decoder.add_module(f'sigmoid_{i}', nn.Sigmoid())
            else:
                self.decoder.add_module(f'relu_{i}', nn.ReLU())
            in_channels = out_channels

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

