from wavenet_model import WaveNetModel

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchaudio

import os

import stempeg
import numpy as np

from auraloss.time import SNRLoss, SISDRLoss, SDSDRLoss, ESRLoss
from auraloss.freq import STFTLoss, MelSTFTLoss, STFTMagnitudeLoss


import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def optimize(model, train_mix, train_voice, vocal_isolation=True):
    
    criterion = STFTLoss(fft_size=2048) if vocal_isolation else SNRLoss() 

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i in range(len(train_mix)):

            optimizer.zero_grad()
            
            if vocal_isolation:
                input_mix = torch.tensor(train_mix[i][0]).unsqueeze(0).to(device=device)
                input_voice = torch.tensor(train_voice[i][0]).unsqueeze(0).to(device=device)
            else:
                input_mix = train_mix[i][0].unsqueeze(0).to(device=device)
                input_voice = train_voice[i][0].unsqueeze(0).to(device=device)

            output = model(input_mix)

            loss = criterion(output, input_voice[:, :, :output.shape[2]])
            
            total_loss += loss.item()

            loss.backward()

            optimizer.step()

        avg_loss = total_loss / len(mix)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    print("Training finished!")