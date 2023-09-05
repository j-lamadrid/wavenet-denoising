import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchaudio

import stempeg
import numpy as np

import os

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_musdb(subset):
    
    assert subset in ['train', 'test']
    
    mix = []
    vocals = []

    for filename in os.listdir('musdb18/{}'.format(subset)):
        
        if filename == '.ipynb_checkpoints':
            continue
            
        # Pull training sample from sparser/quieter region
        audio, sample_rate = stempeg.read_stems('musdb18/{}/'.format(subset) + filename,
                                                dtype=np.float32,
                                                start=30,
                                                duration=10)

        mix.append((audio[0].T, sample_rate))
        vocals.append((audio[4].T, sample_rate))
        
        # Pull training sample from more populated/louder region
        audio, sample_rate = stempeg.read_stems('musdb18/{}/'.format(subset) + filename, 
                                                dtype=np.float32,
                                                start=60,
                                                duration=10)

        mix.append((audio[0].T, sample_rate))
        vocals.append((audio[4].T, sample_rate))
        

        audio, sample_rate = stempeg.read_stems('musdb18/{}/'.format(subset) + filename, 
                                                dtype=np.float32,
                                                start=45,
                                                duration=10)

        mix.append((audio[0].T, sample_rate))
        vocals.append((audio[4].T, sample_rate))
        
    mix_out = []
    vocals_out = []

    for i in range(len(vocals)):
        if np.mean(abs(vocals[i][0][0]) + abs(vocals[i][0][1])) >= 0.05:
            mix_out.append(mix[i])
            vocals_out.append(vocals[i])

    return mix_out, vocals_out

def process_speech():
    
    mix_out = []
    speech_out = []

    for i in range(len(os.listdir('noisy_speech/noisy_trainset_28spk_wav'))):
        
        mix_out.append(torchaudio.load('noisy_speech/noisy_trainset_28spk_wav/' + sorted(os.listdir('noisy_speech/noisy_trainset_28spk_wav'))[i]))
        speech_out.append(torchaudio.load('noisy_speech/clean_trainset_28spk_wav/' + sorted(os.listdir('noisy_speech/clean_trainset_28spk_wav'))[1:][i]))

    return mix_out, speech_out