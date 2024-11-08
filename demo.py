from math import *
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.linalg import toeplitz
from scipy.signal import correlate
import matplotlib.pyplot as plt
import torch

import os
import pickle
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stingray import lightcurve
from stingray.bispectrum import Bispectrum

import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report

from models import *
from bispectral import *

DATA_FOLDER = "/Volumes/ARMAAN USB/LA"

def process_data_handcrafted(filename):
    with open(filename, 'rb') as f:
        data, samplerate = sf.read(filename)
        bicoherence, f = windowed_bicoherence_features(data, 60, window=64, step_size=32, L=10, kmin=0.004, kmax=0.0125)
        bicoherence_final = np.hstack((bicoherence, f))
        bicoherence, f = windowed_bicoherence_features(data, 60, window=64, step_size=32, L=15, kmin=0.004, kmax=0.0125)
        bicoherence_final = np.hstack((bicoherence_final, f))
        bicoherence, f = windowed_bicoherence_features(data, 60, window=64, step_size=32, L=20, kmin=0.004, kmax=0.0125)
        bicoherence_final = np.hstack((bicoherence_final, f))
        # bicoherence, f = windowed_bicoherence_features(data, 60, window=64, step_size=32, L=5, kmin=0.004, kmax=0.0125)
        # bicoherence_final = np.hstack((bicoherence_final, f))
    return bicoherence_final

def process_data_deep(filename):
    duration = 6
    x, fs = sf.read(filename)
    if len(x) < duration * fs:
        x = np.tile(x, int((duration * fs) // len(x)) + 1)
        x = x[0: (int(duration * fs))]
    sample = torch.tensor(x, dtype=torch.float32)
    sample = torch.unsqueeze(sample, 0)
    sample = torch.unsqueeze(sample, 0)
    return sample

def load_deep_model():
    net = DilatedNet()
    net.eval()
    num_total_learnable_params = sum(i.numel() for i in net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    test_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./Inc_TSSDNet_time_frame_28_ASVspoof2019_LA_Loss_0.0043_dEER_1.09%_eEER_4.04%.pth', map_location=test_device)
    net.load_state_dict(model['model_state_dict'])
    return net

def load_handcrafted_model():
    model = pickle.load(open('handcrafted.sav', 'rb'))
    return model

def gen_sample(label):
    labels = get_labels('eval')
    if label == 'bonafide':
        sample = labels.loc[labels['System'] == '-']
    elif label == 'spoofed':
        sample = labels.loc[labels['System'] != '-']
    sample = sample.sample(n=1, replace=False)
    return sample['File'].tolist()[0]

def gen_deep_pred(model):
    labels = get_labels('eval')
    systems = labels['System'].unique()
    rand_samples = dict()
    human_samples, fake_samples = 650, 650
    for system in systems:
        print(system)
        if system == '-':
            samples = labels.loc[labels['System'] == system]
            samples = samples.sample(n=len(samples), replace=False)
            rand_samples['bonafide'] = list(samples['File'])
        else:
            samples = labels.loc[labels['System'] == system]
            samples = samples.sample(n=len(samples), replace=False)
            rand_samples[system] = list(samples['File'])
    correct = 0
    pred = []
    labels = []
    for sample in rand_samples.keys():
        print('Generating samples for {0}'.format(sample))
        if sample == 'bonafide':
            pbar = tqdm(total=human_samples)
        else:
            pbar = tqdm(total=int(fake_samples/(len(rand_samples.keys())-1)))
        i = 0
        for_class = 0
        while True:
            try:
                deep_features = process_data_deep(os.path.join(DATA_FOLDER, 'ASVspoof2019_LA_{0}'.format('eval'), 'flac', '{0}.flac'.format(rand_samples[sample][i])))
                i += 1
                for_class += 1
            except:
                i += 1
                continue
            deep_pred = F.softmax(model(deep_features), dim=1)
            if (deep_pred[0,0] > deep_pred[0,1]) and sample=='bonafide':
                correct += 1
            if (deep_pred[0,0] < deep_pred[0,1]) and sample!='bonafide':
                correct += 1
            if (deep_pred[0,0] > deep_pred[0,1]):
                pred.append(0)
            else:
                pred.append(1)
            if sample=='bonafide':
                labels.append(0)
            else:
                labels.append(1)
            pbar.update(1)
            if sample == 'bonafide' and (i+1 == human_samples or for_class == human_samples):
                break
            elif sample != 'bonafide' and (i+1 == fake_samples or for_class == int(fake_samples/(len(rand_samples.keys())-1))):
                break
        pbar.close()
    print(classification_report(labels, pred))
    print(gen_eer(labels, pred))

def gen_predictions(label, hand_model, deep_model, sample_file=None):
    if sample_file == None:
        sample = gen_sample(label)
        file_path = os.path.join(DATA_FOLDER, 'ASVspoof2019_LA_eval', 'flac', '{0}.flac'.format(sample))
        print(file_path)
    else:
        print('hello')
        file_path = sample_file
    hand_features = process_data_handcrafted(file_path)
    audio_file = AudioSegment.from_file(file_path, format="flac")
    play(audio_file)
    deep_features = process_data_deep(file_path)
    X = np.load('./X.npy')
    X = np.concatenate([X, hand_features.reshape(1,-1)])
    hand_features = StandardScaler().fit_transform(X)
    hand_features = hand_features[-1,:].reshape(1,-1)
    hand_pred = hand_model.predict(hand_features)
    deep_pred = F.softmax(deep_model(deep_features), dim=1)
    if hand_pred[0] == 0:
        print("Handcrafted Model predicted BONAFIDE")
    elif hand_pred[0] == 1:
        print("Handcrafted Model predicted SPOOFED")
    if deep_pred[0,0] > deep_pred[0,1]:
        print("Deep Model predicted BONAFIDE")
    elif deep_pred[0,0] < deep_pred[0,1]:
        print("Deep Model predicted SPOOFED")

if __name__ == '__main__':

    hand_model = load_handcrafted_model()
    deep_model = load_deep_model()
    
    # gen_deep_pred(deep_model)

    # for filename in os.listdir('Sample Audio'):
    #     file = os.path.join('Sample Audio', filename)
    #     if 'original' in file:
    #         print('This sample is bonafide')
    #     elif 'synthetic' in file:
    #         print('This sample is spoofed')
    #     gen_predictions(None, hand_model, deep_model, file)

    # bonafide = ['5059308']
    # i = 0
    # while i < 2:
    #     try:
    #         print("For bonafide sample " + str(i))
    #         gen_predictions('bonafide', hand_model, deep_model)
    #         i += 1
    #     except:
    #         continue
    spoofed = ['8567330', '6917279']
    loc = '/Users/armaanlalani/Documents/Engineering Science Year 5/ESC499 - Thesis/Sample.flac'
    # i = 0
    # while i < 2:
    gen_predictions('spoofed', hand_model, deep_model, loc)
    # i += 1
