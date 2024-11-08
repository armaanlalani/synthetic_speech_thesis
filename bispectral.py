from math import *
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.linalg import toeplitz
from scipy.signal import correlate
import matplotlib.pyplot as plt

import os
import pickle
import pandas as pd

from stingray import lightcurve
from stingray.bispectrum import Bispectrum

import soundfile as sf

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, roc_curve

DATA_FOLDER = "/Volumes/ARMAAN USB/LA"

def sliding_window(a, window, step_size):
    # compute number of chunks
    num_chunks = ((len(a)-window)//step_size)+1
 
    # yield chunks
    for i in range(0,num_chunks*step_size,step_size):
        yield a[i:i+window]

def get_prediction_coefficients(signal, L):
    N = len(signal)
    r = np.correlate(signal, signal, mode='full')
    r = r[N-1:2*N-L-1][::-1]
    R = np.zeros((L, L))
    for i in range(L):
        R[i,:] = r[L-1-i:L-1-i+L]
    a = np.linalg.solve(R, r[:L])
    return a

def get_short_term_prediction_error(signal, L, a):
    N = len(signal)
    e = np.zeros(N)
    for n in range(L, N):
        e[n] = signal[n] - np.dot(a, signal[n-L:n])
    return e

def get_long_term_prediction_error(e, k, beta_k):
    N = len(e)
    q = np.zeros(N)
    for n in range(k, N):
        q[n] = e[n] - beta_k * e[n-k]
    return q

def get_feature_vectors(signal, e, q):
    N = len(signal)
    E_ST = np.sum(e**2) / N
    G_ST = np.sum(signal**2) / (np.sum(e**2) + 1e-100)
    E_LT = np.sum(q**2) / N
    G_LT = np.sum(e**2) / (np.sum(q**2) + 1e-100)
    return np.array([E_ST, G_ST, E_LT, G_LT])

def estimate_long_term_parameters(e, kmin, kmax):
    N = len(e)
    k = None
    beta = None
    min_JLT = float('inf')
    for i in range(N):
        beta_i = e[i] / (e[0] + 1e-100)
        JLT = 0
        for n in range(N-i):
            q = e[n] - beta_i * e[n+i]
            JLT += q**2
        JLT /= N
        if JLT < min_JLT:
            min_JLT = JLT
            k = i
            beta = beta_i
    return k, beta

def speech_features(signal, L, kmin, kmax):
    a = get_prediction_coefficients(signal, L)
    e = get_short_term_prediction_error(signal, L, a)
    k, beta = estimate_long_term_parameters(e, kmin, kmax)
    q = get_long_term_prediction_error(e, k, beta)
    return get_feature_vectors(signal, e, q)

def windowed_bicoherence_features(signal, samplerate, window=32, step_size=16, L=10, kmin=0.004, kmax=0.0125):
    ## accumulator dict
    bs_accum = {
        "bispec_full_sum":0,
        "bispec":0,
        "bispec_mag_sum":0,
        "bispec_phase":0
    }
    ## calculate number of chunks to iter over
    chunks = ((len(signal)-window)//step_size)+1


    ## Iterate over chunks
    for chunk in sliding_window(signal, window, step_size):
        stlt = speech_features(chunk, L, kmin, kmax)
        try:
            stlt_accum = np.vstack((stlt_accum, stlt))
        except:
            stlt_accum = stlt

        ## compute bispectrum
        lc = lightcurve.Lightcurve(np.arange(len(chunk)), chunk, dt=samplerate, skip_checks=True)
        bs = Bispectrum(lc)

        ## Aggregate useful attributes
        bs_accum["bispec_full_sum"] += bs.bispec
        bs_accum["bispec_mag_sum"] += bs.bispec_mag
        bs_accum["bispec_phase"] += bs.bispec_phase
    ## Normalize attributes
    bs_accum["bispec_normal"] = np.abs( bs_accum["bispec_full_sum"] ) /\
                                        bs_accum["bispec_mag_sum"]
    bs_accum["bispec_mag"] = np.abs(bs_accum["bispec_normal"])
    bs_accum["bispec_phase"] /= chunks
    bs_accum["bispec_phase"] = np.interp(bs_accum["bispec_phase"], 
                                         (bs_accum["bispec_phase"].min(), bs_accum["bispec_phase"].max()), 
                                         (0, 1))
    
    f = get_stats(stlt_accum)

    ## Output features
    return (bs_accum["bispec_mag"].mean(), 
            bs_accum["bispec_phase"].mean(),
            bs_accum["bispec_mag"].var(),
            bs_accum["bispec_phase"].var(),
            skew(bs_accum["bispec_mag"].ravel(), axis=0, bias=True),
            skew(bs_accum["bispec_phase"].ravel(), axis=0, bias=True),
            kurtosis(bs_accum["bispec_mag"].ravel(), axis=0, bias=True),
            kurtosis(bs_accum["bispec_phase"].ravel(), axis=0, bias=True),), f

def get_stats(stlt):
    mean_est, mean_elt, mean_gst, mean_glt = np.mean(stlt[:,0]), np.mean(stlt[:,1]), np.mean(stlt[:,2]), np.mean(stlt[:,3])
    std_est, std_elt, std_gst, std_glt = np.std(stlt[:,0]), np.std(stlt[:,1]), np.std(stlt[:,2]), np.std(stlt[:,3])
    max_est, max_elt, max_gst, max_glt = np.max(stlt[:,0]), np.max(stlt[:,1]), np.max(stlt[:,2]), np.max(stlt[:,3])
    min_est, min_elt, min_gst, min_glt = np.min(stlt[:,0]), np.min(stlt[:,1]), np.min(stlt[:,2]), np.min(stlt[:,3])
    return [mean_est, mean_elt, mean_gst, mean_glt, std_est, std_elt, std_gst, std_glt, max_est, max_elt, max_gst, max_glt, min_est, min_elt, min_gst, min_glt]

def plot_vals(feature, results):
    fig, ax = plt.subplots()
    if feature == 'Mean':
        mean, phase = 0, 1
    elif feature == 'Variance':
        mean, phase = 2, 3
    elif feature == 'Skew':
        mean, phase = 4, 5
    elif feature == 'Kurtosis':
        mean, phase = 6, 7
    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:purple', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:gray']
    for i, cat in enumerate(results.keys()):
        ax.scatter(results[cat][:,mean], results[cat][:,phase], c=colors[i], label=cat)
    ax.legend()
    ax.set_xlabel('Magnitude ' + feature)
    ax.set_ylabel('Phase ' + feature)
    ax.set_title(feature + ' Bicoherence Magnitude and Phase')
    ax.grid(True)
    plt.show()

def get_labels(folder):
    if folder == 'train':
        label_path = os.path.join(DATA_FOLDER, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.{0}.trn.txt'.format(folder))
    else:
        label_path = os.path.join(DATA_FOLDER, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.{0}.trl.txt'.format(folder))
    labels = pd.read_csv(label_path, sep=' ', header=None)
    labels = labels.drop([2], axis=1)
    labels = labels.rename(columns={0:'Speaker', 1:'File', 3:'System', 4:'Label'})
    return labels

def get_random_samples(labels, human_samples, fake_samples, all=True):
    systems = labels['System'].unique()
    rand_samples = dict()
    for system in systems:
        if system == '-':
            samples = labels.loc[labels['System'] == system]
            if not all:
                samples = samples.sample(n=human_samples, replace=False)
            samples = samples.sample(frac=1, random_state=523)
            print('Number of samples: ' + str(len(samples)))
            rand_samples['bonafide'] = list(samples['File'])
        else:
            samples = labels.loc[labels['System'] == system]
            if not all:
                samples = samples.sample(n=int(fake_samples/(len(systems)-1)), replace=False)
            samples = samples.sample(frac=1, random_state=71)
            print('Number of samples: ' + str(len(samples)))
            rand_samples[system] = list(samples['File'])
    return rand_samples, human_samples, fake_samples

def gen_data(folder, human_samples, fake_samples, all=True, manual=False):
    labels = get_labels(folder)
    samples, human_samples, fake_samples = get_random_samples(labels, human_samples, fake_samples, all=all)
    results = dict()
    for sample in samples.keys():
        print('Generating samples for {0}'.format(sample))
        if sample == 'bonafide':
            samples_needed = human_samples
        else:
            samples_needed = int(fake_samples / int(len(samples.keys()) - 1))
        i = 0
        pbar = tqdm(total=samples_needed)
        for filename in samples[sample]:
            try:
                with open(os.path.join(DATA_FOLDER, 'ASVspoof2019_LA_{0}'.format(folder), 'flac', '{0}.flac'.format(filename)), 'rb') as f:
                    data, samplerate = sf.read(f)
            except:
                continue
            try:
                bicoherence, f = windowed_bicoherence_features(data, 60, window=64, step_size=32)
                bicoherence = np.array(bicoherence)
                bicoherence = np.concatenate([bicoherence, f])
                i += 1
                pbar.update(1)
            except:
                continue
            try:
                results[sample] = np.vstack((results[sample], bicoherence))
            except:
                results[sample] = bicoherence
            if i == samples_needed:
                break
        pbar.close()
    with open('train_sample_files.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results, human_samples, fake_samples

def svm(results, human_samples, fake_samples, save_model=False, model_name=None):
    X = np.concatenate([results['bonafide'], results['A01'], results['A02'], results['A03'], results['A04'], results['A05'], results['A06']])
    X = np.where(X==np.inf, 1e99, X)
    print(X.shape)
    X_norm = StandardScaler().fit_transform(X)
    Y = np.concatenate([np.ones(human_samples)*0, np.ones(fake_samples)*1])
    rbf_svc = SVC(gamma='scale', class_weight='balanced')
    X_norm[np.isnan(X_norm)] = 0
    rbf_svc.fit(X_norm, Y)
    print(classification_report(Y, rbf_svc.predict(X_norm)))
    if save_model:
        pickle.dump(rbf_svc, open(model_name, 'wb'))
    return rbf_svc

def mlp(results, human_samples, fake_samples, save_model=False, model_name=None):
    X = np.concatenate([results['bonafide'], results['A01'], results['A02'], results['A03'], results['A04'], results['A05'], results['A06']])
    X_norm = StandardScaler().fit_transform(X)
    Y = np.concatenate([np.ones(human_samples)*0, np.ones(fake_samples)*1])
    mlp = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(4,), max_iter=150)
    mlp.fit(X, Y)
    print(classification_report(Y, mlp.predict(X_norm)))
    if save_model:
        pickle.dump(mlp, open(model_name, 'wb'))
    return mlp

def train(human_samples, fake_samples, all=False, plot=False, save_model=False, model_name=None, model_type=mlp, use_data=False, use_data_filename=None, manual=False):
    if use_data:
        with open(use_data_filename, 'rb') as handle:
            train_data = pickle.load(handle)
    else:
        train_data, human_samples, fake_samples = gen_data('train', human_samples, fake_samples, all=all, manual=manual)
    if plot:
        plot_vals('Mean', train_data)
        plot_vals('Variance', train_data)
        plot_vals('Skew', train_data)
        plot_vals('Kurtosis', train_data)
    model = model_type(train_data, human_samples, fake_samples, save_model=save_model, model_name=model_name)
    return model

def gen_data_eval(human_samples, fake_samples):
    labels = get_labels('eval')
    systems = labels['System'].unique()
    rand_samples = dict()
    for system in systems:
        print(system)
        if system == '-':
            samples = labels.loc[labels['System'] == system]
            samples = samples.sample(frac=1, random_state=523)
            rand_samples['bonafide'] = list(samples['File'])
        else:
            samples = labels.loc[labels['System'] == system]
            samples = samples.sample(frac=1, random_state=523)
            rand_samples[system] = list(samples['File'])
    print(samples)
    results = dict()
    for sample in rand_samples.keys():
        print('Generating samples for {0}'.format(sample))
        if sample == 'bonafide':
            pbar = tqdm(total=human_samples)
        else:
            pbar = tqdm(total=int(fake_samples/(len(rand_samples.keys())-1)))
        i = 0
        while True:
            try:
                with open(os.path.join(DATA_FOLDER, 'ASVspoof2019_LA_{0}'.format('eval'), 'flac', '{0}.flac'.format(rand_samples[sample][i])), 'rb') as f:
                    data, samplerate = sf.read(f)
            except:
                i += 1
                continue
            try:
                bicoherence, f = windowed_bicoherence_features(data, 60, window=64, step_size=32)
                bicoherence = np.array(bicoherence)
                bicoherence = np.concatenate([bicoherence, f])
                pbar.update(1)
            except:
                i += 1
                continue
            try:
                results[sample] = np.vstack((results[sample], bicoherence))
            except:
                results[sample] = bicoherence
            i += 1
            if sample == 'bonafide' and (results[sample].shape[0] == human_samples):
                break
            elif sample != 'bonafide' and (results[sample].shape[0] == int(fake_samples/(len(rand_samples.keys())-1))):
                break
        pbar.close()
    with open('eval_sample_files.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results, human_samples, fake_samples

def gen_eer(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1-tpr
    eer = fpr[np.nanargmin(np.absolute(fnr-fpr))]
    return eer

def eval(human_samples, fake_samples, all=False, model=None, load_model=False, model_name=None, use_data=False, use_data_filename=None):
    if load_model:
        model = pickle.load(open(model_name, 'rb'))
    if use_data:
        with open(use_data_filename, 'rb') as handle:
            results = pickle.load(handle)
    else:
        results, human_samples, fake_samples = gen_data_eval(human_samples, fake_samples)
    X = np.concatenate([results['bonafide'], results['A11']])
    for sample in results.keys():
        if sample == 'bonafide' or sample == 'A11':
            continue
        X = np.concatenate([X, results[sample]])
    X_norm = StandardScaler().fit_transform(X)
    Y = np.concatenate([np.ones(human_samples)*0, np.ones(fake_samples)*1])
    predictions = model.predict(X_norm)
    print(predictions)
    print(classification_report(Y, predictions))
    print(gen_eer(Y, predictions))

def gen_l_data(train_eval='train'):
    files = [f'{train_eval}_sample_files_L10.pickle', f'{train_eval}_sample_files_L15.pickle', f'{train_eval}_sample_files_L20.pickle']
    for filename in sorted(os.listdir('Bicoherence + STLT Files')):
        if filename in files:
            print(filename)
            with open(os.path.join('Bicoherence + STLT Files', filename), 'rb') as handle:
                results = pickle.load(handle)
            for key in results.keys():
                try:
                    if key == 'bonafide':
                        X = np.concatenate([results[key], X])
                    else:
                        X = np.concatenate([X, results[key]])
                except:
                    X = results[key]
            labels = np.concatenate([np.ones(results['bonafide'].shape[0])*0, np.ones(X.shape[0] - results['bonafide'].shape[0])*1])
            labels = labels.reshape(labels.shape[0],1)
            X = np.hstack((X, labels))
            X = pd.DataFrame(X)
            X[[0,1,2,3,4,5,6,7]] = X[[0,1,2,3,4,5,6,7]].round(4)
            try:
                data = pd.merge(data, X, on=[0,1,2,3,4,5,6,7,24])
            except:
                data = X
    labels = data[24]
    Y = pd.Series.to_numpy(labels)
    data = data.drop(24, axis=1)
    data = pd.DataFrame.to_numpy(data)
    if train_eval == 'eval':
        data = data[:1300,:]
        Y = Y[:1300]
    np.save('X.npy', data)
    X_norm = StandardScaler().fit_transform(data)
    if train_eval == 'train':
        rbf_svc = SVC(gamma='scale', class_weight='balanced')
        rbf_svc.fit(X_norm, Y)
        pickle.dump(rbf_svc, open('handcrafted.sav', 'wb'))
        print(classification_report(Y, rbf_svc.predict(X_norm)))
    elif train_eval == 'eval':
        model = pickle.load(open('handcrafted.sav', 'rb'))
        print(X_norm.shape)
        predictions = model.predict(X_norm)
        print(classification_report(Y, predictions))
        print(gen_eer(Y, predictions))
    return data

if __name__ == '__main__':
    gen_l_data('train')
    # fake_samples = 3000
    # human_samples = 2520
    # model = train(human_samples, fake_samples, all=True, plot=False, save_model=False, model_name='mlp_2580_2580.sav', model_type=svm, use_data=False, use_data_filename='Bicoherence + STLT Files/train_sample_files_new.pickle', manual=False)
    # eval(human_samples=650, fake_samples=900, all=False, model=svm, load_model=True, model_name='./handcrafted.sav', use_data=False, use_data_filename='Bicoherence + STLT Files/eval_sample_files.pickle')