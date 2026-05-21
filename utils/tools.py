import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import find_peaks
from utils.pate_metric import PATE

plt.switch_backend('agg')

class ProgressBar():
    def __init__(self, iteration, bin = 100):
        self._iteration = iteration
        self.postfix = ""
        self.bin = bin

    def __iter__(self):
        self.len_pre_print = 0
        for i, item in enumerate(self._iteration):
            self.i = i
            self.show()
            yield item
        print()
    
    def show(self):
        percentage = self.i/len(self._iteration)
        progress = "\r["+"="*round(percentage*self.bin) + "-"*(self.bin-round(percentage*self.bin))+f"] {round(percentage*100)}% "+ self.postfix 
        len_current_print = len(progress)
        end_blank = (self.len_pre_print-len_current_print) if len_current_print<self.len_pre_print else 0
        print(progress + " "*end_blank, end="")
        self.len_pre_print =  len_current_print
        
    def set_postfix(self, postfix={}):
        postfix_ls = []
        for key, value in postfix.items():
            postfix_ls.append(f" {key} : {value}")
        if len(postfix_ls)>0:
            self.postfix = "[" + ",".join(postfix_ls) + "]"
            self.show()

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.early_stopping_score_max = np.inf
        self.delta = delta

    def __call__(self, early_stopping_score, model, path):
        score = early_stopping_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(early_stopping_score, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(early_stopping_score, model, path)
            self.counter = 0

    def save_checkpoint(self, early_stopping_score, model, path):
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Best f1 in validation increased ({self.early_stopping_score_max:.6f} --> {early_stopping_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.early_stopping_score_max = early_stopping_score


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def get_periodic_lags(matrix, win_size, min_lag=None):
    T, C = matrix.shape
    if min_lag is None:
        min_lag = win_size
        
    # 1. Chuẩn hóa từng channel
    norm_matrix = (matrix - np.mean(matrix, axis=0)) / (np.std(matrix, axis=0) + 1e-8)
    
    # 2. Tính toán n_fft để chạy FFT nhanh hơn
    n_fft = next_fast_len(2 * T - 1)
    
    # --- SỬA TẠI ĐÂY ---
    # Khởi tạo total_acf có độ dài T để khớp với acf[:T]
    total_acf = np.zeros(T) 
    
    for c in range(C):
        channel = norm_matrix[:, c]
        X = fft(channel, n=n_fft)
        S = X * np.conj(X)
        acf_full = ifft(S).real
        
        # Chỉ lấy phần lag từ 0 đến T-1 (khử nhiễu biên/đối xứng)
        acf = acf_full[:T]
        
        # Cộng dồn (Lúc này cả 2 mảng đều có độ dài T)
        total_acf += acf / (acf[0] + 1e-8)
    
    total_acf /= C
    total_acf[0] = 0 # Triệt tiêu lag 0
    
    # 3. Peak Detection
    limit = T - win_size
    scores = total_acf[:limit]
    
    peaks, props = find_peaks(
        scores,
        distance=min_lag,
        prominence=0.01
    )
    
    valid_mask = peaks >= min_lag
    final_lags = peaks[valid_mask]
    final_scores = props['prominences'][valid_mask]
    
    return final_lags, final_scores

def aggregate(df, overlap_aggregate='max'):
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(by='date', sort=True)[['score', 'label']].aggregate({'score': overlap_aggregate, 'label': 'max'})
    # score = df['score'].values
    # label = df['label'].values
    return df
