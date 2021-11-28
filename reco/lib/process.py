import tensorflow.keras as keras
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import pad
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import os
import sys

plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"savefig.bbox": 'tight'})
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.models import Model
import tensorflow
import time
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from tensorflow.python.client import device_lib
from ROOT import *
import math


#import numpy as np
#import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import Normalization

def blur_neutron(n_hit):
        blur = []        
        for i in range(n_hit.shape[0]):
                blur.append(np.random.normal(n_hit[i], np.sqrt(n_hit[i] * (0.171702**2))))
        return np.array(blur)


def subtract_signals_old(data):
        for x in range(15,3,-1):
                subtr_chan = x - 4
                data[f'channel_{x}'] = data[f'channel_{x}'] - data[f'channel_{subtr_chan}']
        return data


def subtract_signals(data):
        for row in range(3,0,-1):
                for col in range(0,4,1):
                        subtr_row = row - 1
                        data[f'rpd{row}_{col}_Charge'] = data[f'rpd{row}_{col}_Charge'] - data[f'rpd{subtr_row}_{col}_Charge']
        return data

def subtract_signals_peak(data):
        for row in range(3,0,-1):
                for col in range(0,4,1):
                        subtr_row = row - 1
                        data[f'rpd{row}_{col}_Peak_max'] = data[f'rpd{row}_{col}_Peak_max'] - data[f'rpd{subtr_row}_{col}_Peak_max']
        return data

def GetPsiResidual(psi_true, psi_rec):
        psi_truth_res = psi_true.subtract(psi_rec)
        psi_truth_res[psi_truth_res > np.pi] = -2*np.pi + psi_truth_res
        psi_truth_res[psi_truth_res < -np.pi] = 2*np.pi + psi_truth_res
        return psi_truth_res

def GetPsiResidual_np(psi_true, psi_rec):
        psi_truth_res = psi_true - psi_rec
        print("psi_truth_res shape: ", psi_truth_res.shape)
#        psi_truth_res[psi_truth_res > np.pi] = -2*np.pi + psi_truth_res[psi_truth_res > np.pi]
        psi_truth_res[psi_truth_res > np.pi] += -2*np.pi 
        psi_truth_res[psi_truth_res < -np.pi] += 2*np.pi 
        return psi_truth_res

def GetPosResidual_np(Q_true, Q_rec):
        pos_res = Q_true - Q_rec

        return pos_res

def GetPosResidual_mag_np(Q_true_x, Q_true_y, Q_rec_x, Q_rec_y):
#        mag = pow(Q_true_x - Q_rec_x,2) + pow(Q_true_y - Q_rec_y,2)
        mag = (Q_true_x - Q_rec_x) + (Q_true_y - Q_rec_y)
        print ("mag type " , mag.dtype)
#        pos_res = np.sqrt(mag)

        return mag


def reshape_signal(ary, normalization = False, flatten = False, padding = 1):

#        print('ary before: ', ary)
        if normalization:
#                print('signal before norm: ',ary)
                ary -= ary.mean(axis=(0,1))
                ary /= ary.std(axis=(0,1))
#                print('signal after norm: ',ary)
                ary = np.array([i.reshape(4,4,1) for i in ary])
        else:
                ary = np.array([i.reshape(4,4,1) for i in ary])
        if flatten:
                ary = np.array([i.reshape(16) for i in ary])
        if padding:
                ary = np.pad(ary[:, :, :, :], ((0, 0), (padding, padding), (padding, padding), (0,0)), 'constant')
#        print('ary after: ', ary)
        return ary


def findCOM(rpd):
        com = pd.DataFrame(0, index=np.arange(len(rpd)), columns = ['comX', 'comY'])
        totalSignal = rpd.sum(axis=1)

        for ch in range(len(rpd.columns)):
                x = 0
                y = 0
                if ch < 4: y = 1.5
                elif (ch >= 4 and ch < 8): y = 0.5
                elif (ch >= 8 and ch < 12): y = -0.5
                else: y = -1.5

                if ch%4 == 0: x = 1.5
                elif ch%4 == 1: x = 0.5
                elif ch%4 == 2: x = -0.5
                elif ch%4 == 3: x = -1.5

                com.comX += x*rpd.iloc[:,ch]
                com.comY += y*rpd.iloc[:,ch]
        com.comX/=totalSignal
        com.comY/=totalSignal
        return com

def getCOMReactionPlane(com, centerX, centerY):
	com[:,1] = com[:,1]-centerY
	com[:,0] = com[:,0]-centerX
	phi = np.arctan2(com[:,1], com[:,0])
	return phi


def findCOM_np (rpd):
        print("rpd, ",rpd)
        print("rpd_shape ", np.shape(rpd))
        total_signal = np.sum(rpd,axis=1)
        print("total signal ", total_signal)
        com = np.zeros((len(rpd),2))
        for ch in range(np.size(rpd,1)):
                x = 0
                y = 0
                if ch < 4: y = 1.5
                elif (ch >= 4 and ch < 8): y = 0.5
                elif (ch >= 8 and ch < 12): y = -0.5
                else: y = -1.5

                if ch%4 == 0: x = 1.5
                elif ch%4 == 1: x = 0.5
                elif ch%4 == 2: x = -0.5
                elif ch%4 == 3: x = -1.5

                com[:,0] += x*rpd[:,ch]
                com[:,1] += y*rpd[:,ch]
        np.seterr(divide='ignore', invalid='ignore')
        
        com[:,0] /= total_signal
        com[:,1] /= total_signal
        print("com0 ", com[:,0], " com1 " , com[:,1], " total signal " , total_signal)
        return com

def getCOMReactionPlane_np(com, centerX, centerY):
        com_y = com[:,1]-centerY
        com_x = com[:,0]-centerX
        phi = np.arctan2(com_y, com_x)
        print("com_y " , com_y , " com_x " , com_x , " phi " , phi)
        return phi

def get_normalizer(data):
    normalizer = Normalization(axis = None)
    normalizer.adapt(data)
    return normalizer


def get_averages(data):
    print(data.head(5))
    for row in range(4):
        temp = pd.Series(0, index=np.arange(len(data)))
        for column in range(4):
            temp = temp.add(data.iloc[:, 8 + 4*row + column])
        temp = temp.divide(4)
        data[f'rowAvg_{row}']=temp
    for column in range(4):
        temp = pd.Series(0, index = np.arange(len(data)))
        for row in range(4):
            temp = temp.add(data.iloc[:, 8 + 4*row + column])
        temp = temp.divide(4)
        data[f'colAvg_{column}']=temp
    print(data.head(5))
    return data
