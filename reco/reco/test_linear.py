import sys

import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
import reco.lib.vis as vis
import reco.lib.io as io

import tensorflow.keras as keras
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd

##from matplotlib.colors import (normalization)

def test_linear():
    file_num = 1
    filepath = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles"
    bins = 50

    #loads datasets
    test_X = pd.read_pickle(filepath + '//test_X.pickle')
    Q_avg = pd.read_pickle(filepath + '//test_y.pickle')

    model = keras.models.load_model(filepath+f'//linear_{file_num}_mse.h5',compile = False)
    Q_predicted = model.predict([test_X.astype('float')])

    psi_rec = np.arctan2(Q_predicted.iloc[:,1],Q_predicted.iloc[:,0])
    psi_gen = np.arctan2(Q_avg.iloc[:,1],Q_avg.iloc[:,0])
    psi_comp = psi_gen.subtract(psi_rec)

    #collects summary stats for difference in angle
    mean = psi_comp.mean()
    sigma = psi_comp.std()
    error = psi_comp.sem()

    #plots difference in angle
    plt.figure(1)
    plt.hist(psi_comp, - bins, density = True)
    gaussian = np.linspace(psi_comp.min(), psi_comp.max(), 100)
    plt.plot(gaussian, norm.pdf(gaussian, mean, sigma))
    plt.title(r'$\Psi_{True}-\Psi_{recon}$')
    plt.xlabel('Angle Difference (radians)')
    plt.ylabel('Density Function')
    plt.text(-1.5,0.5,f'$\\mu={mean}\\pm {error}$,\n \\sigma={sigma}')
    plt.savefig(filepath + f'//model{file_num}_anglediff.png')

