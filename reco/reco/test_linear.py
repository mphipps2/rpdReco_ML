import sys
sys.path.append('C://Users//Fre Shava Cado//Documents//VSCode Projects//rpdreco//')

#import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
#import reco.lib.vis as vis
import reco.lib.io as io

import tensorflow.keras as keras
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd

##from matplotlib.colors import (normalization)

def test_linear():
    model_loss = 'mse'
    file_num = 11
    filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{file_num}_{model_loss}"
    bins = 50

    #loads datasets
    test_X = pd.read_pickle(filepath + '//test_X.pickle')
    Q_avg = pd.read_pickle(filepath + '//test_y.pickle')

    model = keras.models.load_model(filepath+f'//linear_{file_num}_{model_loss}.h5',compile = False)
    Q_predicted = model.predict([test_X.astype('float')])

    psi_rec = np.arctan2(Q_predicted[:,1],Q_predicted[:,0])
    psi_gen = np.arctan2(Q_avg.iloc[:,1],Q_avg.iloc[:,0])
    psi_comp = psi_gen.subtract(psi_rec)

    #collects summary stats for difference in angle
    mean = np.round(psi_comp.mean(), 3)
    sigma = np.round(psi_comp.std(), 3)
    error = np.round(psi_comp.sem(), 3)

    #plots difference in angle
    plt.figure(2)
    plt.hist(psi_comp, bins = bins, density = True)
    gaussian = np.linspace(psi_comp.min(), psi_comp.max(), 100)
    plt.plot(gaussian, norm.pdf(gaussian, mean, sigma))
    plt.title(r'$\Psi_{True}-\Psi_{recon}$')
    plt.xlabel('Angle Difference (radians)')
    plt.ylabel('Density Function')
    plt.text(-4,0.2,f'$\\mu={mean}\\pm {error}$,\n $\\sigma={sigma}$')
    plt.savefig(filepath + f'//model{file_num}_anglediff.png')
    plt.show()

