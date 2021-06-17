import sys
sys.path.append('C://Users//Fre Shava Cado//Documents//VSCode Projects//rpdreco//')

#import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
#import reco.lib.vis as vis
import reco.lib.io as io

import tensorflow.keras as keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

#To be added:
#Error bars on residual plots, decrease number of datapoints

def fit_function(x, A, B, mu, sigma):
    return A+B*np.exp(-(x-mu)**2/(2*sigma**2))

def test_linear():
    model_loss = 'mse'
    file_num = 16
    filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{file_num}_{model_loss}"
    bins = 100

    #loads datasets
    test_A = pd.read_pickle(filepath + '//test_A.pickle')
    test_X = test_A.iloc[:,24:32]
    Q_avg = test_A.iloc[:,0:2]
    psi_truth = test_A.iloc[:,5]
    pt_nuc = test_A.iloc[:,4]
    numParticles = test_A.iloc[:,7]

    '''
    test_X = pd.read_pickle(filepath + '//test_X.pickle')
    Q_avg = pd.read_pickle(filepath + '//test_y.pickle')
   psi_truth = pd.read_pickle(filepath + '//psi_truth.pickle')
    '''
    model = keras.models.load_model(filepath+f'//linear_{file_num}_{model_loss}.h5',compile = False)
    Q_predicted = model.predict([test_X.astype('float')])

    psi_rec = np.arctan2(Q_predicted[:,1],Q_predicted[:,0])
    psi_gen = np.arctan2(Q_avg.iloc[:,1],Q_avg.iloc[:,0])

    psi_gen_rec = psi_gen.subtract(psi_rec)
    psi_gen_rec[psi_gen_rec > np.pi] = -2*np.pi + psi_gen_rec
    psi_gen_rec[psi_gen_rec < -np.pi] = 2*np.pi + psi_gen_rec

    psi_truth_rec = psi_truth.subtract(psi_rec)
    psi_truth_rec[psi_truth_rec > np.pi] = -2*np.pi + psi_truth_rec
    psi_truth_rec[psi_truth_rec < -np.pi] = 2*np.pi + psi_truth_rec

    #collects summary stats for difference in angle
    mean_gen, sigma_gen = norm.fit(psi_gen_rec)
    error_gen = psi_gen_rec.sem()

    mean_truth, sigma_truth = norm.fit(psi_truth_rec)
    error_truth = np.round(psi_truth_rec.sem(), 3)
    
    genBins = np.linspace(psi_gen_rec.min(), psi_gen_rec.max(), bins+1)
    genCenters = np.array([0.5*(genBins[i]+genBins[i+1]) for i in range(len(genBins)-1)])
    genEntries, bins1 = np.histogram(psi_gen_rec, bins = bins, density = True)
    genPopt,genPcov = curve_fit(fit_function, genCenters, genEntries, p0=[1, 1, genEntries.mean(), genEntries.std()])
    print(genPopt) 
    genXspace = np.linspace(mean_gen-sigma_gen, mean_gen+sigma_gen, 100000)  

    truthBins = np.linspace(psi_truth_rec.min(), psi_truth_rec.max(), bins+1)
    truthCenters = np.array([0.5*(truthBins[i]+truthBins[i+1]) for i in range (len(truthBins)-1)])
    truthEntries, bins2 = np.histogram(psi_truth_rec, bins = bins, density = True)
    truthPopt, truthPcov = curve_fit(fit_function, truthCenters, truthEntries, p0=[1, 1, truthEntries.mean(), truthEntries.std()])
    print(truthPopt)
    truthXspace = np.linspace(mean_truth - sigma_truth, mean_truth + sigma_truth, 100000)
    
    #plots difference in angle
    plt.figure(2)
    plt.hist(psi_gen_rec, bins = bins, density = True)
    plt.plot(genXspace, fit_function(genXspace,*genPopt))
    plt.ylim(bottom = 0)
    plt.title(r'$\Psi_{Gen}-\Psi_{recon}$')
    plt.xlabel('Angle Difference (radians)')
    plt.ylabel('Density Function')
    plt.text(-3,0.3,f'$\\mu={np.round(mean_gen, 3)}\\pm {np.round(error_gen,3)}$,\n $\\sigma={np.round(sigma_gen, 3)}$')
    #plt.savefig(filepath + f'//model{file_num}_gen_anglediff.png')
    
    plt.figure(3)
    plt.hist(psi_truth_rec, bins = bins, density = True)
    plt.plot(truthXspace, fit_function(truthXspace, *truthPopt))
    plt.title(r'$\Psi_{True}-\Psi_{recon}$')
    plt.xlabel('Angle Difference(radians)')
    plt.ylabel('Density Function')
    plt.text(-3,0.3,f'$\\mu={np.round(mean_truth, 3)}\\pm {np.round(error_truth,3)}$,\n $\\sigma={np.round(sigma_truth, 3)}$')
    #plt.savefig(filepath + f'//model{file_num}_truth_anglediff.png') 
    
    sig_data_gen = abs((psi_gen_rec-mean_gen)/sigma_gen)
    sig_data_truth = abs((psi_truth_rec - mean_truth)/sigma_truth)
    df = pd.DataFrame()
    df['numParticles'] = numParticles
    df['pt_nuclear'] = pt_nuc.multiply(1000)
    df['sigma_gen'] = sig_data_gen
    df['sigma_truth'] = sig_data_truth
    
    df1 = df[(df['pt_nuclear']>=5) & (df['pt_nuclear']<15)]
    df2 = df[(df['pt_nuclear']>=15) & (df['pt_nuclear']<25)]
    df3 = df[(df['pt_nuclear']>=25) & (df['pt_nuclear']<35)]
    df4 = df[(df['pt_nuclear']>=35) & (df['pt_nuclear']<45)]
    df5 = df[(df['pt_nuclear']>=45)]

    means1 = df1.groupby('numParticles').mean()
    serr1 = df1.groupby('numParticles').sem()
    means2 = df2.groupby('numParticles').mean()
    serr2 = df2.groupby('numParticles').sem()
    means3 = df3.groupby('numParticles').mean()
    serr3 = df3.groupby('numParticles').sem()
    means4 = df4.groupby('numParticles').mean()
    serr4 = df4.groupby('numParticles').sem()
    means5 = df5.groupby('numParticles').mean()
    serr5 = df5.groupby('numParticles').sem()

    plt.figure(4)
    ax = plt.figure(4).gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, integer=True))
    plt.plot(means1.index, means1.sigma_gen, 'go', label = r'$5\leq\rho_T^{nuclear}<15$ MeV', markevery = 5)
    plt.errorbar(means1.index, means1.sigma_gen, yerr=serr1.sigma_gen, fmt = 'go', errorevery = 5, markevery = 5)
    plt.plot(means2.index, means2.sigma_gen, 'bo', label = r'$15\leq\rho_T^{nuclear}<25$ MeV', markevery = 5,)
    plt.errorbar(means2.index, means2.sigma_gen, yerr=serr2.sigma_gen, fmt = 'bo', errorevery = 5, markevery = 5)
    plt.plot(means3.index, means3.sigma_gen, 'ro', label = r'$25\leq\rho_T^{nuclear}<35$ MeV', markevery = 5)
    plt.errorbar(means3.index, means3.sigma_gen, yerr=serr3.sigma_gen, fmt = 'ro', errorevery = 5, markevery = 5)
    plt.plot(means4.index, means4.sigma_gen, 'mo', label = r'$35\leq\rho_T^{nuclear}<45$ MeV', markevery = 5)
    plt.errorbar(means4.index, means4.sigma_gen, yerr=serr4.sigma_gen, fmt = 'mo', errorevery = 5, markevery = 5)
    plt.plot(means5.index, means5.sigma_gen, 'ko', label = r'$45\leq\rho_T^{nuclear}$ MeV', markevery = 5)
    plt.errorbar(means5.index, means5.sigma_gen, yerr=serr5.sigma_gen, fmt = 'ko', errorevery = 5, markevery = 5)
    plt.title('Gen Residuals')
    plt.xlabel(r'$N_{neutrons}$')
    plt.ylabel(r'$\sigma_{\Psi_{gen}-\Psi_{rec}}$')
    plt.ylim(bottom = 0, top = 1.8)
    plt.legend()
    #plt.savefig(filepath +  f'//model{file_num}_gen_stratsigmas.png')
    
    plt.figure(5)
    ax = plt.figure(5).gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, integer=True))
    plt.plot(means1.index, means1.sigma_truth, 'go', label = r'$5\leq\rho_T^{nuclear}<15$ MeV', markevery = 5)
    plt.errorbar(means1.index, means1.sigma_truth, yerr=serr1.sigma_truth, fmt = 'go', errorevery = 5, markevery = 5)
    plt.plot(means2.index, means2.sigma_truth, 'bo', label = r'$15\leq\rho_T^{nuclear}<25$ MeV', markevery = 5)
    plt.errorbar(means2.index, means2.sigma_truth, yerr=serr2.sigma_truth, fmt = 'bo', errorevery = 5, markevery = 5)
    plt.plot(means3.index, means3.sigma_truth, 'ro', label = r'$25\leq\rho_T^{nuclear}<35$ MeV', markevery = 5)
    plt.errorbar(means3.index, means3.sigma_truth, yerr=serr3.sigma_truth, fmt = 'ro', errorevery = 5, markevery = 5)
    plt.plot(means4.index, means4.sigma_truth, 'mo', label = r'$35\leq\rho_T^{nuclear}<45$ MeV', markevery = 5)
    plt.errorbar(means4.index, means4.sigma_truth, yerr=serr4.sigma_truth, fmt = 'mo', errorevery = 5, markevery = 5)
    plt.plot(means5.index, means5.sigma_truth, 'ko', label = r'$45\leq\rho_T^{nuclear}$ MeV', markevery = 5)
    plt.errorbar(means5.index, means5.sigma_truth, yerr=serr5.sigma_truth, fmt = 'ko', errorevery = 5, markevery = 5)
    plt.title('Truth Residuals')
    plt.xlabel(r'$N_{neutrons}$')
    plt.ylabel(r'$\sigma_{\Psi_{truth}-\Psi_{rec}}$')
    plt.ylim(bottom = 0, top = 1.8)
    plt.legend()
    #plt.savefig(filepath + f'//model{file_num}_truth_stratsigmas.png')
    
    plt.show()

