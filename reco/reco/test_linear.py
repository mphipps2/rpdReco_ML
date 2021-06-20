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
	file_num = 19
	filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{file_num}_{model_loss}"
	bins = 100

    #loads datasets
	test_A = pd.read_pickle(filepath + '//test_A.pickle')
	#set test_x based on model: 8:24 for allchan, 24:32 for avg, 2:4 for Px,Py
	test_X = test_A.iloc[:,8:24]
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
    #plt.title(r'$\Psi_{Gen}-\Psi_{recon}$', fontsize = 12)
	plt.xlabel(r'$\Psi_0^{\rm Gen-A}-\Psi_0^{\rm Rec-A}$ [rad]', fontsize = 12)
	plt.ylabel('Density Function', fontsize = 12)
	#plt.ylim(top = 0.21)
	plt.text(-3,0.18,f'$\\mu={np.round(mean_gen, 3)}\\pm {np.round(error_gen,3)}$,\n $\\sigma={np.round(sigma_gen, 3)}$')
	plt.savefig(filepath + f'//model{file_num}_gen_anglediff.png')

	plt.figure(3)
	plt.hist(psi_truth_rec, bins = bins, density = True)
	plt.plot(truthXspace, fit_function(truthXspace, *truthPopt))
	#plt.title(r'$\Psi_{True}-\Psi_{recon}$', fontsize = 12)
	plt.xlabel(r'$\Psi_0^{\rm Truth-A}-\Psi_0^{\rm Rec-A}$ [rad]', fontsize = 12)
	plt.ylabel('Density Function', fontsize = 12)
	#plt.ylim(top =0.21)
	plt.text(-3,0.18,f'$\\mu={np.round(mean_truth, 3)}\\pm {np.round(error_truth,3)}$,\n $\\sigma={np.round(sigma_truth, 3)}$')
	plt.savefig(filepath + f'//model{file_num}_truth_anglediff.png') 

	df = pd.DataFrame()
	df['numParticles'] = numParticles
	df['pt_nuclear'] = pt_nuc.multiply(1000)
	df['sigma_gen'] = psi_gen_rec
	df['sigma_truth'] = psi_truth_rec

    #stratifies based on pt_nuclear
	df['ptBins'] = pd.cut(x = df.iloc[:,1], bins = [5, 15, 25, 35, 45, np.inf], labels = ['pt1', 'pt2', 'pt3', 'pt4', 'pt5'], right = False, include_lowest = True)
	#creates bins for neutron clusters. For future reference, do not do this in python
	groupLabels = [20, 25, 30, 35]
	df['nbins'] = pd.cut(x = df.iloc[:,0], bins =[20,25,30,35,40], labels = groupLabels, right = False, include_lowest = True)

	std = df.groupby(['ptBins','nbins']).std()
	sem = df.groupby(['ptBins','nbins']).sem()

	plt.figure(4)
	ax = plt.figure(4).gca()
	ax.xaxis.set_major_locator(MaxNLocator(nbins = 4, integer=True))
	plt.plot(groupLabels, std.loc['pt1'].sigma_gen, 'gs', label = r'$5\leq p_T^{nuclear}<15$ MeV')
	plt.errorbar(groupLabels, std.loc['pt1'].sigma_gen, yerr=sem.loc['pt1'].sigma_gen, fmt = 'gs')
	plt.plot(groupLabels, std.loc['pt2'].sigma_gen, 'bs', label = r'$15\leq p_T^{nuclear}<25$ MeV')
	plt.errorbar(groupLabels, std.loc['pt2'].sigma_gen, yerr=sem.loc['pt2'].sigma_gen, fmt = 'bs')
	plt.plot(groupLabels, std.loc['pt3'].sigma_gen, 'rs', label = r'$25\leq p_T^{nuclear}<35$ MeV')
	plt.errorbar(groupLabels, std.loc['pt3'].sigma_gen, yerr=sem.loc['pt3'].sigma_gen, fmt = 'rs')
	plt.plot(groupLabels, std.loc['pt4'].sigma_gen, 'ms', label = r'$35\leq p_T^{nuclear}<45$ MeV')
	plt.errorbar(groupLabels, std.loc['pt4'].sigma_gen, yerr=sem.loc['pt4'].sigma_gen, fmt = 'ms')
	plt.plot(groupLabels, std.loc['pt5'].sigma_gen, 'ks', label = r'$45\leq p_T^{nuclear}$ MeV')
	plt.errorbar(groupLabels, std.loc['pt5'].sigma_gen, yerr=sem.loc['pt5'].sigma_gen, fmt = 'ks')
	plt.grid(axis = 'x')
	#plt.title('Gen Resolution', fontsize = 12)
	plt.xlabel(r'$\rm N_{neutrons}$', fontsize = 12)
	plt.ylabel(r'$\sigma_{\rm \Psi^{\rm Gen-A}_0-\Psi^{Rec-A}_0}$ [rad]', fontsize = 12)
	plt.ylim(bottom = 0, top = 2 * std.loc['pt1'].sigma_gen.max())
	plt.legend()
	plt.savefig(filepath +  f'//model{file_num}_gen_stratsigmas.png')
    
	plt.figure(5)
	ax = plt.figure(5).gca()
	ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, integer=True))
	plt.plot(groupLabels, std.loc['pt1'].sigma_truth, 'gs', label = r'$5\leq p_T^{nuclear}<15$ MeV')
	plt.errorbar(groupLabels, std.loc['pt1'].sigma_truth, yerr=sem.loc['pt1'].sigma_truth, fmt = 'gs')
	plt.plot(groupLabels, std.loc['pt2'].sigma_truth, 'bs', label = r'$15\leq p_T^{nuclear}<25$ MeV')
	plt.errorbar(groupLabels, std.loc['pt2'].sigma_truth, yerr=sem.loc['pt2'].sigma_truth, fmt = 'bs')
	plt.plot(groupLabels, std.loc['pt3'].sigma_truth, 'rs', label = r'$25\leq p_T^{nuclear}<35$ MeV')
	plt.errorbar(groupLabels, std.loc['pt3'].sigma_truth, yerr=sem.loc['pt3'].sigma_truth, fmt = 'rs')
	plt.plot(groupLabels, std.loc['pt4'].sigma_truth, 'ms', label = r'$35\leq p_T^{nuclear}<45$ MeV')
	plt.errorbar(groupLabels, std.loc['pt4'].sigma_truth, yerr=sem.loc['pt4'].sigma_truth, fmt = 'ms')
	plt.plot(groupLabels, std.loc['pt5'].sigma_truth, 'ks', label = r'$45\leq p_T^{nuclear}$ MeV')
	plt.errorbar(groupLabels, std.loc['pt5'].sigma_truth, yerr=sem.loc['pt5'].sigma_truth, fmt = 'ks')
	plt.grid(axis = 'x')
	#plt.title('Truth Resolution')
	plt.xlabel(r'$\rm N_{\rm neutrons}$', fontsize = 12)
	plt.ylabel(r'$\sigma_{\rm \Psi^{\rm Truth-A}_0-\Psi^{Rec-A}_0}$ [rad]')
	plt.ylim(bottom = 0, top = 2*std.loc['pt1'].sigma_truth.max())
	plt.legend()
	plt.savefig(filepath + f'//model{file_num}_truth_stratsigmas.png')
    
	plt.show()