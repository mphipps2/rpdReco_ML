from operator import truth
import os
import sys
import math
sys.path.append('C://Users//Fre Shava Cado//Documents//VSCode Projects//rpdreco//')

import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
#import reco.lib.vis as vis
import reco.lib.io as io

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from random import randint

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.optimize import curve_fit

def get_model():
	model = keras.Sequential([
	#layers.BatchNormalization(input_shape = [2]),
	layers.Dense(units = 2, activation = 'linear', input_shape = [2])
	])
	return model

def fit_function(x, B, mu, sigma):
	return B*np.exp(-1*((x-mu)**2/(2*sigma**2)))
	
def fit_function2(x, A, B, mu, sigma):
	return A + B*np.exp(-1*((x-mu)**2/(2*sigma**2)))

def get_sigmas(df, ptLabels, groupLabels):
	bins = 50
	
	sem = pd.DataFrame(0, index = pd.MultiIndex.from_product([ptLabels,groupLabels], names = ['ptBins','nbins']), columns = ['sigma_gen','sigma_truth'])
	std = pd.DataFrame(0, index = pd.MultiIndex.from_product([ptLabels,groupLabels], names = ['ptBins','nbins']), columns = ['sigma_gen','sigma_truth'])
	for pt in ptLabels:
		for nCount in groupLabels:
			testdf = df.loc[(df['ptBins'] == pt) & (df['nbins'] == nCount)]

			mean_test_gen, sigma_test_gen = norm.fit(testdf.psi_gen)
			testBins_gen = np.linspace(mean_test_gen-sigma_test_gen, mean_test_gen+sigma_test_gen, bins+1)
			genEntries, bins_1 = np.histogram(testdf.psi_gen, bins = testBins_gen, density = False)
			genCenters = np.array([0.5*(testBins_gen[i]+testBins_gen[i+1]) for i in range (len(testBins_gen)-1)])
			genPopt, genPcov = curve_fit(fit_function, xdata = genCenters, ydata = genEntries, p0 = [75000, mean_test_gen, sigma_test_gen])

			mean_test_truth, sigma_test_truth = norm.fit(testdf.psi_truth)
			testBins_truth = np.linspace(mean_test_truth-sigma_test_truth, mean_test_truth+sigma_test_truth, bins+1)
			truthEntries, bins_2 = np.histogram(testdf.psi_truth, bins = testBins_truth, density = False)
			truthCenters = np.array([0.5*(testBins_truth[i]+testBins_truth[i+1]) for i in range (len(testBins_truth)-1)])
			truthPopt, truthPcov = curve_fit(fit_function, xdata = truthCenters, ydata = truthEntries, p0 = [75000, mean_test_truth, sigma_test_truth])
			std.loc[(pt,nCount)] += [np.abs(genPopt[-1]), np.abs(truthPopt[-1])]
			sem.loc[(pt,nCount)] += [np.sqrt(np.diag(genPcov))[-1], np.sqrt(np.diag(truthPcov))[-1]]

	return std, sem

def findCOM (rpd):
	#print(rpd)
	com = pd.DataFrame(0, index = rpd.index, columns = ['comX', 'comY'])
	#input()
	totalSignal = rpd.sum(axis = 1)
	#input()
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
	com.comY = com.comY-centerY
	com.comX = com.comX-centerX
	phi = np.arctan2(com.comY, com.comX)
	return phi

def plotHistograms(A, filepath, file_num, psi_gen, psi_truth, psi_rec):
	bins = 50
	pt_nuc = A.iloc[:,4]
	numParticles = A.iloc[:,7]

	psi_gen_rec = psi_gen.subtract(psi_rec)
	psi_gen_rec[psi_gen_rec > np.pi] = -2*np.pi + psi_gen_rec
	psi_gen_rec[psi_gen_rec < -np.pi] = 2*np.pi + psi_gen_rec

	psi_truth_rec = psi_truth.subtract(psi_rec)
	psi_truth_rec[psi_truth_rec > np.pi] = -2*np.pi + psi_truth_rec
	psi_truth_rec[psi_truth_rec < -np.pi] = 2*np.pi + psi_truth_rec
	print('Event	psi_truth   psi_gen   psi_reco   psi_truth-reco   psi_gen-reco')
	'''
	for i in range(10):
		print(f'Event {i}: {psi_truth.loc[i]}   {psi_gen.loc[i]}   {psi_rec.loc[i]}   {psi_truth_rec.loc[i]}   {psi_gen_rec.loc[i]}')
	'''
	df = pd.DataFrame()
	df['numParticles'] = numParticles
	df['pt_nuclear'] = pt_nuc.multiply(1000)
	df['psi_gen'] = psi_gen_rec
	df['psi_truth'] = psi_truth_rec
	
	#stratifies based on pt_nuclear
	ptLabels = ['pt1', 'pt2', 'pt3', 'pt4', 'pt5']
	df['ptBins'] = pd.cut(x = df.iloc[:,1], bins = [5, 15, 25, 35, 45, np.inf], labels = ptLabels, right = False, include_lowest = True)
	#creates bins for neutron clusters. For future reference, do not do this in python
	groupLabels = [22, 27, 32, 37]
	df['nbins'] = pd.cut(x = df.iloc[:,0], bins =[20,25,30,35,40], labels = groupLabels, right = False, include_lowest = True)

	measuredDf = df
	
	#collects summary stats for difference in angle
	mean_gen, sigma_gen = norm.fit(measuredDf.psi_gen)
	error_gen = measuredDf.psi_gen.sem()

	mean_truth, sigma_truth = norm.fit(measuredDf.psi_truth)
	error_truth = np.round(measuredDf.psi_truth.sem(), 3)

	genBins = np.linspace(mean_gen-sigma_gen, mean_gen+sigma_gen, bins + 1)
	genBins2 = np.linspace(measuredDf.psi_gen.min(), measuredDf.psi_gen.max(), bins + 1)
	genEntries, bins_1 = np.histogram(measuredDf.psi_gen, bins = genBins, density = False)
	genEntries2, bins_2 = np.histogram(measuredDf.psi_gen, bins = genBins2, density = False)
	genCenters = np.array([0.5*(genBins[i]+genBins[i+1]) for i in range(len(genBins)-1)])
	genCenters2 = np.array([0.5*(genBins2[i]+genBins2[i+1]) for i in range(len(genBins2)-1)])
	genPopt, genPcov = curve_fit(fit_function, xdata = genCenters, ydata = genEntries, p0 = [75000, mean_gen, sigma_gen])
	genPopt2, genPcov2 = curve_fit(fit_function2, xdata = genCenters2, ydata = genEntries2, p0 = [60000, 75000, mean_gen, sigma_gen])
	genXspace = np.linspace(mean_gen-sigma_gen, mean_gen+sigma_gen, 1000000)

	truthBins = np.linspace(mean_truth-sigma_truth, mean_truth+sigma_truth, bins + 1)
	truthBins2 = np.linspace(measuredDf.psi_truth.min(), measuredDf.psi_truth.max(), bins + 1)
	truthEntries, bins_3 = np.histogram(measuredDf.psi_truth, bins = truthBins, density = False)
	truthEntries2, bins_4 = np.histogram(measuredDf.psi_truth, bins = truthBins2, density = False)
	truthCenters = np.array([0.5*(truthBins[i]+truthBins[i+1]) for i in range(len(truthBins)-1)])
	truthCenters2 = np.array([0.5*(truthBins2[i]+truthBins2[i+1]) for i in range(len(truthBins2)-1)])
	truthPopt, truthPcov = curve_fit(fit_function, xdata = truthCenters, ydata = truthEntries, p0 = [75000, mean_truth, sigma_truth])
	truthPopt2, truthPcov2 = curve_fit(fit_function2, xdata = truthCenters2, ydata = truthEntries2, p0 = [60000, 75000, mean_truth, sigma_truth])
	truthXspace = np.linspace(mean_truth-sigma_truth, mean_truth+sigma_truth, 1000000)

	#plots difference in angle
	plt.figure(2)
	plt.hist(measuredDf.psi_gen, bins = bins, density = False)
	plt.plot(genXspace, fit_function2(genXspace, *genPopt2))
	plt.ylim(bottom = 0)
	#plt.title(r'$\Psi_{\rm Gen}-\Psi_{\rm Recon}$')
	plt.xlabel(r'$\Psi_0^{\rm Gen-A}-\Psi_0^{\rm Rec-A}$ [rad]', fontsize = 12)
	plt.ylabel('Density Function', fontsize = 12)
	plt.text(-3,40000,f"$\\mu={np.round(mean_gen, 3)}\\pm {np.round(error_gen,3)}$,\n $\\sigma={np.round(genPopt[-1],3)}$")
	#plt.savefig(filepath + f'//model{file_num}_gen_anglediff.png')

	plt.figure(3)
	plt.hist(measuredDf.psi_truth, bins = bins, density = False)
	plt.plot(truthXspace, fit_function2(truthXspace,*truthPopt2))
	#plt.title(r'$\Psi_0^{\rm True-A}-\Psi_{\rm Rec-A}$')
	plt.xlabel(r'$\Psi_0^{\rm True-A}-\Psi_0^{\rm Rec-A}$ [rad]', fontsize = 12)
	plt.ylabel('Density Function', fontsize = 12)
	plt.text(-3,30000,f"$\\mu={np.round(mean_truth, 3)}\\pm {np.round(error_truth,3)}$,\n $\\sigma={np.round(truthPopt[-1], 3)}$")
	#plt.savefig(filepath + f'//model{file_num}_truth_anglediff.png')

	print(df)

	std, sem = get_sigmas(df, ptLabels, groupLabels)
	print(sem)
	
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
	plt.grid()
	#plt.title('Gen Resolution', fontsize = 12)
	plt.xlabel(r'$\rm N_{neutrons}$', fontsize = 12)
	plt.ylabel(r'$\sigma_{\rm \Psi^{\rm Gen-A}_0-\Psi^{Rec-A}_0}$ [rad]', fontsize = 12)
	plt.xlim(left = 20, right = 40)
	plt.ylim(bottom = 0, top = 3)
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
	plt.grid()
	#plt.title('Truth Resolution')
	plt.xlabel(r'$\rm N_{\rm neutrons}$', fontsize = 12)
	plt.ylabel(r'$\sigma_{\rm \Psi^{\rm Truth-A}_0-\Psi^{Rec-A}_0}$ [rad]')
	plt.xlim(left = 20, right = 40)
	plt.ylim(bottom = 0, top = 6)
	plt.legend()
	plt.savefig(filepath + f'//model{file_num}_truth_stratsigmas.png')

	plt.show()

def comTester():
	centerX = 0
	centerY = -0.471659

	A = io.get_dataset(folder = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles", side = '//A')
	A = A.drop_duplicates()
	print(A)

	rpdSignals = A.iloc[:,8:24]
	testEvent = randint(0, len(rpdSignals))
	rpd = rpdSignals.iloc[testEvent:testEvent+1]
	com = findCOM(rpd)
	print(com)
	

def directCOMComparison():
	centerX = 0
	centerY = -0.471659
	model_num = 1
	model_loss = 'CoM'
	filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{model_num}_{model_loss}"

	A = io.get_dataset(folder = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles", side = '//A')
	A = A.drop_duplicates()

	rpdSignals = A.iloc[:,8:24]
	com = findCOM(rpdSignals)

	print(com)
	#input()
	#os.mkdir(filepath)

	comPhi = getCOMReactionPlane(com, centerX, centerY)
	psi_gen = np.arctan2(A.iloc[:,1],A.iloc[:,0])
	psi_truth = A.iloc[:,5]

	plotHistograms(A, filepath, model_num, psi_gen, psi_truth, comPhi)

def linear_model():
	train_size = 0.8
	model_num = 5
	model_loss = 'mse'
	filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{model_num}_{model_loss}"
	random_state = 42

	A = io.get_dataset(folder = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles", side = '//A')
	A = A.drop_duplicates()
	com = findCOM(A.iloc[:,8:24])
	A = pd.concat([A, com], axis = 1)
	print(A)
	input()

	train_A, tmpA = train_test_split(A, train_size = train_size, random_state = random_state)
	val_A, test_A = train_test_split(tmpA, train_size = 0.5, random_state = random_state)

	train_X = train_A.iloc[:,24:26]
	train_y = train_A.iloc[:,0:2]
	val_X = val_A.iloc[:,24:26]
	val_y = val_A.iloc[:,0:2]

	print('Training Data:')
	print(train_X.head())
	print(train_y.head())
	print('Validation Data:')
	print(val_X.head())
	print(val_y.head())
	input()

	os.mkdir(filepath)
	test_A.to_pickle(filepath + f'//test_A.pickle')

	model = get_model()
	print('Model Received.')
	model.summary()

	model.compile(optimizer = 'adam', loss = model_loss, metrics=['mae','mse', 'msle'])
	early_stopping = keras.callbacks.EarlyStopping(min_delta = 0.01, patience = 15, monitor='val_loss', restore_best_weights = True)

	print("Starting training:")
	history = model.fit(
		train_X, train_y,
		validation_data = (val_X, val_y),
		batch_size = 1024,
		epochs = 500,
		callbacks=[early_stopping],
		verbose=1
	)

	print('Training completed.')
	model.save(filepath + f'//linear_{model_num}_{model_loss}.h5')

	train_mse = history.history['mse']
	val_mse = history.history['val_mse']
	train_mae = history.history['mae']
	val_mae = history.history['val_mae']
	train_msle = history.history['msle']
	val_msle = history.history['msle']

	f = open(filepath + f'//linear_{model_num}.txt', 'w')
	f.write('Difference: Trained with CoM')
	f.write('\nval_loss:' + str(np.min(val_mse)))
	weights = model.layers[-1].get_weights()
	f.write('\n' + str(weights))
	f.close()

	#Taken from train_cnn to compare vs cnn model
	epochs = range(1, len(train_mae) + 1)
	plt.figure(0)
	plt.plot(epochs, train_mse, color='black', label='Training set')
	plt.plot(epochs, val_mse, 'b', label='Validation set')
	plt.title('')
	plt.xlabel('Epoch')
	plt.ylabel('Mean Squared Error')
	plt.legend()
	plt.savefig(filepath + f'//model{model_num}_mse_{model_loss}Loss.png')

	plt.figure(1)
	plt.plot(epochs, train_mae, color='black', label='Training mae')
	plt.plot(epochs, val_mae, 'b', label='Validation mae')
	plt.title('')
	plt.xlabel('Epoch')
	plt.ylabel('Mean Absolute Error')
	plt.legend()
	plt.savefig(filepath + f'//model{model_num}_mae_{model_loss}Loss.png')
	print('val loss:', np.min(val_mse))

	Q_avg = test_A.iloc[:,0:2]
	test_X = findCOM(test_A.iloc[:,8:24])
	Q_predicted = model.predict([test_X.astype('float')])

	psi_rec = np.arctan2(Q_predicted[:,1],Q_predicted[:,0])
	psi_gen = np.arctan2(Q_avg.iloc[:,1],Q_avg.iloc[:,0])
	psi_truth = test_A.iloc[:,5]

	plotHistograms(test_A,filepath,model_num,psi_gen,psi_truth,psi_rec)

def com_modeler():
	run_type = 1
	#If 1, run direct model
	#If 2, run linear regression model
	if run_type == 1:
		directCOMComparison()
	elif run_type == 2:
		linear_model()




