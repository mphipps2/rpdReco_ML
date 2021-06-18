import os
import sys
sys.path.append('C://Users//Fre Shava Cado//Documents//VSCode Projects//rpdreco//')

import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
#import reco.lib.vis as vis
import reco.lib.io as io

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.optimize import curve_fit

def get_model():
	model = keras.Sequential([
	layers.BatchNormalization(input_shape = [2]),
	layers.Dense(units = 2, activation = 'linear')
	])
	return model

def fit_function(x, A, B, mu, sigma):
	return A+B*np.exp(-(x-mu)**2/(2*sigma**2))

def tester(A, filepath, file_num, psi_gen, psi_truth, psi_rec):
	bins = 50
	pt_nuc = A.iloc[:,4]
	numParticles = A.iloc[:,7]

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
	plt.savefig(filepath + f'//model{file_num}_gen_anglediff.png')

	plt.figure(3)
	plt.hist(psi_truth_rec, bins = bins, density = True)
	plt.plot(truthXspace, fit_function(truthXspace, *truthPopt))
	plt.title(r'$\Psi_{True}-\Psi_{recon}$')
	plt.xlabel('Angle Difference(radians)')
	plt.ylabel('Density Function')
	plt.text(-3,0.3,f'$\\mu={np.round(mean_truth, 3)}\\pm {np.round(error_truth,3)}$,\n $\\sigma={np.round(sigma_truth, 3)}$')
	plt.savefig(filepath + f'//model{file_num}_truth_anglediff.png')

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
	
	df1['nbins'] = pd.cut(x = df1.iloc[:,0], bins =[20,25,30,35,40], labels = [20, 25, 30, 35], right = False, include_lowest = True)
	df2['nbins'] = pd.cut(x = df2.iloc[:,0], bins =[20,25,30,35,40], labels = [20, 25, 30, 35], right = False, include_lowest = True)
	df3['nbins'] = pd.cut(x = df3.iloc[:,0], bins =[20,25,30,35,40], labels = [20, 25, 30, 35], right = False, include_lowest = True)
	df4['nbins'] = pd.cut(x = df4.iloc[:,0], bins =[20,25,30,35,40], labels = [20, 25, 30, 35], right = False, include_lowest = True)
	df5['nbins'] = pd.cut(x = df5.iloc[:,0], bins =[20,25,30,35,40], labels = [20, 25, 30, 35], right = False, include_lowest = True)
	
	means1 = df1.groupby('nbins').mean()
	serr1 = df1.groupby('nbins').sem()
	means2 = df2.groupby('nbins').mean()
	serr2 = df2.groupby('nbins').sem()
	means3 = df3.groupby('nbins').mean()
	serr3 = df3.groupby('nbins').sem()
	means4 = df4.groupby('nbins').mean()
	serr4 = df4.groupby('nbins').sem()
	means5 = df5.groupby('nbins').mean()
	serr5 = df5.groupby('nbins').sem()

	plt.figure(4)
	ax = plt.figure(4).gca()
	ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, integer=True))
	plt.plot(means1.index, means1.sigma_gen, 'go', label = r'$5\leq\ p_T^{nuclear}<15$ MeV')
	plt.errorbar(means1.index, means1.sigma_gen, yerr=serr1.sigma_gen, fmt = 'go')
	plt.plot(means2.index, means2.sigma_gen, 'bo', label = r'$15\leq p_T^{nuclear}<25$ MeV',)
	plt.errorbar(means2.index, means2.sigma_gen, yerr=serr2.sigma_gen, fmt = 'bo')
	plt.plot(means3.index, means3.sigma_gen, 'ro', label = r'$25\leq p_T^{nuclear}<35$ MeV')
	plt.errorbar(means3.index, means3.sigma_gen, yerr=serr3.sigma_gen, fmt = 'ro')
	plt.plot(means4.index, means4.sigma_gen, 'mo', label = r'$35\leq p_T^{nuclear}<45$ MeV')
	plt.errorbar(means4.index, means4.sigma_gen, yerr=serr4.sigma_gen, fmt = 'mo')
	plt.plot(means5.index, means5.sigma_gen, 'ko', label = r'$45\leq p_T^{nuclear}$ MeV')
	plt.errorbar(means5.index, means5.sigma_gen, yerr=serr5.sigma_gen, fmt = 'ko')
	plt.title('Gen Residuals')
	plt.xlabel(r'$N_{neutrons}$')
	plt.ylabel(r'$\sigma_{\Psi_{gen}-\Psi_{rec}}$')
	plt.ylim(bottom = 0, top = 1.8)
	plt.legend()
	plt.savefig(filepath +  f'//model{file_num}_gen_stratsigmas.png')

	plt.figure(5)
	ax = plt.figure(5).gca()
	ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, integer=True))
	plt.plot(means1.index, means1.sigma_truth, 'go', label = r'$5\leq p_T^{nuclear}<15$ MeV')
	plt.errorbar(means1.index, means1.sigma_truth, yerr=serr1.sigma_truth, fmt = 'go')
	plt.plot(means2.index, means2.sigma_truth, 'bo', label = r'$15\leq p_T^{nuclear}<25$ MeV')
	plt.errorbar(means2.index, means2.sigma_truth, yerr=serr2.sigma_truth, fmt = 'bo')
	plt.plot(means3.index, means3.sigma_truth, 'ro', label = r'$25\leq p_T^{nuclear}<35$ MeV')
	plt.errorbar(means3.index, means3.sigma_truth, yerr=serr3.sigma_truth, fmt = 'ro')
	plt.plot(means4.index, means4.sigma_truth, 'mo', label = r'$35\leq p_T^{nuclear}<45$ MeV')
	plt.errorbar(means4.index, means4.sigma_truth, yerr=serr4.sigma_truth, fmt = 'mo')
	plt.plot(means5.index, means5.sigma_truth, 'ko', label = r'$45\leq p_T^{nuclear}$ MeV')
	plt.errorbar(means5.index, means5.sigma_truth, yerr=serr5.sigma_truth, fmt = 'ko')
	plt.title('Truth Residuals')
	plt.xlabel(r'$N_{neutrons}$')
	plt.ylabel(r'$\sigma_{\Psi_{truth}-\Psi_{rec}}$')
	plt.ylim(bottom = 0, top = 2.4)
	plt.legend()
	plt.savefig(filepath + f'//model{file_num}_truth_stratsigmas.png')

	plt.show()

def regular_model():
	model_num = 20
	model_loss = 'CoM'
	filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{model_num}_{model_loss}"
	os.mkdir(filepath)

	A = io.get_dataset(folder = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles", side = '//A')
	A = A.drop_duplicates()
	A_sub = process.subtract_signals(A)

	psi_rec = np.arctan2(A_sub.iloc[:,3],A_sub.iloc[:,2])
	psi_gen = np.arctan2(A_sub.iloc[:,1],A_sub.iloc[:,0])
	psi_truth = A_sub.iloc[:,5]

	tester(A, filepath, model_num, psi_gen, psi_truth, psi_rec)

def linear_model():
	train_size = 0.65
	model_num = 22
	model_loss = 'mse'
	filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{model_num}_{model_loss}"
	random_state = 42

	A = io.get_dataset(folder = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles", side = '//A')
	A = A.drop_duplicates()
	A_sub = process.subtract_signals(A)

	train_A, tmpA = train_test_split(A_sub, train_size = train_size, random_state = random_state)
	val_A, test_A = train_test_split(A_sub, train_size = train_size, random_state = random_state)

	train_X = train_A.iloc[:,2:4]
	train_y = train_A.iloc[:,0:2]
	val_X = val_A.iloc[:,2:4]
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
	f.write('\nval_loss:' + str(val_mse))
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
	test_X = test_A.iloc[:,2:4]
	Q_predicted = model.predict([test_X.astype('float')])

	psi_rec = np.arctan2(Q_predicted[:,1],Q_predicted[:,0])
	psi_gen = np.arctan2(Q_avg.iloc[:,1],Q_avg.iloc[:,0])
	psi_truth = test_A.iloc[:,5]

	tester(test_A,filepath,model_num,psi_gen,psi_truth,psi_rec)

def com_modeler():
	run_type = 1
	#If 1, run direct model
	#If 2, run linear regression model
	if run_type == 1:
		regular_model()
	elif run_type == 2:
		linear_model()




