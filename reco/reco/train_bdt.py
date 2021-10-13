from math import gamma
import os
from re import I
import sys

sys.path.append('/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/rpdreco/reco')


import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
#import reco.lib.vis as vis
import reco.lib.io as io

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

def normalize(dataset):
	for i in range(len(dataset.columns)):
		dataset.iloc[:,i] = dataset.iloc[:,i]-dataset.iloc[:,i].mean()
		dataset.iloc[:,i] = dataset.iloc[:,i]/dataset.iloc[:,i].std()

	return dataset
	
def train_bdt():
	#boosted decision tree, 16 inputs->psi coordinate
	train_size = 0.8
	model_num = 13
	model_loss = 'mse'
	random_state = 42
	patience = 15
	filepath = f'/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/bdt_models/model_{model_num}_{model_loss}/'

	print('Getting Dataset...')

	A = io.get_dataset('/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/data/', side = 'A', subtract = True)
	A = A.drop_duplicates()
	print('A: ', A)
	print('columns: ', A.columns)
	train_A, tmpA = train_test_split(A, test_size = 1.-train_size, random_state = random_state)
	val_A, test_A = train_test_split(tmpA, test_size = 0.5, random_state = random_state)

	print("Saving Data Set...")
	os.mkdir(filepath)
	test_A.to_pickle(filepath + 'test_A.pickle')

	train_X = train_A.iloc[:,8:24]
	train_y_x = train_A.iloc[:,0]
	train_y_y = train_A.iloc[:,1]
	val_X = val_A.iloc[:,8:24]
	val_y_x = val_A.iloc[:,0]
	val_y_y = val_A.iloc[:,1]

	print(train_X.shape)
	print(train_y_x.shape)

	eval_setX = [(train_X, train_y_x), (val_X, val_y_x)]
	eval_setY = [(train_X, train_y_y), (val_X, val_y_y)]

	modelX = xgb.XGBRegressor(booster = 'dart', learning_rate = 0.05, max_depth = 9, objective="reg:squarederror")
	modelY = xgb.XGBRegressor(booster = 'dart', learning_rate = 0.05, max_depth = 9, objective="reg:squarederror")
	
	modelX.fit(
		train_X, train_y_x, 
		eval_metric=['rmse'],
		eval_set=eval_setX, 
		early_stopping_rounds = patience, 
		verbose=True, 
		callbacks=[xgb.callback.EarlyStopping(rounds=patience, save_best=True)]
	)

	modelY.fit(
		train_X, train_y_y, 
		eval_metric=['rmse'],
		eval_set=eval_setY, 
		early_stopping_rounds = patience, 
		verbose=True, 
		callbacks=[xgb.callback.EarlyStopping(rounds=patience, save_best=True)]
	)
	

	modelX.save_model(filepath + f'bdtX_{model_num}_{model_loss}.json')
	modelY.save_model(filepath + f'bdtY_{model_num}_{model_loss}.json')

	eval_result = modelX.evals_result()
	train_mse = eval_result['validation_0']['rmse']
	val_mse = eval_result['validation_1']['rmse']
	epochs = range(1, len(train_mse) + 1)
	
	plt.figure(0)
	plt.plot(epochs, train_mse, color='black', label='Training set')
	plt.plot(epochs, val_mse, 'b', label='Validation set')
	plt.title('')
	plt.ylim([0.5,1.5*np.min(val_mse)])
	plt.xlabel('Epoch')
	plt.ylabel('Mean Squared Error')
	plt.legend()
	plt.savefig(filepath + f'model{model_num}_mse_{model_loss}Loss.png')

	#to be implemented
	#xgb.plot_tree(model, fmap=filepath + f'bdt_{model_num}_{model_loss}.png', num_trees = model.best_iteration, rankdir='LR')

