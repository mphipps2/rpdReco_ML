from math import gamma
import os
from re import I
import sys
from pathlib import Path

import lib.norm as norm
import lib.models as models
import lib.process as process
import lib.vis_mpl as vis_mpl
import lib.io as io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

if __name__ == '__main__':
    #used for training bdt models
    debug = False
    train_size = 0.8
    model_num = 1
    model_type = "bdt"
    model_loss = "rmse"
    use_unit_vector = True
    do_z_norm = False #recommended setting: False
    make_two_train_samples = False
    subtract = True
    two_trainer_ratio = 0.6
    two_trainer_filename = "40batch"
    scenario = "ToyFermi_qqFibers_LHC_noPedNoise/"
#    scenario = "ToyFermi_qRods_LHC_noPedNoise/"
    data_path = "../data/"+scenario
    model_path = "../models/"+scenario
    output_path = "/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/"+scenario
    data_file = "ToyFermi_qqFibers_LHC_noPedNoiseA_nonlinearDigi.pickle"
    random_state = 42
    my_train_size = 0.8
    my_booster = 'dart' #pick between gbtree, gblinear, and dart
    my_learning_rate = 0.05
    my_max_depth = 9
    my_patience = 15


    print('Getting Dataset...')
    A = io.get_dataset_peak(filename = data_path+data_file)
    A = A.drop_duplicates()
    if subtract:
        A = process.subtract_signals_peak(A)
    if debug == True:    
        print("A: ", A)
        print('columns: ', A.columns)    

    train_A, tmpA = train_test_split(A, test_size = 1.-train_size, random_state = random_state) 
    val_A, test_A = train_test_split(tmpA, test_size = 0.5, random_state = random_state)

    if make_two_train_samples:
            train_A, train_B = train_test_split(train_A, test_size= 1.-two_trainer_ratio, random_state = random_state)
            val_A, val_B = train_test_split(val_A, test_size= 1.-0.5, random_state = random_state)
            train_B.to_pickle(data_path + f'train2_{two_trainer_filename}.pickle')
            val_B.to_pickle(data_path + f'val2_{two_trainer_filename}.pickle')
    
    if do_z_norm:
        scaler = StandardScaler()
        train_X = train_A.iloc[:,6:22]
        train_X = scaler.fit_transform(train_X)
        val_X = val_A.iloc[:,6:22]
        val_X = scaler.transform(val_X)
        test_X = test_A.iloc[:,6:22]
        test_X = scaler.transform(test_X)
    else:
        train_X = train_A.iloc[:,6:22]
        val_X = val_A.iloc[:,6:22]
        test_X = test_A.iloc[:,6:22] 

    test_A.to_pickle(data_path + f'test_A.pickle')
    if do_z_norm:
        np.save(data_path + f'test_A_znorm.npy',test_X)
    
    train_A = train_A.to_numpy()
    val_A = val_A.to_numpy()    
    test_A = test_A.to_numpy()

    train_y = train_A[:,1:3]
    val_y = val_A[:,1:3]
    if use_unit_vector:
        train_y = norm.get_unit_vector(train_y)
        val_y = norm.get_unit_vector(val_y)

    if debug == True:
        print('Training Data:\n')
        print(train_X)
        print(train_y)
        print('Validation Data:\n')
        print(val_X)
        print(val_y)

    train_y_x = train_y[:,0]
    train_y_y = train_y[:,1]
    val_y_x = val_y[:,0]
    val_y_y = val_y[:,1]

    eval_setX = [(train_X, train_y_x), (val_X, val_y_x)]
    eval_setY = [(train_X, train_y_y), (val_X, val_y_y)]    

    #Two trees, one to predict Qx and one to predict Qy
    modelX = xgb.XGBRegressor(booster = my_booster, learning_rate = my_learning_rate, max_depth = my_max_depth, objective = 'reg:squarederror')
    modelY = xgb.XGBRegressor(booster = my_booster, learning_rate = my_learning_rate, max_depth = my_max_depth, objective = 'reg:squarederror')
    
    print('Starting Training Model X...')
    modelX.fit(
		train_X, train_y_x, 
		eval_metric=[model_loss],
		eval_set=eval_setX, 
		early_stopping_rounds = my_patience, 
		verbose=True, 
		callbacks=[xgb.callback.EarlyStopping(rounds=my_patience, save_best=True)]
	)

    print('Starting Training Model Y...')
    modelY.fit(
		train_X, train_y_y, 
		eval_metric=[model_loss],
		eval_set=eval_setY, 
		early_stopping_rounds = my_patience, 
		verbose=True, 
		callbacks=[xgb.callback.EarlyStopping(rounds=my_patience, save_best=True)]
	)

    print('Training Completed.')

    if make_two_train_samples:
        modelX.save_model(model_path + f'{model_type}X_{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.json')
        modelY.save_model(model_path + f'{model_type}Y_{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.json')
    else:
        modelX.save_model(model_path + f'{model_type}X_{model_num}_{model_loss}.json')
        modelY.save_model(model_path + f'{model_type}Y_{model_num}_{model_loss}.json')

    eval_result = modelX.evals_result()
    train_mse = eval_result['validation_0']['rmse']
    val_mse = eval_result['validation_1']['rmse']
    epochs = range(1, len(train_mse) + 1)

    Path(output_path + f'{model_type}_model{model_num}').mkdir(parents=True, exist_ok=True)
    if make_two_train_samples:
        vis_mpl.PlotTrainingComp(len(epochs), train_mse, val_mse, "Mean Squared Error", output_path+f'{model_type}_model{model_num}/ValTrainingComp_{model_type}_model{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.png')
    else:
        vis_mpl.PlotTrainingComp(len(epochs), train_mse, val_mse, "Mean Squared Error", output_path+f'{model_type}_model{model_num}/ValTrainingComp_{model_type}_model{model_num}_{model_loss}_.png')

    print('loss: ', np.min(train_mse))
    print('val loss:', np.min(val_mse))    
