
from re import I
import sys
from pathlib import Path

# Add package to python path in .bashrc file: eg) export PYTHONPATH='/home/mike/Desktop/rpdReco/reco/'
import lib.norm as norm
import lib.models as models
import lib.process as process
import lib.vis_root as vis_root
import lib.vis_mpl as vis_mpl
import lib.io as io

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.python.client import device_lib
import tensorflow

# Note: to use this you should first create/evaluate individual models using train_mike.py. After this you can run the ensemble predictions through test_mike.py

if __name__ == '__main__':

    debug = False
    model_num = 301
    nmembers = 2
    base_model_files = ["modelcnn_100_mse_twotrainer0.6.h5", "modelfcn_150_mse_twotrainer0.6.h5"]
    model_type = "stack_ensemble"
    model_type_1_label = "fcn_ensemble_model"
    model_type_2_label = "cnn_model"
    model_loss = "mse"
    two_trainer_filename = "40batch"
    scenario = "ToyFermi_qqFibers_LHC_noPedNoise/"
#    scenario = "ToyFermi_qRods_LHC_noPedNoise/"
    use_unit_vector = True
    data_file = "test_A_40batch.pickle"
    data_path = "../data/"+scenario
    model_path = "../models/"+scenario
    output_path = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Training/" + scenario
    upperRange_gen = 0.9
    upperRange_truth = 2.5
    random_state = 42
    my_batch_size = 2048
    my_epochs = 500
    my_optimizer = 'adam'
    my_metrics = ['mae','mse']
    my_monitor = 'val_loss'
    my_patience = 25
    my_min_delta = 0.01

    
    print("Tensorflow version: ", tensorflow.__version__)
    #print(device_lib.list_local_devices())
    print("# of GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    #tensorflow.test.is_gpu_available()
    
    print("Getting Dataset...")

#    train_A = io.get_dataset(filename = data_path + "train2_" + two_trainer_filename + ".pickle", subtract = False)
    train_A = io.get_dataset_peak(filename = data_path + "train2_" + two_trainer_filename + ".pickle")
#    val_A = io.get_dataset(filename = data_path + "val2_" + two_trainer_filename + ".pickle", subtract = False)
    val_A = io.get_dataset_peak(filename = data_path + "val2_" + two_trainer_filename + ".pickle")
    test_A = pd.read_pickle(data_path + data_file).to_numpy()

    
    if debug == True:    
        print("train_A: ", train_A)
        print("val_A: ", val_A)
        print('columns: ', train_A.columns)


        
    train_A = train_A.to_numpy()
    val_A = val_A.to_numpy()    
        
    train_X = train_A[:,6:22]
    val_X = val_A[:,6:22]
    train_y = train_A[:,1:3]
    val_y = val_A[:,1:3]
    test_X = test_A[:,6:22]
    test_y = test_A[:,1:3]
    if use_unit_vector:
        train_y = norm.get_unit_vector(train_y)
        val_y = norm.get_unit_vector(val_y)
        test_y = norm.get_unit_vector(test_y)
        
    train_X_CNN = process.reshape_signal(train_X)
    val_X_CNN = process.reshape_signal(val_X)
    test_X_CNN = process.reshape_signal(test_X)

    psi_truth_A = test_A[:,3]
    pt_nuc_A = test_A[:,5] * 1000
    neutrons_A = test_A[:,0]
    neutrons_A_smeared = process.blur_neutron(test_A[:,0])
    psi_gen_A = np.arctan2(test_y[:,1],test_y[:,0])
    
    #sanity check
    if debug == True:
        print('Training Data:\n')
        print(train_X)
        print(train_y)
        print('Validation Data:\n')
        print(val_X)
        print(val_y)


    print("train_X ", train_X)
    print("train_X_CNN", train_X_CNN)
    print("train_X shape ", np.shape(train_X))
    print("train_X_CNN shape", np.shape(train_X_CNN))
    print("val_X shape ", np.shape(val_X))
    print("val_X_CNN shape", np.shape(val_X_CNN))

    model_cnn = keras.models.load_model(model_path + base_model_files[0], compile = False)
    model_fcn = keras.models.load_model(model_path + base_model_files[1], compile = False)        
    model_stack =  models.get_meta_fcn_test()
    train_y_CNN = model_cnn.predict([train_X_CNN.astype('float')])
    train_y_FCN = model_fcn.predict([train_X.astype('float')])
    val_y_CNN = model_cnn.predict([val_X_CNN.astype('float')])
    val_y_FCN = model_fcn.predict([val_X.astype('float')])
    
    model_stack.summary()
    model_stack.compile(optimizer = my_optimizer, loss = model_loss, metrics=my_metrics)
    early_stopping = keras.callbacks.EarlyStopping(min_delta = my_min_delta, patience = my_patience, monitor=my_monitor, restore_best_weights = True)

    train_y_base_models = np.concatenate([train_y_CNN,train_y_FCN], axis=1)
    val_y_base_models = np.concatenate([val_y_CNN,val_y_FCN], axis = 1)
    print("Start Training:")
    print("min_delta: ", my_min_delta, " patience: ", my_patience, " monitor: ", my_monitor)
    history = model_stack.fit(
        train_y_base_models, train_y,
        validation_data = (val_y_base_models, val_y),
        batch_size = my_batch_size,
        epochs = my_epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Training completed.")

    model_stack.save(model_path + f'model{model_type}_{model_num}_{model_loss}.h5') 
    train_mse = history.history['mse']
    val_mse = history.history['val_mse']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    Path(output_path + f'model{model_num}').mkdir(parents=True, exist_ok=True)
#    f = open(output_path + f'model{model_num}/{model_num}.txt', 'w')
#    f.write('\nval_loss:' + str(np.min(val_mse)))
#    weights = model_stack.layers[-1].get_weights()
#    f.write('\n' + str(weights))
#    f.close()

#    f = open(output_path + f'model{model_num}/{model_num}_{model_loss}_summary.txt', 'w')
#    model_stack.summary(print_fn = lambda x: f.write(x+'\n'))
#    f.close()


    vis_mpl.PlotTrainingComp(len(train_mae), train_mse, val_mse, "Mean Squared Error", output_path+f'model{model_num}/ValTrainingComp_{model_type}_model{model_num}_{model_loss}_.png')

    print('loss: ', np.min(train_mse))
    print('val loss:', np.min(val_mse))

    Q_predicted_CNN = model_cnn.predict([test_X_CNN.astype('float')])
    psi_rec_CNN = np.arctan2(Q_predicted_CNN[:,1],Q_predicted_CNN[:,0])
    psi_gen_rec_CNN = process.GetPsiResidual_np(psi_gen_A, psi_rec_CNN)
    
    Q_predicted_FCN = model_fcn.predict([test_X.astype('float')])
    psi_rec_FCN = np.arctan2(Q_predicted_FCN[:,1],Q_predicted_FCN[:,0])
    psi_gen_rec_FCN = process.GetPsiResidual_np(psi_gen_A, psi_rec_FCN)

    
    meta_model_input = np.concatenate([Q_predicted_CNN,Q_predicted_FCN], axis=1)
    Q_pred_stack = model_stack.predict(meta_model_input)
    print("stack output shape: ", np.shape(Q_pred_stack))
    psi_rec_stack = np.arctan2(Q_pred_stack[:,1],Q_pred_stack[:,0])
    psi_gen_stack = process.GetPsiResidual_np(psi_gen_A, psi_rec_stack)
    psi_truth_stack = process.GetPsiResidual_np(psi_truth_A, psi_rec_stack)
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    #    vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_wAvgEnsemble, upperRange_gen, "psi_gen_wAvgEnsemble", output_path, save_residuals = False)
    #    vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_wAvgEnsemble, upperRange_truth, "psi_truth_wAvgEnsemble", output_path, save_residuals = False)
    vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_rec_CNN, upperRange_gen, "psi_gen_CNN", output_path, save_residuals = False)
    vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_rec_FCN, upperRange_gen, "psi_gen_FCN", output_path, save_residuals = False)
    vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_stack, upperRange_gen, "psi_gen_stack", output_path, save_residuals = False)
    vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_stack, upperRange_truth, "psi_truth_stack", output_path, save_residuals = False)
    vis_root.PlotRatio_ptnuc_hist(pt_nuc_A, pt_nuc_A, psi_gen_rec_CNN,  psi_gen_stack, upperRange_gen, model_type_2_label , model_type_1_label, output_path, is_gen = True, save_residuals = False)
