
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
from sklearn.preprocessing import StandardScaler

from deepstack.ensemble import DirichletEnsemble
from deepstack.ensemble import StackEnsemble
from deepstack.base import KerasMember
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Note: to use this you should first create/evaluate individual models using train_mike.py. After this you can run the ensemble predictions through test_mike.py

if __name__ == '__main__':

    debug = False
    model_num = 300    
    nmembers = 2
    base_model_files = ["modelcnn_100_mse_twotrainer0.6.h5", "modelfcn_150_mse_twotrainer0.6.h5"]
    two_trainer_filename = "40batch"
    scenario = "ToyFermi_qqFibers_LHC_noPedNoise/"
    #scenario = "ToyFermi_qRods_LHC_noPedNoise/"
    model_type_1_label = "ensemble_model"
    model_type_2_label = "cnn_model"
    use_unit_vector = True
    data_file = "test_A_40batch.pickle"
    data_path = "../data/"+scenario
    model_path = "../models/"+scenario
    output_path = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Training/" + scenario + "Model" + str(model_num) + "/"
    upperRange_gen = 0.9
    upperRange_truth = 2.5
    random_state = 42                        
    print("Tensorflow version: ", tensorflow.__version__)
    #print(device_lib.list_local_devices())
    print("# of GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    #tensorflow.test.is_gpu_available()
    
    print("Getting Dataset...")

#    train_A = io.get_dataset(filename = data_path + "train2_" + two_trainer_filename + ".pickle", subtract = False)
    train_A = io.get_dataset_peak(filename = data_path + "train2_" + two_trainer_filename + ".pickle")
    #val_A = io.get_dataset(filename = data_path + "val2_" + two_trainer_filename + ".pickle", subtract = False)
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
#    wAvgEnsemble = DirichletEnsemble(N=2000 * nmembers, metric=mean_squared_error, maximize=False)
    stack = StackEnsemble()
    stack.model = RandomForestRegressor(verbose=1, n_estimators=200, max_depth=15, n_jobs=20, min_samples_split=20)
#    stack.model = RandomForestRegressor(verbose=1, n_estimators=300 * nmembers, max_depth=nmembers * 2, n_jobs=4)

    

    model_cnn = keras.models.load_model(model_path + base_model_files[0], compile = False)
    member_cnn = KerasMember(name=f'model_cnn', keras_model= model_cnn, train_batches=(train_X_CNN, train_y), val_batches=(val_X_CNN, val_y))
    stack.add_member(member_cnn)
    
    model_fcn = keras.models.load_model(model_path + base_model_files[1], compile = False)    
    member_fcn = KerasMember(name=f'model_fcn', keras_model= model_fcn, train_batches=(train_X, train_y), val_batches=(val_X, val_y))
    stack.add_member(member_fcn)
    
#        wAvgEnsemble.add_member(member)

#    wAvgEnsemble.fit()
#    wAvgEnsemble.describe()
    stack.fit()
    stack.describe(metric=mean_squared_error, maximize=False)
    
    print("Training completed.")

    #    Q_pred_wAvgEnsemble = wAvgEnsemble.predict(X=test_X_A)
    #    psi_rec_wAvgEnsemble = np.arctan2(Q_pred_wAvgEnsemble[:,1],Q_pred_wAvgEnsemble[:,0])
    #    psi_gen_wAvgEnsemble = process.GetPsiResidual_np(psi_gen_A, psi_rec_wAvgEnsemble)
    #    psi_truth_wAvgEnsemble = process.GetPsiResidual_np(psi_truth_A, psi_rec_wAvgEnsemble)

    Q_predicted_CNN = model_cnn.predict([test_X_CNN.astype('float')])
    psi_rec_CNN = np.arctan2(Q_predicted_CNN[:,1],Q_predicted_CNN[:,0])
    psi_gen_rec_CNN = process.GetPsiResidual_np(psi_gen_A, psi_rec_CNN)
    
    Q_predicted_FCN = model_fcn.predict([test_X.astype('float')])
    psi_rec_FCN = np.arctan2(Q_predicted_FCN[:,1],Q_predicted_FCN[:,0])
    psi_gen_rec_FCN = process.GetPsiResidual_np(psi_gen_A, psi_rec_FCN)
    
#    meta_model_input = np.concatenate([test_X_CNN,test_X], axis=1)
    meta_model_input = np.concatenate([Q_predicted_CNN,Q_predicted_FCN], axis=1)
    print("input: ", meta_model_input)
    print("meta_model_input shape ", np.shape(meta_model_input))
    print("ndim: ", meta_model_input.ndim)
    Q_pred_stack = stack.predict(X=meta_model_input)
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
#    vis_root.PlotRatio_ptnuc_hist(pt_nuc_A, pt_nuc_A, psi_gen_stack, psi_gen_rec_CNN, upperRange_gen, model_type_2_label , model_type_1_label, output_path, is_gen = True, save_residuals = False)
    vis_root.PlotRatio_ptnuc_hist(pt_nuc_A, pt_nuc_A, psi_gen_rec_CNN, psi_gen_stack, upperRange_gen, model_type_2_label , model_type_1_label, output_path, is_gen = True, save_residuals = False)
 #   stack.save(folder=model_path + f'ensemble_stack_{model_num}.h5') 
#    wAvgEnsemble.save(folder=model_path + f'ensemble_wAvg_{model_num}.h5') 
