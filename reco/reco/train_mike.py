
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
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))


if __name__ == '__main__':

    debug = False
    train_size = 0.1
    test_val_size = 0.9
    val_size = 0.1
    test_size = 0.9
    model_num = 21
    model_type = "cnn"
#    model_loss = "root_mean_squared_error"
    model_loss = "mse"
    use_neutrons = False
    use_unit_vector = True
    do_z_norm = False
    make_two_train_samples = False
    do_truth_pos_plot = True
    do_reco_pos_plot = True
    do_unsubtracted_channel_plot = False
    do_subtracted_channel_plot = False
    use_padding = True
    subtract = True
    two_trainer_ratio = 0.6
    two_trainer_filename = "40batch"
    scenario = "ToyFermi_qqFibers_LHC_noPedNoise/"
#    scenario = "ToyFermi_qpFibers_LHC_noPedNoise/"
#    scenario = "ToyFermi_mini_qpFibers_LHC_noPedNoise/"
#    scenario = "ToyFermi_qRods_LHC_noPedNoise/"
    data_path = "../data/"+scenario
    model_path = "../models/"+scenario
    output_path = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Training/"+scenario
    data_file = "ToyFermi_qqFibers_LHC_noPedNoiseA.pickle"
    data_file_B = "ToyFermi_qqFibers_LHC_noPedNoiseB.pickle"
    test_file = 'test_A_800k.pickle'
    test_file_B = 'test_B_800k.pickle'
#    data_file = "ToyFermi_qpFibers_LHC_noPedNoiseA.pickle"
#    data_file = "ToyFermi_mini_qpFibers_LHC_noPedNoiseA.pickle"
#    data_file = "A.pickle"
    random_state = 42
    my_batch_size = 2048
    my_epochs = 500
    my_optimizer = 'adam'
    my_metrics = ['mae','mse']
    #my_metrics = [tensorflow.keras.metrics.RootMeanSquaredError(name='rmse')]
    my_monitor = 'val_loss'
    my_patience = 25
    my_min_delta = 0.0001

    print("Tensorflow version: ", tensorflow.__version__)
    #print(device_lib.list_local_devices())
    print("# of GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    #tensorflow.test.is_gpu_available()
    
    print("Getting Dataset...")

    A = io.get_dataset_peak(filename = data_path+data_file)
    A = A.drop_duplicates()
    A_np = A.to_numpy()

    B = io.get_dataset_peak(filename = data_path+data_file_B)
    B = B.drop_duplicates()
    B_np = B.to_numpy()

    
    if do_unsubtracted_channel_plot:
        vis_root.PlotUnsubtractedChannels(A_np[:,6:22], output_path)
    if do_truth_pos_plot:
        print("qx: ", A_np[:,1], "qy: ", A_np[:,2])
        vis_root.PlotTruthPos(A_np[:,1], A_np[:,2], "genPos", output_path)        
    if subtract:
        A = process.subtract_signals_peak(A)
        B = process.subtract_signals_peak(B)
    if do_reco_pos_plot:
        print("qx: ", A_np[:,6], "qy: ", A_np[:,21])
        com_A = process.findCOM_np(A_np[:,6:22])
        vis_root.PlotRecoPos(com_A[:,0], com_A[:,1], "com", output_path)

    if do_subtracted_channel_plot:
        vis_root.PlotSubtractedChannels(A_np[:,6:22], output_path)

        
    if debug == True:    
        print("A: ", A)
        print('columns: ', A.columns)

    #using state 42 for verification purposes
    train_A, tmpA = train_test_split(A, test_size = test_val_size, train_size = train_size, random_state = random_state)
#    val_A, test_A = train_test_split(tmpA, train_size = 0.5, random_state = random_state)
    val_A, test_A = train_test_split(tmpA, test_size = test_size, train_size = val_size, random_state = random_state)

    train_B, tmpB = train_test_split(B, test_size = test_val_size, train_size = train_size, random_state = random_state)
#    val_B, test_B = train_test_split(tmpB, train_size = 0.5, random_state = random_state)
    val_B, test_B = train_test_split(tmpB, test_size = test_size, train_size = val_size, random_state = random_state)


    
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

    if debug == True:
        print("Saving Data Set")
    test_A.to_pickle(data_path + test_file)
    test_B.to_pickle(data_path + test_file_B)
    if do_z_norm:
        np.save(data_path + f'test_A_znorm.npy',test_X)
    
    train_A = train_A.to_numpy()
    val_A = val_A.to_numpy()    
    test_A = test_A.to_numpy()

    if not do_z_norm:
        train_X = train_A[:,6:22]
        val_X = val_A[:,6:22]
        test_X = test_A[:,6:22]    

    train_neutrons = train_A[:, 0]
    val_neutrons = val_A[:, 0]
    test_neutrons = test_A[:, 0]
    
    # blur to zdcs resolution
    train_neutrons = process.blur_neutron(train_neutrons)
    val_neutrons = process.blur_neutron(val_neutrons)
    test_neutrons = process.blur_neutron(test_neutrons)

    train_y = train_A[:,1:3]
    val_y = val_A[:,1:3]
    if use_unit_vector:
        train_y = norm.get_unit_vector(train_y)
        val_y = norm.get_unit_vector(val_y)

    if do_truth_pos_plot:
        vis_root.PlotTruthPos(train_y[:,0], train_y[:,1], "unitVec", output_path)        
    if model_type == 'cnn' or model_type == 'cnn_test':
        #reshape from (None, 16) to (None, 6, 6, 1)
        if use_padding: 
            train_X = process.reshape_signal(train_X)
            val_X = process.reshape_signal(val_X)
        else:
            train_X = process.reshape_signal(train_X,normalization=False, flatten=False, padding = False)
            val_X = process.reshape_signal(val_X,normalization=False, flatten=False, padding = False)
    if use_neutrons:
        train_X = {"neutron_branch": train_neutrons, 'rpd_branch': train_X}
        val_X = {"neutron_branch": val_neutrons, 'rpd_branch': val_X}
        
    #sanity check
    if debug == True:
        print('Training Data:\n')
        print(train_X)
        print(train_y)
        print('Validation Data:\n')
        print(val_X)
        print(val_y)

        
    # normalizer = process.get_normalizer(rpdSignals)
    
    if model_type == 'cnn':
        model = models.get_cnn()
    elif model_type == 'cnn_test':
        model = models.get_cnn_test()
    elif model_type == 'fcn':
        model = models.get_fcn()
    elif model_type == 'fcn_test':
        model = models.get_fcn_test()
    elif model_type == 'linear':
        model = models.get_linear()
    elif model_type == 'linear_test':
        model = models.get_linear_test()
    elif model_type == 'bdt':
        model = models.get_bdt()
    elif model_type == 'bdt_test':
        model = models.get_bdt_test()
    else:
        print('unknown model (hint: use lower case)')
        
    model.summary()
    model.compile(optimizer = my_optimizer, loss = tensorflow.keras.metrics.mean_squared_error, metrics=my_metrics)
#    model.compile(optimizer = my_optimizer, loss = root_mean_squared_error, metrics=tensorflow.keras.metrics.RootMeanSquaredError(name='rmse'))
    early_stopping = keras.callbacks.EarlyStopping(min_delta = my_min_delta, patience = my_patience, monitor=my_monitor, restore_best_weights = True)

    print("Start Training:")
    print("min_delta: ", my_min_delta, " patience: ", my_patience, " monitor: ", my_monitor)
    history = model.fit(
        train_X, train_y,
        validation_data = (val_X, val_y),
        batch_size = my_batch_size,
        epochs = my_epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Training completed.")
    if make_two_train_samples:
        model.save(model_path + f'model{model_type}_{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.h5') 
    else:
        model.save(model_path + f'model{model_type}_{model_num}_{model_loss}.h5') 
    train_mse = history.history['mse']
    val_mse = history.history['val_mse']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    Path(output_path + f'model{model_num}').mkdir(parents=True, exist_ok=True)
    f = open(output_path + f'model{model_num}/linear_{model_num}.txt', 'w')
    f.write('\nval_loss:' + str(np.min(val_mse)))
    weights = model.layers[-1].get_weights()
    f.write('\n' + str(weights))
    f.close()

    f = open(output_path + f'model{model_num}/linear_{model_num}_{model_loss}_summary.txt', 'w')
    model.summary(print_fn = lambda x: f.write(x+'\n'))
    f.close()

    if make_two_train_samples:
        vis_root.PlotTrainingComp(len(train_mae), train_mse, val_mse, model_loss, output_path+f'model{model_num}/ValTrainingComp_{model_type}_model{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.png')
    else:
        vis_root.PlotTrainingComp(len(train_mae), train_mse, val_mse, model_loss, output_path+f'model{model_num}/ValTrainingComp_{model_type}_model{model_num}_{model_loss}_.png')
    print('loss: ', np.min(train_mse))
    print('val loss:', np.min(val_mse))


