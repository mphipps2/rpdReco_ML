
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

if __name__ == '__main__':

    debug = False
    train_size = 0.8
    model_num = 10
    model_type = "cnn"
    model_loss = "mse"
    use_neutrons = False
    subtract = True
    use_unit_vector = True
    do_z_norm = False
    make_two_train_samples = False
    do_truth_pos_plot = True
    do_subtracted_channel_plot = False
    do_position_resolution = True
    two_trainer_ratio = 0.6
    two_trainer_filename = "40batch"
#    scenario = "ToyFermi_qqFibers_LHC_noPedNoise"
    scenario = "ToyFermi_qRods_LHC_noPedNoise/"
    data_path = "../data/"+scenario
    model_path = "../models/"+scenario
    output_path = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Training/"+scenario
    data_file = "Acharge.pickle"
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

    A = io.get_dataset(filename = data_path + data_file)
    A = A.drop_duplicates()
    
    if subtract:
        A = process.subtract_signals(A)
    if debug == True:    
        print("A: ", A)
        print('columns: ', A.columns)

    #using state 42 for verification purposes
    train_A, tmpA = train_test_split(A, test_size= 1.-train_size, random_state = random_state)
    val_A, test_A = train_test_split(tmpA, train_size = 0.5, random_state = random_state)

    if make_two_train_samples:
            train_A, train_B = train_test_split(train_A, test_size= 1.-two_trainer_ratio, random_state = random_state)
            val_A, val_B = train_test_split(val_A, test_size= 1.-0.5, random_state = random_state)
            train_B.to_pickle(data_path + f'train2_{two_trainer_filename}.pickle')
            val_B.to_pickle(data_path + f'val2_{two_trainer_filename}.pickle')
    
    if do_z_norm:
        scaler = StandardScaler()
        train_X = train_A.iloc[:,8:24]
        train_X = scaler.fit_transform(train_X)
        val_X = val_A.iloc[:,8:24]
        val_X = scaler.transform(val_X)
        test_X = test_A.iloc[:,8:24]
        test_X = scaler.transform(test_X)

    if debug == True:
        print("Saving Data Set")
    test_A.to_pickle(data_path + f'test_A.pickle')
    if do_z_norm:
        np.save(data_path + f'test_A_znorm.npy',test_X)
    
    train_A = train_A.to_numpy()
    val_A = val_A.to_numpy()    
    test_A = test_A.to_numpy()

    if not do_z_norm:
        train_X = train_A[:,8:24]
        val_X = val_A[:,8:24]
        test_X = test_A[:,8:24]    

    train_neutrons = train_A[:, 7]
    val_neutrons = val_A[:, 7]
    test_neutrons = test_A[:, 7]
    
    # blur to zdcs resolution
    train_neutrons = process.blur_neutron(train_neutrons)
    val_neutrons = process.blur_neutron(val_neutrons)
    test_neutrons = process.blur_neutron(test_neutrons)

    train_y = train_A[:,0:2]
    val_y = val_A[:,0:2]
    if use_unit_vector:
        train_y = norm.get_unit_vector(train_y)
        val_y = norm.get_unit_vector(val_y)
    
    if model_type == 'cnn' or model_type == 'cnn_test':
        #reshape from (None, 16) to (None, 6, 6, 1)
        train_X = process.reshape_signal(train_X)
        val_X = process.reshape_signal(val_X)
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

    A_np = A.to_numpy()
    if do_truth_pos_plot:
        vis_root.PlotTruthPos(A_np[:,0], A_np[:,1], output_path)
    if do_subtracted_channel_plot:
        vis_root.PlotSubtractedChannels(A_np[:,8:24], output_path)
        
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
    model.compile(optimizer = my_optimizer, loss = model_loss, metrics=my_metrics)
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
        vis_mpl.PlotTrainingComp(len(train_mae), train_mse, val_mse, "Mean Squared Error", output_path+f'model{model_num}/ValTrainingComp_{model_type}_model{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.png')
    else:
        vis_mpl.PlotTrainingComp(len(train_mae), train_mse, val_mse, "Mean Squared Error", output_path+f'model{model_num}/ValTrainingComp_{model_type}_model{model_num}_{model_loss}_.png')

    print('loss: ', np.min(train_mse))
    print('val loss:', np.min(val_mse))

