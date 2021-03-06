import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
import reco.lib.vis as vis
import reco.lib.io as io

from sklearn.model_selection import train_test_split

from lib.Dataloader import get_training_and_validation
import tensorflow.keras as keras
import time
import pandas as pd

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"savefig.bbox": 'tight'})

import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import device_lib
import tensorflow

def get_model():
        nNeutronInput = Input(shape=(1,), name = 'neutron_branch')
        x = Dense(16, activation="relu")(nNeutronInput)
        x = Dense(4, activation="relu")(x)
        x = Dense(1,activation="linear")(x)
        x = Model(inputs=nNeutronInput, outputs=x)

        #input: (6,6), includes padding
        signalInput = Input(shape=(6,6,1), name = 'rpd_branch')
        y = Conv2D(filters = 16, kernel_size = (1,1), padding = 'Same', activation ='relu')(signalInput)
        y = BatchNormalization()(y)
        y = Conv2D(filters = 32, kernel_size = (2,2), padding = 'Same', activation ='relu')(y)
        y = BatchNormalization()(y)
        y = Conv2D(filters =64, kernel_size = (3,3), padding = 'Same', activation ='relu')(y)
        y = BatchNormalization()(y)
        y = Flatten()(y)

        y = Dense(32, activation = "relu")(y)
        y = Dense(16, activation = "relu")(y)
        y = Dense(8, activation = "relu")(y)
        y = Model(inputs=signalInput, outputs=y)

        combined = keras.layers.concatenate([x.output, y.output])
        combined = Dense(8, activation="relu", name = 'combined')(combined)
        combined = Dense(8, activation="relu")(combined)
        Q_avg = Dense(2, activation="tanh", name = 'Q_avg')(combined)
        model = Model(inputs=[x.input, y.input], outputs=[Q_avg])
        return model



def train_cnn():
    train_size = 0.8
    model_num = 13
    model_loss = 'mae'
    do_z_norm = False
    do_neutron_norm = False
    do_q_unit_vector = True
    
    A = io.get_dataset(folder = "./Data/test_set/", side = 'A', subtract = True)
    B = io.get_dataset(folder = "./Data/test_set/", side = 'B', subtract = True)
    print('A: ', A)
    
    if do_z_norm:
        A = z_norm(A)
        B = z_norm(B)
    
    print('A after dropping duplicates and subtracting: ', A)

    # eg) 60/20/20 (training, validation, testing) -- data is shuffled but in a predictable way that's the same every time
    trainA, tmpA, trainB, tmpB = train_test_split(A, B, test_size=1. - train_size, random_state = 42)
    testA, valA, testB, valB = train_test_split(tmpA, tmpB, test_size=0.5, random_state = 42)    
    
    train = trainA.append(trainB).to_numpy()
    val = valA.append(valB).to_numpy()
    test  =  testA.append(testB).to_numpy()
    print('train in numpy: ', train)

    # RPD data
    train_signal = train[:,8:]
    val_signal = val[:,8:]

    # note: by default normalization == false; flatten == false; padding == true
#    train_signal = process_signal(train_signal,True)
    train_signal = process.process_signal(train_signal)
    val_signal = process.process_signal(val_signal)
    
    # Generated Q avg vector
    train_q_avg = train[:, 0:2]
    val_q_avg = val[:, 0:2]
    
    if do_q_unit_vector:
        train_q_vector = norm.get_unit_vector(train_q_avg)
        val_q_vector = norm.get_unit_vector(val_q_avg)

    train_hit = train[:, 7]
    val_hit = val[:, 7]
        
    if do_neutron_norm:
        train_hit = norm.norm_neutron(train_hit)
        val_hit = norm.norm_neutron(val_hit)
            
    train_hit = process.blur_neutron(train_hit)
    val_hit = process.blur_neutron(val_hit)

    # Summed charge map across all events
    integrated_data = train_signal.sum(axis=0).flatten()
    integrated_data = np.reshape(integrated_data(4,4))
    heatmap2d(integrated_data,'viridis','Output/fig/integratedChargeMap_testSet.png')
    
    train_data = {"neutron_branch": train_hit, 'rpd_branch': train_signal}
    train_target = { "Q_avg": train_q_vector}
    val_data = {"neutron_branch": val_hit, 'rpd_branch': val_signal}
    val_target = {"Q_avg": val_q_vector}

    print('Checking GPU Availability:')
    print(device_lib.list_local_devices())
    print(tensorflow.__version__)
    print("# of GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    tensorflow.test.is_gpu_available()
    
    model = get_model()    
    model.summary()
    model.compile(optimizer='adam',loss=[model_loss],metrics=['mae','mse'])
    checkpoint_filepath = '/tmp/checkpoint'
    # use best performing val_loss model for model weights. Set stopping condition s.t. we stop scanning after 10 straight epochs of no improvement
    callbacks_list = [
            keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=1
            ),            
            tensorflow.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='val_loss', 
                    save_best_only=True,
                    verbose=1
            )
    ]

    history = model.fit(train_data, train_target, epochs=100, batch_size=256, callbacks=callbacks_list, validation_data=(val_data,val_target))
    #    print(history.history.keys())

    model.load_weights(checkpoint_filepath)
    model.save(f'./Output/Model/model_{model_num}_{model_loss}Loss.h5')
    
    model_info_file = f'Output/ModelInfo/modelDesign_{model_num}.txt'    
    f = open(model_info_file,'w')
    sys.stdout = f
    f.close()
    
#    plot_model(model, show_shapes=True, rankdir='TB', expand_nested=True, to_file=f'Output/ModelInfo/modelDesign_{model_num}.png')
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    train_mse = history.history['mse']
    val_mse = history.history['val_mse']

    
    epochs = range(1, len(train_mae) + 1)
    plt.figure(1)
    plt.plot(epochs, train_mae, 'o', color='black', label='Training set')
    plt.plot(epochs, val_mae, 'b', label='Validation set')
#    plt.ylim([0.372,0.387])
    plt.ylim([0.383,0.398])
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(f'Output/fig/model{model_num}_mae_{model_loss}Loss.png')

    plt.figure(2)
    plt.plot(epochs, train_mse, 'o', color='black', label='Training mse')
    plt.plot(epochs, val_mse, 'b', label='Validation mse')
#    plt.ylim([0.268,0.283])
    plt.ylim([0.25,0.265])
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(f'Output/fig/model{model_num}_mse_{model_loss}Loss.png')

    
        
    
    #    plot_model(model, show_shapes=True, to_file=f'/mnt/c/Users/mwp89/Desktop/ML/RPD/Model/modelDesign_{model_num}.png')
    
    #    plt.show()


    # plt.plot(epochs, train_mse, 'bo', label='Training mse')
    # plt.plot(epochs, val_mse, 'b', label='Validation mse')
    # plt.title('Training and validation mae')
    # plt.xlabel('Epochs')
    # plt.ylabel('MSE')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'./Output/fig/mse_model_{model_num}.png',format='png')
    # model.save(f'./Output/Model/mse_model_{model_num}.h5')
