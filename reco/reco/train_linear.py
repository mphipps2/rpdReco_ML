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

def get_model():
    #linear model, 16 input charges -> 2 positional coordinates
    model = keras.Sequential([
        layers.Dense(units=2, input_shape=[16])
    ])
    return model


def train_linear():
    train_size = 0.65
    model_num = 4
    model_loss = 'mae'
    filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{model_num}_{model_loss}"
    os.mkdir(filepath)
    print("Getting Dataset...")
    random_state = 42

    A = io.get_dataset(folder = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles", side = '//A')
    A = A.drop_duplicates()
    print("A: ", A)
    print('columns: ', A.columns)
    A_sub = process.subtract_signals(A)
    #sets input as rpd_charges and output as avgQPos
    X = A_sub.iloc[:,8:24]
    y = A_sub.iloc[:,0:2]
    print("Dataset retrieved.")

    #using state 42 for verification purposes
    train_X, tmpX, train_y, tmpy = train_test_split(X, y, train_size = train_size, random_state = random_state)
    test_X, val_X, test_y, val_y = train_test_split(tmpX,tmpy, train_size = train_size, random_state = random_state)

    print("Saving Data Set")
    test_X.to_pickle(filepath + '//test_X.pickle')
    test_y.to_pickle(filepath + '//test_y.pickle')

    model = get_model()
    print("Model Received.")
    model.summary()
    model.compile(optimizer = 'adam', loss = model_loss, metrics=['mae','mse', 'msle'])
    early_stopping = keras.callbacks.EarlyStopping(min_delta = 0.01, monitor='val_loss', patience = 10, restore_best_weights = True)

    print("Starting training:")
    history = model.fit(
        train_X, train_y,
        validation_data = (val_X, val_y),
        batch_size = 128,
        epochs = 250,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Training completed.")
    model.save(filepath + f'//linear_{model_num}_{model_loss}.h5')
    
    

    train_mse = history.history['mse']
    val_mse = history.history['val_mse']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    #train_msle = history.history['msle']
    #val_msle = history.history['msle']

    f = open(filepath + f'//linear_{model_num}.txt', 'w')
    f.write('Difference: Using MAE')
    f.write('\nval_loss:' + str(val_mae))
    weights = model.layers[-1].get_weights()
    f.write('\n' + str(weights))
    f.close()

    #Taken from train_cnn to compare vs cnn model
    epochs = range(1, len(train_mae) + 1)
    plt.figure(1)
    plt.plot(epochs, train_mse, 'o', color='black', label='Training set')
    plt.plot(epochs, val_mse, 'b', label='Validation set')
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(filepath + f'//model{model_num}_mse_{model_loss}Loss.png')

    plt.figure(2)
    plt.plot(epochs, train_mae, 'o', color='black', label='Training mae')
    plt.plot(epochs, val_mae, 'b', label='Validation mae')
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(filepath + f'//model{model_num}_mae_{model_loss}Loss.png')
    '''
    plt.figure(3)
    plt.plot(epochs, train_msle, 'o', color='black', label='Training msle')
    plt.plot(epochs, val_msle, 'b', label='Validation msle')
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Logarithmic Error')
    plt.legend()
    plt.savefig(filepath + f'//model{model_num}_msle_{model_loss}Loss.png')
    '''
    print('val loss:', val_mse)


