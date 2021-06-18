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
    #linear model, 8 input averages -> 2 positional coordinates
    model = keras.Sequential([
        layers.BatchNormalization(input_shape = [8]),
        layers.Dense(units=2, activation = 'linear')
    ])
    return model

def get_averages(data):
    print(data.head(5))
    for row in range(4):
        temp = pd.Series(0, index=np.arange(len(data)))
        for column in range(4):
            temp = temp.add(data.iloc[:, 8 + 4*row + column])
        temp = temp.divide(4)
        data[f'rowAvg_{row}']=temp
    for column in range(4):
        temp = pd.Series(0, index = np.arange(len(data)))
        for row in range(4):
            temp = temp.add(data.iloc[:, 8 + 4*row + column])
        temp = temp.divide(4)
        data[f'colAvg_{column}']=temp
    print(data.head(5))
    print('Continue?')
    x = input()
    if not (x == 'y' or x == 'yes'):
        exit()
    return data
            

def train_linear():
    train_size = 0.65
    model_num = 19
    model_loss = 'mse'
    filepath = f"C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//model_{model_num}_{model_loss}"
    random_state = 42

    print("Getting Dataset...")

    A = io.get_dataset(folder = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles", side = '//A')
    A = A.drop_duplicates()
    print("A: ", A)
    print('columns: ', A.columns)
    A_sub = process.subtract_signals(A)
    #A_sub = get_averages(A_sub)
    #using state 42 for verification purposes
    train_A, tmpA = train_test_split(A_sub, train_size = train_size, random_state = random_state)
    val_A, test_A = train_test_split(tmpA, train_size = train_size, random_state = random_state)

    print("Saving Data Set")
    os.mkdir(filepath)
    test_A.to_pickle(filepath + '//test_A.pickle')
    '''
    test_X.to_pickle(filepath + '//test_X.pickle')
    test_y.to_pickle(filepath + '//test_y.pickle')
    test_psi.to_pickle(filepath + '//psi_truth.pickle')
    '''
    train_X = train_A.iloc[:,24:32]
    train_y = train_A.iloc[:,0:2]
    val_X = val_A.iloc[:,24:32]
    val_y = val_A.iloc[:,0:2]
    
    #sanity check
    print('Training Data:\n')
    print(train_X.head())
    print(train_y.head())
    print('Validation Data:\n')
    print(val_X.head())
    print(val_y.head())
    input()

    model = get_model()
    print("Model Received.")
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

    print("Training completed.")
    model.save(filepath + f'//linear_{model_num}_{model_loss}.h5') 

    train_mse = history.history['mse']
    val_mse = history.history['val_mse']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    train_msle = history.history['msle']
    val_msle = history.history['msle']

    f = open(filepath + f'//linear_{model_num}.txt', 'w')
    f.write('Difference: Reduced averages by two orders of magnitude.')
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
    print('val loss:', np.min(val_mse))


