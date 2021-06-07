import sys

import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
import reco.lib.vis as vis
import reco.lib.io as io

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib as plt

def get_model():
    #linear model, 16 input charges -> 2 positional coordinates
    Q_Avg = keras.Sequential([
        layers.Dense(units=2, activation = 'linear', input_shape=[16])
    ])
    model = keras.Model(inputs=[Q_Avg.input], outputs=[Q_Avg])
    return model



def train_linear():
    train_size = 0.8
    model_num = 1
    model_loss = 'mse'
    filepath = "C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles"

    A = io.get_dataset(folder = filepath, side = 'A')
    A = A.drop_duplicates()
    print("A: ", A)
    print('columns: ', A.columns)
    A_sub = process.subtract_signals3(A)
    #sets input as rpd_charges and output as avgQPos
    X = A_sub.iloc[:,8:24]
    y = A_sub.iloc[:,0:2]

    #using state 42 for verification purposes
    train_X, train_y, val_X, val_y = train_test_split(X, y, - train_size, random_state = 42)
    print("Saving Data Set")
    train_X.to_pickle(filepath)
    train_y.to_pickle(filepath)
    val_X.to_pickle(filepath)
    val_y.to_pickle(filepath)

    model = get_model()
    model.summary()
    model.compile(optimizer = 'adam', loss = model_loss, metrics=['mae','mse'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights = True)

    history = model.fit(
        train_X, train_y,
        validation_data = (val_X, val_y),
        batch_size = 128,
        epochs = 250,
        callbacks=[early_stopping],
        verbose=1
    )

    model.load_weights(filepath)
    model.save(f'C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//linear_{model_num}_{model_loss}.h5')
    f = open(f'C://Users//Fre Shava Cado//Documents//VSCode Projects//SaveFiles//linear_{model_num}.txt', 'w')
    sys.stdout = f
    f.close()

    train_mse = history.history['mse']
    val_mse = history.history['val_mse']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    
    #Taken from train_cnn to compare vs cnn model
    epochs = range(1, len(train_mae) + 1)
    plt.figure(1)
    plt.plot(epochs, train_mse, 'o', color='black', label='Training set')
    plt.plot(epochs, val_mse, 'b', label='Validation set')
    plt.ylim([0.383,0.398])
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(f'Output/fig/model{model_num}_mse_{model_loss}Loss.png')

    plt.figure(2)
    plt.plot(epochs, train_mae, 'o', color='black', label='Training mae')
    plt.plot(epochs, val_mae, 'b', label='Validation mae')
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(f'Output/fig/model{model_num}_mae_{model_loss}Loss.png')


