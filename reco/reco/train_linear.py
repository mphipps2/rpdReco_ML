import os
import sys

sys.path.append('/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/rpdreco/reco')


import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
#import reco.lib.vis as vis
import reco.lib.io as io

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt

def get_linear_model(normalizer):
    #linear model, input layer -> normalized -> 2 positional coordinates
    inputs = keras.Input(shape = (16,))
    normed = normalizer(inputs)
    prediction = layers.Dense(2, activation = 'linear')(normed)

    model = keras.models.Model(inputs=inputs, outputs = prediction)

    return model

def get_fc_model(normalizer):
    #fully connected model, input -> normalized -> 64 node hidden layer -> 64 node hidden layer -> 2 positional coordinates
    inputs = keras.Input(shape = (16,))
    #normed = normalizer(inputs)
    out = layers.Dense(64)(inputs)#normed)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(64)(out)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dropout(0.3)(out)
    prediction = layers.Dense(2, activation = 'linear')(out)

    model = keras.models.Model(inputs=inputs, outputs = prediction)

    return model

def get_normalizer(data):
    normalizer = Normalization(axis = None)
    normalizer.adapt(data)
    return normalizer

def findCOM (rpd):
	#print(rpd)
	com = pd.DataFrame(0, index = rpd.index, columns = ['comX', 'comY'])
	#input()
	totalSignal = rpd.sum(axis = 1)
	#input()
	for ch in range(len(rpd.columns)):
		x = 0
		y = 0
		if ch < 4: y = 1.5
		elif (ch >= 4 and ch < 8): y = 0.5
		elif (ch >= 8 and ch < 12): y = -0.5
		else: y = -1.5

		if ch%4 == 0: x = 1.5
		elif ch%4 == 1: x = 0.5
		elif ch%4 == 2: x = -0.5
		elif ch%4 == 3: x = -1.5

		com.comX += x*rpd.iloc[:,ch]
		com.comY += y*rpd.iloc[:,ch]
	com.comX/=totalSignal
	com.comY/=totalSignal
	return com

def getCOMReactionPlane(com, centerX, centerY):
	com.comY = com.comY-centerY
	com.comX = com.comX-centerX
	phi = np.arctan2(com.comY, com.comX)
	return phi

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
    return data

def directCOMComparison():
    centerX = 0
    centerY = -0.471659
    model_num = 1
    model_loss = 'CoM'
    filepath = f"/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/models/model_{model_num}_{model_loss}/"

    A = io.get_dataset(folder = "/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/data/", side = 'A')
    A = A.drop_duplicates()

    rpdSignals = A.iloc[:,8:24]
    com = findCOM(rpdSignals)

    print(com)
    #input()
    os.mkdir(filepath)

    comPhi = getCOMReactionPlane(com, centerX, centerY)
    A['ComPhi'] = comPhi
    someA, tmpA = train_test_split(A, train_size = 0.8, random_state = 42)
    someA2, testA = train_test_split(tmpA, train_size = 0.5, random_state = 42)
    testA.to_pickle(filepath + 'test_A.pickle')


def train_linear():
    train_size = 0.8
    model_num = 28
    model_loss = 'mse'
    filepath = f"/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/models/model_{model_num}_{model_loss}/"
    random_state = 42

    print("Getting Dataset...")

    A = io.get_dataset(folder = "/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/data/", side = 'A')
    A = A.drop_duplicates()
    print("A: ", A)
    print('columns: ', A.columns)
    #A = pd.concat([A,findCOM(A.iloc[:,8:24])],axis = 1)
    #A = get_averages(A)
    #using state 42 for verification purposes
    train_A, tmpA = train_test_split(A, test_size= 1.-train_size, random_state = random_state)
    val_A, test_A = train_test_split(tmpA, train_size = 0.5, random_state = random_state)

    print("Saving Data Set")
    os.mkdir(filepath)
    test_A.to_pickle(filepath + 'test_A.pickle')
    '''
    test_X.to_pickle(filepath + 'test_X.pickle')
    test_y.to_pickle(filepath + 'test_y.pickle')
    test_psi.to_pickle(filepath + 'psi_truth.pickle')
    '''
    train_X = train_A.iloc[:,8:24]
    train_y = train_A.iloc[:,0:2]
    val_X = val_A.iloc[:, 8:24]
    val_y = val_A.iloc[:,0:2]
    
    #sanity check
    print('Training Data:\n')
    print(train_X.head())
    print(train_y.head())
    print('Validation Data:\n')
    print(val_X.head())
    print(val_y.head())

    normalizer = get_normalizer(train_X)
    model = get_fc_model(normalizer)
    print("Model Received.")
    model.summary()
    model.compile(optimizer = 'adam', loss = model_loss, metrics=['mae','mse','msle'])
    early_stopping = keras.callbacks.EarlyStopping(min_delta = 0.05, patience = 15, monitor='val_loss', restore_best_weights = True)

    print("Starting training:")
    history = model.fit(
        train_X, train_y,
        validation_data = (val_X, val_y),
        batch_size = 256,
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

    f = open(filepath + f'linear_{model_num}.txt', 'w')
    f.write('Difference: Batchnorm after activation with dropout, no normalizer')
    f.write('\nval_loss:' + str(np.min(val_mse)))
    weights = model.layers[-1].get_weights()
    f.write('\n' + str(weights))
    f.close()

    f = open(filepath + f'linear_{model_num}_{model_loss}_summary.txt', 'w')
    model.summary(print_fn = lambda x: f.write(x+'\n'))
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
    plt.savefig(filepath + f'model{model_num}_mse_{model_loss}Loss.png')

    plt.figure(1)
    plt.plot(epochs, train_mae, color='black', label='Training mae')
    plt.plot(epochs, val_mae, 'b', label='Validation mae')
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(filepath + f'model{model_num}_mae_{model_loss}Loss.png')
    '''
    plt.figure(3)
    plt.plot(epochs, train_msle, 'o', color='black', label='Training msle')
    plt.plot(epochs, val_msle, 'b', label='Validation msle')
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Logarithmic Error')
    plt.legend()
    plt.savefig(filepath + f'model{model_num}_msle_{model_loss}Loss.png')
    '''
    print('loss: ', np.min(train_mse))
    print('val loss:', np.min(val_mse))


