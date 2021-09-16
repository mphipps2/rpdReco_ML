import tensorflow.keras as keras
from tensorflow.keras import layers

# Note: test functions are for testing new model architecture. Keep best/stable model in non-test function

def get_linear():
    
    inputs = keras.Input(shape = (16,))
    prediction = layers.Dense(2, activation = 'linear')(inputs)
    model = keras.models.Model(inputs=inputs, outputs = prediction)

    return model


def get_fcn():

    inputs = keras.Input(shape = (16,))

    out = layers.Dense(256)(inputs)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)

    out = layers.Dense(256)(out)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)

    out = layers.Dense(256)(out)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)
    
    out = layers.Dense(256)(out)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)

    prediction = layers.Dense(2, activation = 'linear')(out)

    model = keras.models.Model(inputs=inputs, outputs = prediction)

    return model

def get_cnn():
    nNeutronInput = keras.Input(shape=(1,), name = 'neutron_branch')
    x = layers.Dense(16, activation="relu")(nNeutronInput)
    x = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(1,activation="linear")(x)
    x = keras.models.Model(inputs=nNeutronInput, outputs=x)

    #input: (6,6), includes padding
    signalInput = keras.Input(shape=(6,6,1), name = 'rpd_branch')
    y = layers.Conv2D(filters = 16, kernel_size = (1,1), padding = 'Same', activation ='relu')(signalInput)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(filters = 32, kernel_size = (2,2), padding = 'Same', activation ='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(filters =64, kernel_size = (3,3), padding = 'Same', activation ='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Flatten()(y)
    
    y = layers.Dense(32, activation = "relu")(y)
    y = layers.Dense(16, activation = "relu")(y)
    y = layers.Dense(8, activation = "relu")(y)
    y = keras.models.Model(inputs=signalInput, outputs=y)
    
    combined = keras.layers.concatenate([x.output, y.output])
    combined = layers.Dense(8, activation="relu", name = 'combined')(combined)
    combined = layers.Dense(8, activation="relu")(combined)
    Q_avg = layers.Dense(2, activation="tanh", name = 'Q_avg')(combined)
    
    model = keras.models.Model(inputs=[x.input, y.input], outputs=[Q_avg])
    return model


def get_linear_test():
    inputs = keras.Input(shape = (16,))
    prediction = layers.Dense(2, activation = 'linear')(inputs)
    model = keras.models.Model(inputs=inputs, outputs = prediction)

    return model


def get_fcn_test():
    inputs = keras.Input(shape = (16,))
    #out = layers.BatchNormalization()(inputs)

    out = layers.Dense(256)(inputs)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)
    #out = layers.Dropout(0.4)(out)

    out = layers.Dense(256)(out)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)
    #out = layers.Dropout(0.4)(out)

    out = layers.Dense(256)(out)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)
    #out = layers.Dropout(0.4)(out)
    
    out = layers.Dense(256)(out)
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization()(out)
    #out = layers.Dropout(0.4)(out)

    prediction = layers.Dense(2, activation = 'linear')(out)

    model = keras.models.Model(inputs=inputs, outputs = prediction)

    return model

def get_bdt():
    return model

def get_cnn_test():
#    nNeutronInput = keras.Input(shape=(1,), name = 'neutron_branch')
#    x = layers.Dense(16, activation="relu")(nNeutronInput)
#    x = layers.Dense(4, activation="relu")(x)
#    x = layers.Dense(1,activation="linear")(x)
#    x = keras.models.Model(inputs=nNeutronInput, outputs=x)

    #input: (6,6), includes padding
    signalInput = keras.Input(shape=(6,6,1), name = 'Subtracted RPD')
    y = layers.Conv2D(filters = 32, kernel_size = (2,2), padding = 'Same', activation ='relu')(signalInput)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(filters = 32, kernel_size = (2,2), padding = 'Same', activation ='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Flatten()(y)
    
#    y = layers.Dense(32, activation = "relu")(y)
    y = layers.Dense(16, activation = "relu")(y)
    y = layers.Dense(8, activation = "relu")(y)
    Q_avg = layers.Dense(2, activation="tanh", name = 'Q_avg')(y)
    
    #    y = keras.models.Model(inputs=signalInput, outputs=y)    
#    combined = keras.layers.concatenate([x.output, y.output])
#    combined = layers.Dense(8, activation="relu", name = 'combined')(combined)
#    combined = layers.Dense(8, activation="relu")(combined)

    model = keras.models.Model(inputs=signalInput, outputs=[Q_avg])
    return model

def get_bdt_test():
    return model
