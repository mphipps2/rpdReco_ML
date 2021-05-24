def get_cnn():
# define two sets of inputs
        nNeutronInput = Input(shape=(1,), name = 'neutron_branch')
        # Keras functional API that passes tensors between layers -- allows for multiple input/output branches
        x = Dense(16, activation="relu")(nNeutronInput)
        x = Dense(4, activation="relu")(x)
        x = Dense(1,activation="linear")(x)
        x = Model(inputs=nNeutronInput, outputs=x)

        #input: 6,6 to include padding
        signalInput = Input(shape=(6,6,1), name = 'rpd_branch')
        # may not want to use drop out if we're using batch normalization
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
        # combine the output of the two branches
        combined = keras.layers.concatenate([x.output, y.output])
        combined = Dense(8, activation="relu", name = 'combined')(combined)
        combined = Dense(8, activation="relu")(combined)
        Q_avg = Dense(2, activation="tanh", name = 'Q_avg')(combined)
        # apply a FC layer and then a regression prediction on the
        # combined outputs
        # relu: F(x) = max(0,x)
        # Dense implements operation: output = activation(dot(input,kernel)+bias)
        # why 2 neurons and why tanh activation function?
        # linear or relu more common with single neuron for linear regression. Try leaky relu
        #        Q_avg = Dense(2, activation="tanh", name = 'Q_avg')(Q_avg)
        #        Q_avg = Dense(2, activation="linear", name = 'Q_avg')(Q_avg)
        model = Model(inputs=[x.input, y.input], outputs=[Q_avg])
        return model
