import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

def create_dnn(k):
    '''
    Creates a dnn to predict the Algebraic Multigrid parameters of a system
    stored in embedded sampled row form. The number of statistics computed on
    each row is denoted by k
    '''
    
    # Number of rows being sampled from the matrix
    nrows = 1601

    # Number of AMG parameters being used
    nparam = 12

    # Define the input shape from the passed number of statistics
    input_shape  = (1601, k, 1)

    # Define the model structure
    # Input layer
    vis   = Input(shape=input_shape)

    # Convolution/Pooling layers
    conv1 = Conv2D(32, kernel_size=(4, 2), activation='relu')(vis)
    pool1 = MaxPooling2D(pool_size=(8, 2))(conv1)
    conv2 = Conv2D(16, kernel_size=(4, 2), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(4, 2))(conv2)

    # Fully connected (dense) layers
    flat  = Flatten()(pool2)
    hid1  = Dense(100, activation='relu')(flat)
    hid2  = Dense(50, activation='relu')(hid1)
    hid3  = Dense(25, activation='relu')(hid2)

    # Output Layer
    out   = Dense(nparam, activation='sigmoid')(hid3)

    # Create and return the model
    model = Model(inputs=vis, outputs=out)

    return model

if __name__ == '__main__':

    model = create_dnn(10)
    print(model.summary())
