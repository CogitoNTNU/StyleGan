from keras.models import Model
from keras.backend import constant
from keras.layers import Input
from keras.layers import Conv2D, Reshape,Concatenate,Dense, Lambda, Conv1D,UpSampling2D,LeakyReLU, Add
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.layers import Layer
batch_size,start_res,end_res,latent_size=16,4,32,512
import cv2
from keras.preprocessing import image
import os

filters = 256

def AdaIN(x):
    mean = K.mean(x[0], axis=[1, 2], keepdims=True)
    std = K.std(x[0], axis=[1, 2], keepdims=True) + 1e-7
    y = (x[0] - mean) / std

    # Reshape scale and bias parameters
    pool_shape = [-1, 1, 1, y.shape[-1]]
    scale = K.reshape(x[1], pool_shape)
    bias = K.reshape(x[2], pool_shape)

    return y * scale + bias


def generate_learned_constants(inp):
    out = Dense(1)(inp)
    out = Lambda(lambda x: x*0+1)(out)
    out = Dense(4*4*filters)(out)
    out = Reshape((4,4,filters))(out)
    return out

def generate_block(lat_in,image_in):
    a = Dense(filters, activation="relu")(lat_in)
    b = Dense(filters, activation="relu")(lat_in)
    image_in = Conv2D(filters,3,padding="same")(image_in)
    image_in = LeakyReLU()(image_in)
    out = Lambda(AdaIN)([image_in, a,b])
    a = Dense(filters, activation="relu")(lat_in)
    b = Dense(filters, activation="relu")(lat_in)
    out = Conv2D(filters, 3, padding="same")(out)
    out = LeakyReLU()(out)
    out = Lambda(AdaIN)([out, a,b])
    return out

def noise_input(dim):
    return Input((dim,dim,1))
def generate_noise(noise_in,channels):
    channel_noise = Conv2D(channels,1)(noise_in)

    return channel_noise

def generate_model(lat_in,noises,upsizes):
    constants = generate_learned_constants(lat_in)
    out = constants
    ni1 = noises[0]
    n1 = generate_noise(ni1, filters)
    out = Add()([out, n1])
    out = generate_block(lat_in, out)
    for i in range(upsizes):
        out = UpSampling2D(size=(2, 2), interpolation='nearest')(out)
        ni1 = noises[i+1]
        n1 = generate_noise(ni1,filters)
        out = Add()([out,n1])
        out = generate_block(lat_in,out)
    out = Conv2D(3, 3, activation="tanh",padding="same")(out)
    print(out.shape)
    return out

def generate_latent_input():
    return Input((latent_size,))

def Generator(upsizes = 4):
    latent_in = generate_latent_input()
    noises = []
    for i in range(upsizes+1):
        noises.append(noise_input(4*2**(i)))
    model = generate_model(latent_in,noises,upsizes)

    m = Model([latent_in,*noises],model)
    return m