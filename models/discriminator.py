import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Add, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras import Input, Model


def get_resnet_discriminator(img_size=(64, 64), filters=16, dense_units=64):

    img_width = img_size[1]
    img_height = img_size[0]

    z = Input(shape=(img_height, img_width, 3))

    # fRGB 
    x = Conv2D(filters, kernel_size=1, padding="same")(z)
    x = BatchNormalization()(x) # BatchNorm can also be added after activation 
    x = LeakyReLU(alpha=0.2)(x)

    size = min(img_width, img_height)
    while size > 4:
        # Pass x through separate convolutional and downsampling blocks (resnet)

        # Convolution
        x1 = Dropout(rate=0.2)(x)
        x1 = Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="random_normal")(x1)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        
        x1 = Dropout(rate=0.2)(x)
        x1 = Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="random_normal")(x1)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)

        # Downsampling
        x2 = AveragePooling2D(pool_size=(2, 2))(x)
        x2 = Conv2D(filters, kernel_size=1, kernel_initializer="random_normal", padding="same")(x2) # According to Fig. 7
        x1 = BatchNormalization()(x1)
        x2 = LeakyReLU(alpha=0.2)(x2)

        # Add back to a single image
        x = Add()([x1, x2])
        x = BatchNormalization()(x)
        size = size//2

    x = Flatten()(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation="sigmoid")(x)

    discriminator = tf.keras.Model(inputs=z, outputs=x, name="discriminator")
    return discriminator
