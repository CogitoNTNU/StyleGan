import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Add, LeakyReLU, Dropout
from tensorflow.keras import Input, Model

# def get_discriminator(img_size):
#     inp = Input(shape=(img_size,img_size,3))
#     discriminator = Conv2D(16, kernel_size=1, padding="same")(inp)
#     discriminator = LeakyReLU(0.2)(discriminator)


#     while img_size > 4:
#         if (1024 * 16) / img_size >= 512:
#             filter = 512
#         else:
#             filter = int((1024 * 16) / img_size)
#         discriminator = box(discriminator, filter, (img_size, img_size, 3))
#         img_size = img_size / 2


#     discriminator = Conv2D(512, kernel_size=3,input_shape=(4, 4, 3), padding="same")(discriminator)
#     discriminator = LeakyReLU(0.2)(discriminator)

#     discriminator = Conv2D(512, kernel_size=4,input_shape=(4, 4, 3), padding="same")(discriminator)
#     discriminator = LeakyReLU(0.2)(discriminator)

#     discriminator = Flatten()(discriminator)

#     discriminator = Dense(1, activation="sigmoid")(discriminator)

#     return discriminator

# def box(discriminator, filter, img_size):
#     # conv_part = Conv2D(filter, kernel_size=3,input_shape=img_size, padding="same")(discriminator)
#     # conv_part = LeakyReLU(0.2)(conv_part)
#     if (filter < 512):
#         filter *= 2
#     conv_part = Conv2D(filter,strides=2, kernel_size=3,input_shape=img_size, padding="same")(conv_part)
#     conv_part = LeakyReLU(0.2)(conv_part)
#     downsampling_part = AveragePooling2D(pool_size=(2, 2))(discriminator)
#     discriminator = Add()([conv_part, downsampling_part])
#     return discriminator


def get_resnet_discriminator(img_size, filters=16):

    z = Input(shape=(img_size, img_size, 3))

    # fRGB 
    x = Conv2D(filters, kernel_size=1, padding="same")(z)
    x = LeakyReLU(alpha=0.2)(x)

    size = img_size
    while size > 4:
        # Pass x through separate convolutional and downsampling blocks (resnet)

        # Convolution
        x1 = Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="random_normal")(x)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="random_normal")(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)

        # Downsampling
        x2 = AveragePooling2D(pool_size=(2, 2))(x)
        x2 = Conv2D(filters, kernel_size=1, kernel_initializer="random_normal", padding="same")(x2) # According to Fig. 7
        x2 = LeakyReLU(alpha=0.2)(x2)

        # Add back to a single image
        x = Add()([x1, x2])

        size = size//2

    x = Flatten()(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(4*4*filters)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(4*4*filters)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation="sigmoid")(x)

    discriminator = tf.keras.Model(inputs=z, outputs=x, name="discriminator")
    return discriminator


def get_simple_discriminator(img_size, filters=16):

    z = Input(shape=(img_size, img_size, 3))

    # fRGB 
    x = Conv2D(filters, kernel_size=1, padding="same")(z)
    x = LeakyReLU(alpha=0.2)(x)

    size = img_size
    while size > 4:
        # Simple version
        x = Conv2D(filters, kernel_size=3, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, kernel_size=3, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        size = size//2

    x = Flatten()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Dense(4*4*filters)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation="sigmoid")(x)

    discriminator = tf.keras.Model(inputs=z, outputs=x, name="discriminator")
    return discriminator


def get_discriminator(img_size):
    discriminator = Sequential()

    discriminator.add(Conv2D(16, kernel_size=1,
                             input_shape=(img_size, img_size, 3), padding="same"))
    discriminator.add(LeakyReLU(0.2))

    while(img_size > 4):
        if ((1024*16)/img_size >= 512):
            filter = 512
        else:
            filter = int((1024*16)/img_size)
        discriminator = box(discriminator, filter, (img_size, img_size, 3))
        img_size = img_size/2

    discriminator.add(Conv2D(512, kernel_size=3,
                             input_shape=(4, 4, 3), padding="same"))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(512, kernel_size=4,
                             input_shape=(4, 4, 3), padding="same"))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())

    discriminator.add(Dense(1, activation="sigmoid"))

    return discriminator


def box(discriminator, filter, img_size):
    discriminator.add(Conv2D(filter, kernel_size=3,
                             input_shape=img_size, padding="same"))
    discriminator.add(LeakyReLU(0.2))
    if (filter < 512):
        filter *= 2
    discriminator.add(Conv2D(filter, kernel_size=3,
                             input_shape=img_size, padding="same"))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(AveragePooling2D(pool_size=(2, 2)))
    return discriminator
