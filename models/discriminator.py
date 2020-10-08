from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU


def get_discriminator(img_size):
    discriminator = Sequential()

    discriminator.add(Conv2D(16, kernel_size=1,
                             input_shape=(img_size, img_size, 3), padding='same'))
    discriminator.add(LeakyReLU(0.2))

    while(img_size > 4):
        if ((1024*16)/img_size >= 512):
            filter = 512
        else:
            filter = int((1024*16)/img_size)
        discriminator = box(discriminator, filter, (img_size, img_size, 3))
        img_size = img_size/2

    discriminator.add(Conv2D(512, kernel_size=3,
                             input_shape=(4, 4, 3), padding='same'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(512, kernel_size=4,
                             input_shape=(4, 4, 3), padding='same'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())

    discriminator.add(Dense(1, activation="sigmoid"))

    return discriminator


def box(discriminator, filter, img_size):
    discriminator.add(Conv2D(filter, kernel_size=3,
                             input_shape=img_size, padding='same'))
    discriminator.add(LeakyReLU(0.2))
    if (filter < 512):
        filter *= 2
    discriminator.add(Conv2D(filter, kernel_size=3,
                             input_shape=img_size, padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(AveragePooling2D(pool_size=(2, 2)))
    return discriminator