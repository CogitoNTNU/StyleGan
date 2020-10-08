from keras.models import Model
from keras.optimizers import Adam
def Adverserial(generator,discriminator):
    discriminator.trainable=False
    input_gen = generator.input

    generator_output = generator(input_gen)
    gan_output = discriminator(generator_output)

    gan = Model(input_gen, gan_output)

    return gan