from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def get_adverserial(generator, discriminator):
    discriminator.trainable = False
    input_gen = generator.input
    generator_output = generator(input_gen)
    gan_output = discriminator(generator_output)
    gan = Model(input_gen, gan_output)
    return gan
