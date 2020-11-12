# Training loop that loads images into tensors using prefetching etc (tf.data.dataset..)
# Manual GAN training loop from tutorial https://www.tensorflow.org/tutorials/generative/dcgan

import tensorflow as tf 
import models.generator
import models.discriminator
import time
import math
from datetime import datetime
import os

START_SIZE = (4, 4)
TARGET_SIZE = (512, 512) 

MODEL_SAVE_INTERVAL = 1000
SAVE_INTERVAL = 10
EPOCHS = 1000

# Generator parameters
LATENT_DIM = 16
CHANNELS = 32
LATENT_STYLE_LAYERS = 2

# Discriminator parameters
FILTERS = 32
DENSE_UNITS = 16
BATCH_SIZE = 4

now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
OUTPUT_FOLDER = f"generated_images/{now_str}_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
os.mkdir(OUTPUT_FOLDER)
MODEL_FOLDER = f"trained_models/{now_str}_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
os.mkdir(MODEL_FOLDER)

# Optimizer
DISCRIMINATOR_LEARNING_RATE = 0.0005
GENERATOR_LEARNING_RATE = 0.0005
BETA_1 = 0.0 
BETA_2 = 0.99
EPSILON = 0.00001

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=GENERATOR_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIMINATOR_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

discriminator = models.discriminator.get_resnet_discriminator(
    img_size=TARGET_SIZE, 
    filters=FILTERS, 
    dense_units=DENSE_UNITS
)
print(discriminator.summary())
#discriminator = tf.keras.models.load_model("trained_models/artist/discriminator_30000.h5")
print(discriminator.summary())
generator = models.generator.get_skip_generator(
    start_size=START_SIZE,
    target_size=TARGET_SIZE,
    latent_dim=LATENT_DIM,
    channels=CHANNELS,
    latent_style_layers=LATENT_STYLE_LAYERS,
)
#generator = tf.keras.models.load_model("trained_models/artist/generator_30000.h5")
print(generator.summary())

dataset = tf.keras.preprocessing.image_dataset_from_directory("datasets/keras_abstract/", label_mode=None, batch_size=BATCH_SIZE, image_size=TARGET_SIZE)
dataset = dataset.map(lambda x: x/127.5 -1.0).prefetch(16)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def tf_random_generator_input(batch_size, latent_dim, start_size=(4,4), target_size=(64,64)):
    dummy_input = tf.zeros(shape=(batch_size, 1))
    latent_noise = tf.random.normal(shape=(batch_size, latent_dim))
    noises = []

    height = start_size[0]
    width = start_size[1]

    num_upsamples = int(math.log2(target_size[1]/start_size[1]))
    noises.append(tf.random.normal(shape=(batch_size, height, width, 1)))
    for i in range(1, num_upsamples):
        height = 2*height
        width = 2*width
        noises.append(tf.random.normal(shape=(batch_size, height, width, 1)))
        noises.append(tf.random.normal(shape=(batch_size, height, width, 1)))
    
    noises.append(tf.random.normal(shape=(batch_size, target_size[0], target_size[1], 1)))
    return [dummy_input, latent_noise, *noises]


@tf.function
def train_step(images):
    noise = tf_random_generator_input(BATCH_SIZE, LATENT_DIM, start_size=START_SIZE, target_size=TARGET_SIZE)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        tf.print("real_output", real_output)
        tf.print("fake_output", fake_output)

        min_clip = 0.001
        max_clip = 0.999

        real_output = tf.clip_by_value(real_output, min_clip, max_clip)
        fake_output = tf.clip_by_value(fake_output, min_clip, max_clip)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        tf.print("gen_loss", gen_loss)
        tf.print("disc_loss", disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


step = 0
for epoch in range(EPOCHS):
    for image_batch in dataset:
        print(step)
        train_step(image_batch)
        
        if(step % SAVE_INTERVAL == 0):
            noise = tf_random_generator_input(BATCH_SIZE, LATENT_DIM, start_size=START_SIZE, target_size=TARGET_SIZE)
            generated_images = generator(noise, training=False)
            for i in range(BATCH_SIZE):
                tf.keras.preprocessing.image.save_img(f"{OUTPUT_FOLDER}/{step}_{i}.png", generated_images[i])

        if(step % MODEL_SAVE_INTERVAL == 0):
            generator.save(f"{MODEL_FOLDER}/generator_{step}.h5")
            discriminator.save(f"{MODEL_FOLDER}/discriminator_{step}.h5")

        step = step + 1
