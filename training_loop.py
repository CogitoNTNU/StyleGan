#import models.silence
import models.adverserial
import models.discriminator
import models.generator
from models.generator import random_generator_input
import data_tools.image_generator
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf

import numpy as np
import math
import cv2
from datetime import datetime
import os
import time

# Keras prefers height x width
START_SIZE = (4, 4)
TARGET_SIZE = (512, 512) 
#START_SIZE = (6, 4)
#TARGET_SIZE = (96, 64) 
#TARGET_SIZE = (192, 128) 
GENERATOR_LEARNING_RATE = 0.00025  # Default 0.002 Apparently the latent FC mapping network has a 100x lower learning rate? (appendix B)
DISCRIMINATOR_LEARNING_RATE = 0.0005
BETA_1 = 0.0  # Exponential decay rate for first moment estimates, BETA_1=0 in the paper. Makes sense since the discriminator changes?
BETA_2 = 0.99
EPSILON = 1e-8
BATCH_SIZE = 1
NUM_BATCHES = 10000000
DATA_FOLDER = "datasets/abstract/512"
#DATA_FOLDER = "datasets/plants/96x64"
#DATA_FOLDER = "datasets/plants/192x128"
SAVE_INTERVAL = 10
MODEL_SAVE_INTERVAL = 5000
#LEARNNG_RATE_INTERVAL = 500

# Generator parameters
LATENT_DIM = 128
CHANNELS = 128
LATENT_STYLE_LAYERS = 8

# Discriminator parameters
FILTERS = 128
DENSE_UNITS = 128

# Output folder
now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
OUTPUT_FOLDER = f"generated_images/{now_str}_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
os.mkdir(OUTPUT_FOLDER)
MODEL_FOLDER = f"trained_models/{now_str}_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
os.mkdir(MODEL_FOLDER)


disc = models.discriminator.get_resnet_discriminator(img_size=TARGET_SIZE, filters=FILTERS, dense_units=DENSE_UNITS)
print(disc.summary())
disc_optimizer = Adam(
    lr=DISCRIMINATOR_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON
)
disc.compile(optimizer=disc_optimizer, loss="binary_crossentropy", metrics=["accuracy"])

gen = models.generator.get_skip_generator(
    start_size=START_SIZE,
    target_size=TARGET_SIZE,
    latent_dim=LATENT_DIM,
    channels=CHANNELS,
    latent_style_layers=LATENT_STYLE_LAYERS,
)
print(gen.summary())
adv = models.adverserial.get_adverserial(gen, disc)
adv_optimizer = Adam(
    lr=GENERATOR_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON
)
adv.compile(optimizer=adv_optimizer, loss="binary_crossentropy", metrics=["accuracy"])

image_generator = data_tools.image_generator.image_generator(BATCH_SIZE, DATA_FOLDER)

#dataset = tf.keras.preprocessing.image_dataset_from_directory("datasets/keras_abstract/", label_mode=None, batch_size=BATCH_SIZE, image_size=(512, 512))
#dataset = dataset.map(lambda x: x/127.5 -1.0).prefetch(8)

for step in range(NUM_BATCHES):
    
    # Load a batch of real images and label them as zeroes
    real_images = next(image_generator)
    real_labels = np.zeros((BATCH_SIZE, 1))

    # Generate a batch of images
    gen_input = random_generator_input(BATCH_SIZE, LATENT_DIM, start_size=START_SIZE, target_size=TARGET_SIZE)
    generated_images = gen.predict(gen_input)
    generated_labels = np.ones((BATCH_SIZE, 1))

    # Combine real and generated images
    combined_images = np.concatenate([generated_images, real_images])
    combined_labels = np.concatenate([generated_labels, real_labels])
    
    # Train discriminator
    disc_loss = disc.train_on_batch([combined_images], combined_labels, return_dict=True)

    # Train generator to fool discriminator (discriminator should label generated images as real)
    gen_input = random_generator_input(BATCH_SIZE, LATENT_DIM, start_size=START_SIZE, target_size=TARGET_SIZE) # Use new random input for the generator
    target_labels = np.zeros((BATCH_SIZE, 1))
    adv_loss = adv.train_on_batch(gen_input, target_labels, return_dict=True)
    print(f"step {step}, discriminator loss: {disc_loss['loss']:.2f}, generator loss: {adv_loss['loss']:.2f}")

    # Store the batch of generated images
    if step % SAVE_INTERVAL == 0:
        disc_labels = disc.predict_on_batch(generated_images).flatten()
        for i in range(BATCH_SIZE):
            tf.keras.preprocessing.image.save_img(f"{OUTPUT_FOLDER}/{step}_{i}_{disc_labels[i]:.2f}.png", generated_images[i])

    # Store the generator and discriminator
    if step % MODEL_SAVE_INTERVAL == 0:
        gen.save(f"{MODEL_FOLDER}/gen_{step}.h5")
        disc.save(f"{MODEL_FOLDER}/disc_{step}.h5")

    # Pick a new random learning rate
    # if step % LEARNNG_RATE_INTERVAL == 0:
    #     lr = 10.0**(-np.random.uniform(3, 6))
    #     disc_optimizer.learning_rate.assign(lr)
    #     adv_optimizer.learning_rate.assign(lr)
    #     print(f"Learning rate: {lr:.7f}")