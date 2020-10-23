import models.silence
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

IMG_SIZE=64
GENERATOR_LEARNING_RATE=0.001 # Default 0.002 Apparently the latent FC mapping network has a 100x lower learning rate? (appendix B)
DISCRIMINATOR_LEARNING_RATE=0.001
BETA_1=0.9 # Exponential decay rate for first moment estimates, BETA_1=0 in the paper. Makes sense since the discriminator changes?
BETA_2=0.99
EPSILON=1e-8
BATCH_SIZE=16
NUM_BATCHES=10000
DATA_FOLDER=f"datasets/abstract/64"
SAVE_INTERVAL=500 

# Generator parameters
LATENT_DIM=16
CHANNELS=16
LATENT_STYLE_LAYERS=1

# Discriminator parameters
FILTERS=16

# Output folder
now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
OUTPUT_FOLDER = f"generated_images/{now_str}_{IMG_SIZE}"
os.mkdir(OUTPUT_FOLDER)

disc = models.discriminator.get_simple_discriminator(IMG_SIZE, filters=FILTERS)
print(disc.summary())
disc_optimizer=Adam(lr=DISCRIMINATOR_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
disc.compile(optimizer=disc_optimizer, loss="binary_crossentropy", metrics=['accuracy'])

gen = models.generator.get_skip_generator(latent_dim=LATENT_DIM, channels=CHANNELS, target_size=IMG_SIZE, latent_style_layers=LATENT_STYLE_LAYERS)
print(gen.summary())
adv = models.adverserial.get_adverserial(gen, disc)
adv_optimizer=Adam(lr=GENERATOR_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
adv.compile(optimizer=adv_optimizer, loss="binary_crossentropy", metrics=['accuracy'])

image_generator = data_tools.image_generator.image_generator(BATCH_SIZE, DATA_FOLDER)

discriminator_steps = 0
generator_steps = 0

for step in range(NUM_BATCHES):

    # Discriminator training 

    # Load a batch of real images and label them as zeroes
    real_images = next(image_generator)
    real_labels = np.zeros((BATCH_SIZE,1))

    # Generate a batch of images
    gen_input = random_generator_input(BATCH_SIZE, LATENT_DIM, IMG_SIZE)
    generated_images = gen.predict(gen_input)
    generated_labels = np.ones((BATCH_SIZE, 1))

    # Combine real and generated images
    combined_images = np.concatenate([generated_images, real_images])
    combined_labels = np.concatenate([generated_labels, real_labels])

    disc_loss = disc.train_on_batch([combined_images], combined_labels, return_dict=True)
    tf.keras.preprocessing.image.save_img(f"{OUTPUT_FOLDER}/{step}.png", generated_images[0])

    # Train generator to fool discriminator (discriminator should label generated images as real)
    target_labels = np.zeros((BATCH_SIZE,1))
    adv_loss = adv.train_on_batch(gen_input, target_labels, return_dict=True)
    print(f"step {step}, discriminator accuracy: {disc_loss['accuracy']:.2f}, generator accuracy: {adv_loss['accuracy']:.2f}")