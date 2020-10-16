import models.silence
import models.adverserial
import models.discriminator
import models.generator
from models.generator import random_generator_input
import data_tools.image_generator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import numpy as np
import math
import cv2
from datetime import datetime
import os

IMG_SIZE=512
LEARNING_RATE=0.0001
BATCH_SIZE=4
NUM_BATCHES=10000
DATA_FOLDER="datasets/abstract/512"
SAVE_INTERVAL = 500 

# Generator parameters
LATENT_DIM=64
CHANNELS=64
LATENT_STYLE_LAYERS=2

# Output folder
now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
OUTPUT_FOLDER = f"generated_images/{now_str}_{IMG_SIZE}"
os.mkdir(OUTPUT_FOLDER)

disc = models.discriminator.get_discriminator(IMG_SIZE)
print(disc.summary())
disc_optimizer=Adam(lr=LEARNING_RATE)
disc.compile(optimizer=disc_optimizer, loss="binary_crossentropy", metrics=['accuracy'])

gen = models.generator.get_generator(latent_dim=LATENT_DIM, channels=CHANNELS, target_size=IMG_SIZE, latent_style_layers=LATENT_STYLE_LAYERS)
print(gen.summary())
adv = models.adverserial.get_adverserial(gen, disc)
adv_optimizer=Adam(lr=LEARNING_RATE)
adv.compile(optimizer=adv_optimizer, loss="binary_crossentropy", metrics=['accuracy'])

image_generator = data_tools.image_generator.image_generator(BATCH_SIZE, DATA_FOLDER)

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

    discriminator_loss = disc.train_on_batch([combined_images], combined_labels)
    
    # Generator training

    # Train generator to fool discriminator (discriminator should label generated images as real)
    gen_input = random_generator_input(BATCH_SIZE, LATENT_DIM, IMG_SIZE) 
    adverserial_loss = adv.train_on_batch(gen_input, real_labels) 

    # Log and save results

    print("discriminator accuracy", discriminator_loss[1])
    print("generator fooling accuracy", adverserial_loss[1])

    # Store all generated images
    for i in range(BATCH_SIZE):
        tf.keras.preprocessing.image.save_img(f"{OUTPUT_FOLDER}/{step}_{i}.png", generated_images[i])

    #cv2.imshow("1", generated_images[0])
    #cv2.imwrite(f"generated_images/{step}.png", generated_images[0])

    # if not step%SAVE_INTERVAL:
    #     print("discriminator_loss",discriminator_loss)
    #     print("adverserial_loss",adverserial_loss)

    #     gen.save("temp.h5")
    #     adv.save("temp.h5")