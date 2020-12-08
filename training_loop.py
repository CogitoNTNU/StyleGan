import models.silence
import models.adverserial
import models.discriminator
import models.generator
from models.generator import random_generator_input
import data_tools.image_generator
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from config import IMG_SIZE, CONTINUE_TRAINING, MODEL_TRAIN_GEN_WEIGHTS, MODEL_TRAIN_DISC_WEIGHTS, \
    GENERATOR_LEARNING_RATE, DISCRIMINATOR_LEARNING_RATE, BETA_1, BETA_2, EPSILON, BATCH_SIZE, NUM_BATCHES, \
    DATA_FOLDER, SAVE_INTERVAL, LATENT_DIM, CHANNELS, LATENT_STYLE_LAYERS, FILTERS
import numpy as np
import math
import cv2
from datetime import datetime
import os
import time

# Output folder
now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = f"generated_images/{now_str}_{IMG_SIZE}"
os.makedirs("generated_images", exist_ok=True)
os.mkdir(OUTPUT_FOLDER)
os.makedirs("weights",exist_ok=True)

disc = models.discriminator.get_resnet_discriminator(IMG_SIZE, filters=FILTERS)
print(disc.summary())
disc_optimizer = Adam(
    lr=DISCRIMINATOR_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON
)
disc.compile(optimizer=disc_optimizer, loss="binary_crossentropy", metrics=["accuracy"])

gen = models.generator.get_skip_generator(
    latent_dim=LATENT_DIM,
    channels=CHANNELS,
    target_size=IMG_SIZE,
    latent_style_layers=LATENT_STYLE_LAYERS
)

print(gen.summary())
adv = models.adverserial.get_adverserial(gen, disc)
adv_optimizer = Adam(
    lr=GENERATOR_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON
)
adv.compile(optimizer=adv_optimizer, loss="binary_crossentropy", metrics=["accuracy"])


if CONTINUE_TRAINING:
    gen.load_weights(MODEL_TRAIN_GEN_WEIGHTS)
    disc.load_weights(MODEL_TRAIN_DISC_WEIGHTS)


image_generator = data_tools.image_generator.image_generator(BATCH_SIZE, DATA_FOLDER)

for step in range(NUM_BATCHES):

    # Discriminator training

    # Load a batch of real images and label them as zeroes
    real_images = next(image_generator)
    real_labels = np.zeros((BATCH_SIZE, 1))

    # Generate a batch of images
    gen_input = random_generator_input(BATCH_SIZE, LATENT_DIM, IMG_SIZE)
    generated_images = gen.predict(gen_input)
    generated_labels = np.ones((BATCH_SIZE, 1))

    # Store the best and worst generated image according to the discriminator
    if step % SAVE_INTERVAL == 0:
        disc_labels = disc.predict_on_batch(generated_images).flatten()
        i_max = np.argmax(disc_labels)
        i_min = np.argmin(disc_labels)
        tf.keras.preprocessing.image.save_img(
            f"{OUTPUT_FOLDER}/{step}_{disc_labels[i_max]:.2f}.png",
            generated_images[i_max],
        )
        tf.keras.preprocessing.image.save_img(
            f"{OUTPUT_FOLDER}/{step}_{disc_labels[i_min]:.2f}.png",
            generated_images[i_min],
        )
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        gen.save_weights(os.path.join("weights", "gen_" + now_str+".h5"))
        disc.save_weights(os.path.join("weights", "disc_" + now_str+".h5"))

    # Combine real and generated images
    combined_images = np.concatenate([generated_images, real_images])
    combined_labels = np.concatenate([generated_labels, real_labels])

    disc_loss = disc.train_on_batch(
        [combined_images], combined_labels, return_dict=True
    )

    # Train generator to fool discriminator (discriminator should label generated images as real)
    target_labels = np.zeros((BATCH_SIZE, 1))
    adv_loss = adv.train_on_batch(gen_input, target_labels, return_dict=True)
    print(
        f"step {step}, discriminator accuracy: {disc_loss['accuracy']:.2f}, generator accuracy: {adv_loss['accuracy']:.2f}"
    )
