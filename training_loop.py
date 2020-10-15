import logging
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import models.adverserial
import models.discriminator
import models.generator
#import models.naive_generator
import data_tools.image_generator
from tensorflow.keras.optimizers import Adam

import numpy as np
import math
import cv2

IMG_SIZE=64
LEARNING_RATE=0.0001
BATCH_SIZE=16
NUM_BATCHES=10000
LATENT_DIM=512
DATA_FOLDER="datasets/"

SAVE_INTERVAL = 500 #In batches
gen = models.generator.SimpleGenerator(LATENT_DIM)
print(gen.summary())
disc = models.discriminator.get_discriminator(IMG_SIZE)
print(disc.summary())

disc_optimizer=Adam(lr=LEARNING_RATE)
disc.compile(optimizer=disc_optimizer,loss="binary_crossentropy", metrics=['accuracy'])

adv = models.adverserial.get_adverserial(gen, disc)

adv_optimizer=Adam(lr=LEARNING_RATE)
adv.compile(optimizer=adv_optimizer,loss="binary_crossentropy", metrics=['accuracy'])

image_generator = data_tools.image_generator.image_generator(BATCH_SIZE,DATA_FOLDER)

for step in range(1,NUM_BATCHES+1):
    print(f"Training step {step}")
    real_images = next(image_generator)
    real_labels = np.zeros((BATCH_SIZE,1))

    z_noise = np.random.normal(size=(BATCH_SIZE,LATENT_DIM))
    null_input = np.zeros((BATCH_SIZE, 1))
    noises = []
    num_upsamles = int(math.log2(IMG_SIZE)-2)
    noises.append(np.random.normal(size=(BATCH_SIZE, 4, 4, 1)))
    for i in range(1,num_upsamles):
        noises.append(np.random.normal(size=(BATCH_SIZE,4*(2**i),4*(2**i),1)))
        noises.append(np.random.normal(size=(BATCH_SIZE,4*(2**(i)),4*(2**(i)),1)))
    noises.append(np.random.normal(size=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1)))

    generated_images = gen.predict([null_input,z_noise,*noises])
    generated_labels = np.ones((BATCH_SIZE,1))

    combined_images=np.concatenate([generated_images,real_images])
    combined_labels = np.concatenate([generated_labels,real_labels])

    discriminator_loss = disc.train_on_batch([combined_images],combined_labels)

    random_latent_vectors=np.random.normal(size=(BATCH_SIZE,LATENT_DIM))

    adverserial_loss = adv.train_on_batch([null_input,random_latent_vectors,*noises],real_labels)


    cv2.imshow("1", generated_images[0])
    cv2.waitKey(0)
    if not step%SAVE_INTERVAL:
        print("discriminator_loss",discriminator_loss)
        print("adverserial_loss",adverserial_loss)

        gen.save("temp.h5")
        adv.save("temp.h5")