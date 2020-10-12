import models.adverserial
import models.discriminator
import models.generator
import data_tools.image_generator
from keras.optimizers import Adam
import numpy as np


IMG_SIZE=64
LEARNING_RATE=0.0001
BATCH_SIZE=64
NUM_BATCHES=10000
LATENT_DIM=512
DATA_FOLDER="datasets/dataset1"

SAVE_INTERVAL = 500 #In batches
gen = models.generator.get_generator(IMG_SIZE)
disc = models.discriminator.get_discriminator(IMG_SIZE)

disc_optimizer=Adam(lr=LEARNING_RATE)
disc.compile(optimizer=disc_optimizer,loss="binary_crossentropy", metrics=['accuracy'])
disc.compile()

adv = models.adverserial.get_adverserial(gen,disc)

adv_optimizer=Adam(lr=LEARNING_RATE)
adv.compile(optimizer=adv_optimizer,loss="binary_crossentropy", metrics=['accuracy'])

image_generator = data_tools.image_generator.image_generator(BATCH_SIZE,DATA_FOLDER)

for step in range(1,NUM_BATCHES+1):
    real_images = next(image_generator)
    real_labels = np.zeros(BATCH_SIZE,1)

    z_noise = np.random.normal((BATCH_SIZE,LATENT_DIM))

    generated_images = gen.predict([z_noise])
    generated_labels = np.ones(BATCH_SIZE,1)

    combined_images=np.concatenate([generated_images,real_images])
    combined_labels = np.concatenate([generated_labels,real_labels])

    discriminator_loss = disc.train_on_batch([combined_images],combined_labels)

    random_latent_vectors=np.random.normal(size=(BATCH_SIZE,LATENT_DIM))

    adverserial_loss = adv.train_on_batch(random_latent_vectors,real_labels)

    if not step%SAVE_INTERVAL:
        print("discriminator_loss",discriminator_loss)
        print("adverserial_loss",adverserial_loss)

        gen.save("temp.h5")
        adv.save("temp.h5")