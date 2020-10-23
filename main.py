import models.silence
import models.generator
import tensorflow as tf
from models.generator import random_generator_input
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from datetime import datetime
import math

IMG_SIZE=64
LATENT_DIM=4
CHANNELS=4
LATENT_STYLE_LAYERS=2
BATCH_SIZE=8
TRAIN_STEPS=10000
SAVE_STEPS=100

now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
OUTPUT_FOLDER = f"generated_images/{now_str}_{IMG_SIZE}"
os.mkdir(OUTPUT_FOLDER)

# Train the generator to generate a single image

# Green target
# target_images = -1*np.ones(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
# target_images[:,:,:,1] = 1

# Random target
# target_images = 2*np.random.random(size=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)) - 1.0

# Half red, half green
target_images = -1*np.ones(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
target_images[:,:IMG_SIZE//2,:,0] = 1
target_images[:,IMG_SIZE//2:,:,1] = 1

# More complex
target_images[:,:,IMG_SIZE//2:,2] = 1

# Save the target
tf.keras.preprocessing.image.save_img(f"{OUTPUT_FOLDER}/0.png", target_images[0])


gen = models.generator.get_skip_generator(latent_dim=LATENT_DIM, channels=CHANNELS, target_size=IMG_SIZE, latent_style_layers=LATENT_STYLE_LAYERS)
gen.compile(optimizer=Adam(learning_rate=0.01), loss="mae")
gen.summary()
mae = tf.keras.losses.MeanAbsoluteError()

for step in range(TRAIN_STEPS):
    gen_input = random_generator_input(BATCH_SIZE, LATENT_DIM, IMG_SIZE) 
    if step % SAVE_STEPS == 0:
        generated_images = gen.predict(gen_input)
        for i in range(4):
            log2_mae = math.log2(mae(target_images[i], generated_images[i]).numpy())
            tf.keras.preprocessing.image.save_img(f"{OUTPUT_FOLDER}/{step}_{i}_{-log2_mae:.2f}.png", generated_images[i])
    loss = gen.train_on_batch(gen_input, target_images)
    print(f"{math.log2(loss):.2f}")