import models.adverserial
import models.discriminator
import models.generator
import data_tools.image_generator
from keras.optimizers import Adam

IMG_SIZE=64
LEARNING_RATE=0.0001


gen = models.generator.get_generator(IMG_SIZE)
disc = models.discriminator.get_discriminator(IMG_SIZE)

disc_optimizer=Adam(lr=LEARNING_RATE)
disc.compile(optimizer=disc_optimizer,loss="binary_crossentropy", metrics=['accuracy'])
disc.compile()

adv = models.adverserial.get_adverserial(gen,disc)

adv_optimizer=Adam(lr=LEARNING_RATE)
adv.compile(optimizer=adv_optimizer,loss="binary_crossentropy", metrics=['accuracy'])