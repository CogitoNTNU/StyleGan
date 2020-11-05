import models.silence
import models.discriminator
import models.generator
import tensorflow as tf

#discriminator = models.discriminator.get_resnet_discriminator(image_size=(2*128, 3*128))
#print(discriminator.summary())
# tf.keras.utils.plot_model(discriminator, "discriminator.png", show_shapes=True)

generator = models.generator.get_skip_generator(start_size=(4,6), target_size=(256, 384))
tf.keras.utils.plot_model(generator, "generator.png", show_shapes=True)

print(generator.summary())