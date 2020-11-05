import models.silence
import models.discriminator
import models.generator
import tensorflow as tf

discriminator = models.discriminator.get_resnet_discriminator(img_size=(384, 256))
print(discriminator.summary())
# tf.keras.utils.plot_model(discriminator, "discriminator.png", show_shapes=True)

generator = models.generator.get_skip_generator(start_size=(6,4), target_size=(384, 256))
#tf.keras.utils.plot_model(generator, "generator.png", show_shapes=True)

print(generator.summary())