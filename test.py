import models.silence
import models.discriminator
import models.generator
import tensorflow as tf

dataset = tf.keras.preprocessing.image_dataset_from_directory("datasets/keras_abstract/", label_mode=None, batch_size=32, image_size=(512, 512))
dataset = dataset.map(lambda x: x/127.5 -1.0).prefetch(8)


for batch in dataset:
    print(batch)