import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math


def normalize_channel_std(x):
    epsilon = 1e-6
    std = keras.backend.std(x, axis=[-3, -2], keepdims=True) # Should result in (BATCH, 1, 1, CHANNELS) output shape
    return tf.math.multiply(1 / (std + epsilon), x)


def scale_channels(x):
    return tf.math.multiply(x[0], x[1])


# Style block according to Figure 2c in the StyleGAN2 paper
# StyleGAN2 Appendix B: "We initialize all weights using the standard normal distribution, and all bias and noise scaling factors are initialized to zero.""
def style_block(x, latent_in, noise_in, channels=64, latent_style_layers=2, upsample=True, name="1"):
    # Channel scaling parameter from latent input
    s = latent_in
    for i in range(latent_style_layers):
        s = layers.Dense(channels, activation=None, name=f"latent_scale_{i}_{name}")(s)
        s = layers.LeakyReLU(alpha=0.2)(s)

    # Scale channels
    s = layers.Reshape((1,1,channels), name=f"scale_reshape_{name}")(s)
    x = layers.Lambda(scale_channels, name=f"channel_scale_{name}")([s, x])

    # Upsampling
    if upsample:
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"upsampling_{name}")(x)

    # Standard 2D convolution with 3x3 kernel
    x = layers.Conv2D(channels, kernel_size=(3, 3), padding="same", kernel_initializer="random_normal",
                      bias_initializer="zeros", name=f"conv2d_{name}")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Normalize to unit standard deviation in each output channel
    x = layers.Lambda(normalize_channel_std, name=f"norm_channel_std_{name}")(x)

    # Add noise
    # We use a single channel of standard normal noise that is broadcasted across the channels of the generated image
    # For each channel in the generated image we have a learnable scaling factor and bias parameter, implemented using a Conv2D layer with a kernel of size 1.
    b = layers.Conv2D(channels, kernel_size=(1, 1), kernel_initializer="zeros", bias_initializer="zeros",
                      name=f"broadcast_modulate_noise_{name}")(noise_in)
    x = layers.Add(name=f"add_noise_{name}")([b, x])

    return x


def get_generator(latent_dim=64, channels=64, target_size=64, latent_style_layers=2):
    num_upsamples = int(math.log2(target_size) - 2)
    side_length = 4

    # Learnable constant image
    dummy_in = layers.Input(shape=(1,), name="dummy_in")
    x = layers.Dense(side_length * side_length * channels, name="const_img", kernel_initializer="zeros",
                     bias_initializer="random_normal")(dummy_in)
    x = layers.Reshape((side_length, side_length, channels))(x)

    # Latent input
    latent_in = layers.Input(shape=(latent_dim,), name="latent_in")

    noise_inputs = []
    for i in range(num_upsamples):
        # Style block without upsampling
        noise_in = layers.Input(shape=(side_length, side_length, 1), name=f"noise_in_{side_length}x{side_length}")
        noise_inputs.append(noise_in)
        x = style_block(x, latent_in, noise_in, channels=channels, latent_style_layers=latent_style_layers,
                        upsample=False, name=f"{side_length}x{side_length}")

        # Style block with upsampling
        side_length = 2 * side_length
        noise_in = layers.Input(shape=(side_length, side_length, 1),
                                name=f"noise_in_upsample_{side_length}x{side_length}")
        noise_inputs.append(noise_in)
        x = style_block(x, latent_in, noise_in, channels=channels, latent_style_layers=latent_style_layers,
                        upsample=True, name=f"upsample_{side_length}x{side_length}")

    # Convert feature maps to RGB image
    x = layers.Conv2D(filters=3, kernel_size=(3, 3), kernel_initializer="random_normal", bias_initializer="zeros",
                      activation="tanh", padding="same")(x)

    generator = tf.keras.Model(inputs=[dummy_in, latent_in] + noise_inputs, outputs=x, name="generator")
    return generator

def get_skip_generator(latent_dim=64, channels=64, target_size=64, latent_style_layers=2):
    num_upsamples = int(math.log2(target_size) - 2)
    side_length = 4

    # Learnable constant image
    dummy_in = layers.Input(shape=(1,), name="dummy_in")
    x = layers.Dense(side_length * side_length * channels, name="const_img", kernel_initializer="zeros",
                     bias_initializer="random_normal")(dummy_in)
    x = layers.Reshape((side_length, side_length, channels))(x)

    # RGB output
    y = layers.Conv2D(filters=3, kernel_size=(3, 3), kernel_initializer="random_normal", bias_initializer="zeros",
                      activation="tanh", padding="same")(x)
    
    # Latent input
    latent_in = layers.Input(shape=(latent_dim,), name="latent_in")

    noise_inputs = []
    for i in range(num_upsamples):
        # Style block without upsampling
        noise_in = layers.Input(shape=(side_length, side_length, 1), name=f"noise_in_{side_length}x{side_length}")
        noise_inputs.append(noise_in)
        x = style_block(x, latent_in, noise_in, channels=channels, latent_style_layers=latent_style_layers,
                        upsample=False, name=f"{side_length}x{side_length}")

        # Style block with upsampling
        side_length = 2 * side_length
        noise_in = layers.Input(shape=(side_length, side_length, 1),
                                name=f"noise_in_upsample_{side_length}x{side_length}")
        noise_inputs.append(noise_in)
        x = style_block(x, latent_in, noise_in, channels=channels, latent_style_layers=latent_style_layers,
                        upsample=True, name=f"upsample_{side_length}x{side_length}")

        # Convert deep image to RGB
        z = layers.Conv2D(filters=3, kernel_size=(3, 3), kernel_initializer="random_normal", bias_initializer="zeros",
                    activation="tanh", padding="same", name=f"to_rgb_{side_length}x{side_length}")(x)

        # Add deep RGB to upsampled current RGB
        y = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"rgb_upsampling_{side_length}x{side_length}")(y)
        y = layers.Add(name=f"add_deep_rgb_{side_length}x{side_length}")([y, z])

    generator = tf.keras.Model(inputs=[dummy_in, latent_in] + noise_inputs, outputs=y, name="generator")
    return generator

def get_dropout_generator(latent_dim=64, channels=64, target_size=64):
    side_length = 4
    latent_in = layers.Input(shape=(latent_dim,), name="latent_in")
    x = layers.Dense(4*4*channels)(latent_in)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((side_length, side_length, channels))(x)

    dropout_inputs = []
    while side_length < target_size:
        # 2x Conv2D
        x = layers.Conv2D(filters=channels, kernel_size=(3,3), padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(filters=channels, kernel_size=(3,3), padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Dropout entire channels
        s = layers.Input(shape=(channels,))
        dropout_inputs.append(s)
        s = layers.Reshape((1,1,channels))(s)
        x = layers.Lambda(scale_channels)([s, x])
        
        # Upsampling
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        side_length = 2*side_length

    x = layers.Conv2D(filters=3, kernel_size=(3,3), padding="same")(x)

    generator = tf.keras.Model(inputs=[latent_in] + dropout_inputs, outputs=x, name="generator")
    return generator


def random_dropout_input(batch_size, latent_size, channels, img_size, alpha=0.2):
    latent_noise = np.random.normal(size=(batch_size,latent_size))
    dropout_noise = []
    num_upsamples = int(math.log2(img_size)-2)
    for i in range(num_upsamples):
        dropout_noise.append(np.random.choice([1.0, 0.0], size=(batch_size, channels), replace=True, p=[1.0-alpha, alpha]))
    return [latent_noise, *dropout_noise]

def random_generator_input(batch_size, latent_dim, img_size):
    null_input = np.zeros((batch_size, 1))
    z_noise = np.random.normal(size=(batch_size,latent_dim))
    noises = []
    num_upsamles = int(math.log2(img_size)-2)
    noises.append(np.random.normal(size=(batch_size, 4, 4, 1)))
    for i in range(1,num_upsamles):
        noises.append(np.random.normal(size=(batch_size,4*(2**i),4*(2**i),1)))
        noises.append(np.random.normal(size=(batch_size,4*(2**(i)),4*(2**(i)),1)))
    noises.append(np.random.normal(size=(batch_size, img_size, img_size, 1)))
    return [null_input, z_noise, *noises]