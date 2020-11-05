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
    x = layers.Conv2D(channels, kernel_size=(3, 3), padding="same", kernel_initializer="random_normal", bias_initializer="zeros", name=f"conv2d_{name}")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Normalize to unit standard deviation in each output channel
    x = layers.Lambda(normalize_channel_std, name=f"norm_channel_std_{name}")(x)

    # Add noise
    # We use a single channel of standard normal noise that is broadcasted across the channels of the generated image
    # TODO: Try different random noise for each channel
    # For each channel in the generated image we have a learnable scaling factor and bias parameter, implemented using a Conv2D layer with a kernel of size 1.
    b = layers.Conv2D(channels, kernel_size=(1, 1), kernel_initializer="zeros", bias_initializer="zeros", name=f"broadcast_modulate_noise_{name}")(noise_in)
    x = layers.Add(name=f"add_noise_{name}")([b, x])

    return x


def get_skip_generator(start_size=(4,4), target_size=(64, 64), latent_dim=64, channels=64, latent_style_layers=2):
    width = start_size[0]
    height = start_size[1]
    target_width = target_size[0]

    # Learnable constant image
    dummy_in = layers.Input(shape=(1,), name="dummy_in")
    x = layers.Dense(width * height * channels, name="const_img", kernel_initializer="zeros", bias_initializer="random_normal")(dummy_in)
    x = layers.Reshape((width, height, channels))(x)

    # RGB output
    y = layers.Conv2D(filters=3, kernel_size=(3, 3), kernel_initializer="random_normal", bias_initializer="zeros", activation="tanh", padding="same")(x)
    
    # Latent input
    latent_in = layers.Input(shape=(latent_dim,), name="latent_in")

    noise_inputs = []
    while width < target_width:
        # Style block without upsampling
        noise_in = layers.Input(shape=(width, height, 1), name=f"noise_in_{width}x{height}")
        noise_inputs.append(noise_in)
        x = style_block(x, latent_in, noise_in, channels=channels, latent_style_layers=latent_style_layers, upsample=False, name=f"{width}x{height}")

        # Style block with upsampling
        width = 2*width
        height = 2*height
        noise_in = layers.Input(shape=(width, height, 1), name=f"noise_in_upsample_{width}x{height}")
        noise_inputs.append(noise_in)
        x = style_block(x, latent_in, noise_in, channels=channels, latent_style_layers=latent_style_layers, upsample=True, name=f"upsample_{width}x{height}")

        # Convert deep image to RGB
        z = layers.Conv2D(filters=3, kernel_size=(3, 3), kernel_initializer="random_normal", bias_initializer="zeros", activation="tanh", padding="same", name=f"to_rgb_{width}x{height}")(x)

        # Upsample current RGB and add deep RGB
        y = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"rgb_upsampling_{width}x{height}")(y)
        y = layers.Add(name=f"add_deep_rgb_{width}x{height}")([y, z])

    generator = tf.keras.Model(inputs=[dummy_in, latent_in] + noise_inputs, outputs=y, name="generator")
    return generator

def random_generator_input(batch_size, latent_dim, start_size=(2,2), target_size=(64,64)):
    dummy_input = np.zeros((batch_size, 1))
    latent_noise = np.random.normal(size=(batch_size, latent_dim))
    noises = []

    width = start_size[0]
    height = start_size[1]

    num_upsamles = int(math.log2(target_size[0]/start_size[0]))
    noises.append(np.random.normal(size=(batch_size, width, height, 1)))
    for i in range(1, num_upsamles):
        width = 2*width
        height = 2*height
        noises.append(np.random.normal(size=(batch_size, width, height, 1)))
        noises.append(np.random.normal(size=(batch_size, width, height, 1)))
    
    noises.append(np.random.normal(size=(batch_size, target_size[0], target_size[1], 1)))
    return [null_input, z_noise, *noises]