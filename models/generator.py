import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


def normalize_channel_std(x):
    epsilon = 1e-6
    std = keras.backend.std(x, axis=[-3,-2]) 
    return tf.math.multiply(1/(std + epsilon), x) 

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
    x = layers.Lambda(scale_channels, name=f"channel_scale_{name}")([s, x])

    # Upsampling
    if upsample:
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"upsampling_{name}")(x)

    # Standard 2D convolution with 3x3 kernel
    x = layers.Conv2D(channels, kernel_size=(3,3), padding="same", kernel_initializer="random_normal", bias_initializer="zeros", name=f"conv2d_{name}")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Normalize to unit standard deviation in each output channel
    x = layers.Lambda(normalize_channel_std, name=f"norm_channel_std_{name}")(x)

    # Add noise
    # We use a single channel of standard normal noise that is broadcasted across the channels of the generated image
    # For each channel in the generated image we have a learnable scaling factor and bias parameter, implemented using a Conv2D layer with a kernel of size 1.
    b = layers.Conv2D(channels, kernel_size=(1,1), kernel_initializer="zeros", bias_initializer="zeros", name=f"broadcast_modulate_noise_{name}")(noise_in)
    x = layers.Add(name=f"add_noise_{name}")([b, x])

    return x

def SimpleGenerator(latent_dim=64, channels=64, target_size=64, latent_style_layers=2):

    num_upsamples = int(math.log2(target_size) - 2)
    side_length = 4

    # Learnable constant image
    dummy_in = layers.Input(shape=(1,), name="dummy_in")
    x = layers.Dense(side_length*side_length*channels, name="const_img", kernel_initializer="zeros", bias_initializer="random_normal")(dummy_in)
    x = layers.Reshape((side_length, side_length, channels))(x)

    # Latent input
    latent_in = layers.Input(shape=(latent_dim,), name="latent_in")

    noise_inputs = []
    for i in range(num_upsamples):
        
        # Style block without upsampling
        noise_in = layers.Input(shape=(side_length, side_length, 1), name=f"noise_in_{side_length}x{side_length}")
        noise_inputs.append(noise_in)
        x = style_block(x, latent_in, noise_in, channels=channels, latent_style_layers=latent_style_layers, upsample=False, name=f"{side_length}x{side_length}")
        
        # Style block with upsampling
        side_length = 2*side_length
        noise_in = layers.Input(shape=(side_length, side_length, 1), name=f"noise_in_upsample_{side_length}x{side_length}")
        noise_inputs.append(noise_in)
        x = style_block(x, latent_in, noise_in, channels=channels, latent_style_layers=latent_style_layers, upsample=True, name=f"upsample_{side_length}x{side_length}")


    # Convert feature maps to RGB image
    x = layers.Conv2D(filters=3, kernel_size=(3,3), kernel_initializer="random_normal", bias_initializer="zeros", activation="tanh", padding="same")(x)

    generator = tf.keras.Model(inputs=[dummy_in, latent_in] + noise_inputs, outputs=x, name="generator")
    return generator


def get_generator():
    return None