import silence

import tensorflow as tf
from tensorflow import keras
import generator
import pytest

def test_normalize_channel_std():

    x = tf.constant([
        [[0.0, 1.0], [1.0, 0.0]],
        [[4.0, 4.0], [0.0, 0.0]],
        [[8.0, 8.0], [0.0, 0.0]],
    ])

    # Transpose to ensure channels last
    x = tf.transpose(x)

    # Verify computation of standard deviation
    target_std = tf.constant([0.5, 2.0, 4.0])
    actual_std = keras.backend.std(x, axis=[-3, -2])
    
    assert all(tf.math.equal(actual_std, target_std))

    # Test normalization manually
    z = tf.math.multiply(1.0/actual_std, x) 
    z_std = keras.backend.std(z, axis=[-3, -2])
    target_z_std =  tf.constant([1.0, 1.0, 1.0])
    assert all(tf.math.equal(z_std, target_z_std))

    x_norm = generator.normalize_channel_std(x)
    x_std = keras.backend.std(z, axis=[-3, -2])
    for i in x_std.numpy():
        assert i == pytest.approx(1.0)

def test_scale_channels():
    x = tf.constant([
        [[0.0, 1.0], [1.0, 0.0]],
        [[4.0, 4.0], [0.0, 0.0]],
        [[8.0, 8.0], [0.0, 0.0]],
    ])

    # Transpose to ensure channels last
    x = tf.transpose(x)
    s = tf.constant([1.0, 2.0, 3.0])
    z = generator.scale_channels([s, x])
    target_z = tf.transpose(tf.constant([
        [[0.0, 1.0], [1.0, 0.0]],
        [[8.0, 8.0], [0.0, 0.0]],
        [[24.0, 24.0], [0.0, 0.0]],
    ]))

    assert tf.reduce_all(tf.equal(z, target_z))
