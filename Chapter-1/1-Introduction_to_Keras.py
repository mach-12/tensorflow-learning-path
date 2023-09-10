
"""
A perceptron is a simple algorithm that given an input vector of x of ,m values, outputs either yes or no

f(x) = 1 if wx + b > 0 else 0
w is a vector of weights, b is bias. wx + b defines a hyperplane which changes position according to the values assigned to w and b.
"""

import tensorflow as tf
from tensorflow import keras

NB_CLASSES = 10
RESHAPED = 784
model = tf.keras.models.Sequential()
model.add(
    keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPED,), kernel_initializer='zeros',
                       name='dense_layer', activation='softmax')
)

# Kernel can initialized with random_uniform, or random_normal distribution as well.


