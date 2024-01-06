"""
AutoGraph converts eager-style Python Code into Graph-Generating code.
This is harnessed by using the @tf.functoin decorator
"""
import tensorflow as tf


def linear_layer(x):
    return 3 * x + 2


@tf.function
def simple_nn(x):
    return tf.nn.relu(linear_layer(x))


def simple_function(x):
    return 3 * x

print(simple_nn)

# Output: <tensorflow.python.eager.polymorphic_function.polymorphic_function.Function object at 0x2d009f810>