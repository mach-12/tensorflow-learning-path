"""
 PERCEPTION doesn't show step-by-step learning behaviour.

    sigmoid(x) =  1 / (1 + e^(-x)
    tanh(z) = e^z - e^-z / e^z + e^-z
    ReLU = max(0, x); Zero for negative values and grows linearly for positive values. It addressed some optimized
    ELU
    LeakyReLU
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Parameters
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VAL_SPLIT = 0.2

# Loading Dataset
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Checking Shapes
print("Shapes")
print('X_train', X_train.shape)
print('Y_train', Y_train.shape)
print('X_test', X_test.shape)
print('Y_test', Y_test.shape)

# Reshaping Data
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizing input
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One hot coding
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# Building the model
model = tf.keras.models.Sequential()

# Hidden Layer 1
model.add(
    keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPED,), name='dense_layer_1', activation='relu'))
# Hidden Layer 2
model.add(
    keras.layers.Dense(N_HIDDEN, name='dense_layer_2', activation='relu'))

model.add(
    keras.layers.Dense(NB_CLASSES, name='output', activation='softmax'))

# Compiling Model
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

# Training Model
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VAL_SPLIT)

# Evaluation of Model
test_less, test_acc = model.evaluate(X_test, Y_test)
print('Test Accuracy:', test_acc)

# RESULTS
# Train Accuracy = 0.9765
# Val Accuracy = 0.9754
# Test Accuracy = 0.9765
