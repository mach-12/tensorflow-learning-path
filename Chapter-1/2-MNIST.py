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
DROPOUT = 0.1

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

# Dropout Layer
model.add(
    keras.layers.Dropout(DROPOUT)
)

# Hidden Layer 2
model.add(
    keras.layers.Dense(N_HIDDEN, name='dense_layer_2', activation='relu'))

# Dropout Layer
model.add(
    keras.layers.Dropout(DROPOUT)
)
# Output
model.add(
    keras.layers.Dense(NB_CLASSES, name='output', activation='softmax'))

# Compiling Model
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

# Training Model
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VAL_SPLIT)

# Evaluation of Model
test_loss, test_acc = model.evaluate(X_test, Y_test)

# Printing Data
train_loss, train_acc = history.history['loss'][-1], history.history['accuracy'][-1]
val_loss, val_acc = history.history['val_loss'][-1], history.history['val_accuracy'][-1]

metrics_list = [train_acc, val_acc, test_acc, train_loss, val_loss, test_loss]
metrics_list = [f'{float(f"{i:.4g}"):g}' for i in metrics_list]  # Rounding to 4 Significant Digits

print(*["TRAIN_ACC", "VAL_ACC", "TEST_ACC", "TRAIN_LOSS", "VAL_LOSS", "TEST_LOSS"], sep="\t")
print(*metrics_list, sep="\t\t")

# RESULTS
# Train Accuracy = 0.998
# Val Accuracy = 0.9799
# Test Accuracy = 0.9810
# Train Loss = 0.0052
# Val Loss = 0.1699
# Test Loss = 0.1600
