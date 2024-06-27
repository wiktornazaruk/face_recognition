import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Input
from keras.losses import categorical_crossentropy
from keras import layers
from keras import regularizers
from keras import backend as K
from matplotlib import pyplot as plt
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
import re


CLASSES = 48
IMAGE_SIZE = 24


def read_file(filename):
    with open(filename, 'r') as file:
        data = file.read()

    data_list = data.split()

    data_list = [float(item) for item in data_list]

    data_list = np.delete(data_list, [0, 1])

    print(data_list[:9])

    num_subarrays = len(data_list) // 584

    array_of_Images = []
    array_of_Indexes = []

    for i in range(num_subarrays):
        subarray = data_list[i * 584: (i + 1) * 584]
        array_of_Images.append(subarray)
        Index = int(subarray[578])
        array_of_Indexes.append(Index)

    print(len(array_of_Images))
    print(array_of_Images[2])

    print(array_of_Indexes)

    for i in range(num_subarrays):
        array_of_Images[i] = array_of_Images[i][:-8]

    Images = np.array(array_of_Images).reshape((num_subarrays, 24, 24))
    Images = np.array(Images, dtype=float)

    return Images, array_of_Indexes


Imagies1, Indexes1 = read_file('./famous48/x24x24.txt')

Imagies2, Indexes2 = read_file('./famous48/y24x24.txt')

Imagies3, Indexes3 = read_file('./famous48/z24x24.txt')

Imagies = np.concatenate((Imagies1, Imagies2, Imagies3))
Indexes = np.concatenate((Indexes1, Indexes2, Indexes3))

NUMBER_OF_TRAIN_EXAMPLES = int(len(Imagies) * 0.8)
NUMBER_TEST_EXAMPLES = len(Imagies) - NUMBER_OF_TRAIN_EXAMPLES


IMGtrain, IMGtest, INDEXtrain, INDEXtest = train_test_split(
    Imagies, Indexes, train_size=NUMBER_OF_TRAIN_EXAMPLES, test_size=NUMBER_TEST_EXAMPLES, random_state=1)

INDEXtrain_fixed = np.zeros((INDEXtrain.shape[0], CLASSES))
INDEXtest_fixed = np.zeros((INDEXtest.shape[0], CLASSES))

for i, value in enumerate(INDEXtrain):
    print(value)
    INDEXtrain_fixed[i][value] = 1

for i, value in enumerate(INDEXtest):
    INDEXtest_fixed[i][value] = 1

print(len(IMGtest))


conv_regularizer = regularizers.l2(0.0006)
dense_regularizer = regularizers.l2(0.01)

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_regularizer=conv_regularizer)


model = keras.Sequential(
    [
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        DefaultConv2D(96),
        layers.MaxPooling2D(pool_size=3, strides=2),

        tf.keras.layers.Dropout(0.3),
        DefaultConv2D(256, kernel_size=5),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

        tf.keras.layers.Dropout(0.4),
        DefaultConv2D(384),
        tf.keras.layers.Dropout(0.5),
        DefaultConv2D(384),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(384, activation='relu',
                              kernel_regularizer=dense_regularizer),
        tf.keras.layers.Dense(CLASSES, activation='softmax'),
    ]
)

model.summary()

model.compile(loss=categorical_crossentropy, metrics=["accuracy"])

batch_size = 200
epochs = 200

history = model.fit(
    IMGtrain, INDEXtrain_fixed,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    shuffle=True,
)

model.evaluate(IMGtest, INDEXtest_fixed)


length = len(history.history['accuracy'])
print("train set accuracy:", round(history.history['accuracy'][length-1], 3))
print("train loss:", round(history.history['loss'][length-1], 3))
print("validation set accuracy:", round(
    history.history['val_accuracy'][length-1], 3))
print("validation set loss:", round(history.history['val_loss'][length-1], 3))


results = model.evaluate(IMGtest, INDEXtest_fixed)
print("test set accuracy:", round(results[1], 3))
print("test set loss:", round(results[0], 3))


plt.figure(figsize=(16, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
