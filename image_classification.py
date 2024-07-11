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

# Constants
NUM_CLASSES = 48
IMG_SIZE = 24


def read_file(filename):
    with open(filename, 'r') as file:
        data = file.read()

    data_list = [float(item) for item in data.split()]
    data_list = np.delete(data_list, [0, 1])

    print(data_list[:9])

    num_images = len(data_list) // 584

    images = []
    labels = []

    for i in range(num_images):
        subarray = data_list[i * 584: (i + 1) * 584]
        images.append(subarray[:-8])
        label = int(subarray[578])
        labels.append(label)

    print(f"Number of images: {len(images)}")
    print(f"Sample image data: {images[2]}")
    print(f"Labels: {labels}")

    images_array = np.array(images).reshape((num_images, IMG_SIZE, IMG_SIZE))
    images_array = np.array(images_array, dtype=float)

    return images_array, labels


# Read and combine datasets
images1, labels1 = read_file('./famous48/x24x24.txt')
images2, labels2 = read_file('./famous48/y24x24.txt')
images3, labels3 = read_file('./famous48/z24x24.txt')

all_images = np.concatenate((images1, images2, images3))
all_labels = np.concatenate((labels1, labels2, labels3))

# Split into training and testing datasets
train_size = int(len(all_images) * 0.8)
test_size = len(all_images) - train_size

X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, train_size=train_size, test_size=test_size, random_state=1)

# One-hot encode labels
y_train_one_hot = np.zeros((y_train.shape[0], NUM_CLASSES))
y_test_one_hot = np.zeros((y_test.shape[0], NUM_CLASSES))

for i, label in enumerate(y_train):
    y_train_one_hot[i][label] = 1

for i, label in enumerate(y_test):
    y_test_one_hot[i][label] = 1

print(f"Test dataset size: {len(X_test)}")

# Regularizers
conv_regularizer = regularizers.l2(0.0006)
dense_regularizer = regularizers.l2(0.01)

DefaultConv2D = partial(layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_regularizer=conv_regularizer)

# Build the model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    DefaultConv2D(96),
    layers.MaxPooling2D(pool_size=3, strides=2),
    layers.Dropout(0.3),
    DefaultConv2D(256, kernel_size=5),
    layers.MaxPooling2D(pool_size=3, strides=2),
    layers.Dropout(0.4),
    DefaultConv2D(384),
    layers.Dropout(0.5),
    DefaultConv2D(384),
    layers.MaxPooling2D(pool_size=3, strides=2),
    layers.Flatten(),
    layers.Dropout(0.6),
    layers.Dense(384, activation='relu', kernel_regularizer=dense_regularizer),
    layers.Dense(NUM_CLASSES, activation='softmax'),
])

model.summary()

# Compile the model
model.compile(loss=categorical_crossentropy, metrics=["accuracy"])

# Training parameters
batch_size = 200
epochs = 200

# Train the model
history = model.fit(
    X_train, y_train_one_hot,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    shuffle=True,
)

# Evaluate the model
model.evaluate(X_test, y_test_one_hot)

# Print final metrics
length = len(history.history['accuracy'])
print(f"train set accuracy: {round(history.history['accuracy'][length-1], 3)}")
print(f"train loss: {round(history.history['loss'][length-1], 3)}")
print(
    f"validation set accuracy: {round(history.history['val_accuracy'][length-1], 3)}")
print(
    f"validation set loss: {round(history.history['val_loss'][length-1], 3)}")

test_results = model.evaluate(X_test, y_test_one_hot)
print(f"test set accuracy: {round(test_results[1], 3)}")
print(f"test set loss: {round(test_results[0], 3)}")

# Plot accuracy
plt.figure(figsize=(16, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
