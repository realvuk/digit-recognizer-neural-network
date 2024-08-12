# Digit Recognizer with Neural Network

## Overview

This project demonstrates how to build a neural network model to recognize handwritten digits using the MNIST dataset. We utilize TensorFlow and Keras to create and train a simple neural network that can accurately classify digits from images. The dataset is provided by Keras, which simplifies the data loading process.
Prerequisites

To run this project, you will need:

Python 3.x
TensorFlow
Keras
NumPy
Pandas
Matplotlib
scikit-learn

You can install the required packages using pip:


```
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Import Packages

First, we need to import the necessary libraries for data processing, visualization, and neural network creation:


```
import numpy as np  # Linear algebra
import pandas as pd  # Data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # For plots
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_openml
```

## Load the Data

We will use the MNIST dataset, which is a collection of 28x28 pixel grayscale images of handwritten digits. The dataset is conveniently available through Keras.

```
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.keys()
```

Split the Data

We will split the dataset into training, validation, and test sets. The training set will be used to train the model, the validation set will be used to tune hyperparameters, and the test set will be used to evaluate the final model's performance.

```
X, y = mnist["data"], mnist["target"]
X.shape
y.shape

mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.
```

Note: The pixel values of images are normalized to the range [0, 1] by dividing by 255.

## Building the Model

We will build a simple neural network with two hidden layers using Keras. The network will consist of:

An input layer that flattens the 28x28 images into a 1D vector.
Two hidden layers with 300 and 100 neurons, respectively, using the selu activation function.
An output layer with 10 neurons and the softmax activation function to classify digits from 0 to 9.

```
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(10, activation="softmax")
])
```

## Model Summary

To understand the model architecture, we can print a summary:

```
model.layers
model.summary()
```

We can also visualize the model architecture using:

```
keras.utils.plot_model(model, "my_model.png", show_shapes=True)
```

Inspect Weights and Biases

To inspect the weights and biases of the first hidden layer:

```
hidden1 = model.layers[1]
hidden1.name
model.get_layer(hidden1.name) is hidden1

weights, biases = hidden1.get_weights()
weights
weights.shape
biases
biases.shape
```

## Compile the Model

We compile the model with the following settings:

Loss function: sparse_categorical_crossentropy
Optimizer: sgd (Stochastic Gradient Descent)
Metrics: accuracy

```
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```

## Train the Model

We train the model using the training data and validate it with the validation data:

```
mnist_fit = model.fit(X_train, y_train, epochs=30,
                      validation_data=(X_valid, y_valid))
```
              

## Monitor Training Progress

To monitor the training progress, plot the training and validation accuracy:

```
pd.DataFrame(mnist_fit.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
```

## Evaluate the Model

Finally, evaluate the model's performance on the test set:

```
model.evaluate(X_test, y_test)
```

## Results

The model achieves an accuracy of approximately 97.35% on the test set.
