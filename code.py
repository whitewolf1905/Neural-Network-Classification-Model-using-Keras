
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import matplotlib.pyplot as plt 

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

plt.imshow(X_train[0])

num_pixels = X_train.shape[1]*X_train.shape[2] # find the size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')  # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')     # flatten test images

X_train = X_train/255
X_test = X_test/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]


def classification_model():
	model = Sequential()
	model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

model = classification_model()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

scores = model.evaluate(X_test,y_test,verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))     
