# Import necessary libraries
import numpy as np
import keras
from keras.datasets import cifar10
from keras import models, layers
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils import load_img, img_to_array
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# y_train = y_train.reshape(-1,)
# y_test = y_test.reshape(-1,)
# classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# X_train = X_train / 255.0
# X_test = X_test / 255.0

# ann = models.Sequential([
#         layers.Flatten(input_shape=(32,32,3)),
#         layers.Dense(3000, activation='relu'),
#         layers.Dense(1000, activation='relu'),
#         layers.Dense(10, activation='softmax')    
#     ])

# ann.compile(optimizer='SGD',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# ann.fit(X_train, y_train, epochs=5)


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)

cnn.save('cifar10_model.h5')


# model = load_model('cifar10_model.h5')
# img_path = 'image.jpg'
# img = image.load_img(img_path, target_size=(32, 32))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = x / 255.0

# preds = model.predict(x)

# class_idx = np.argmax(preds)
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# class_name = class_names[class_idx]
# print(f"Predicted class: {class_name}")