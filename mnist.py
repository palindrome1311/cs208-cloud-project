import tensorflow as tf
from tensorflow import keras
import datetime
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

print('----------TRAINING MNIST DATASET USING CUSTOM SEQUENTIAL MODEL-------------')
a = datetime.datetime.now()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test,y_test)
b = datetime.datetime.now()
c=b-a


print('----------TRAINING ENDED -----------')
print('Time Required in seconds - ',c.total_seconds())