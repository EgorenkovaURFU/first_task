import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
     
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),  # преобразую матрицу изображения 28*28 в вектор 784
    Dense(128, activation='relu'),  # создаю скрытый слой из 128 полносвязных нейронов с ф-цией акт relu
    Dense(10, activation='softmax')  # создаю скрытый слой из 10 полносвязных нейронов с ф-цией акт softmax
])

myOpt = keras.optimizers.SGD(learning_rate=0.001, nesterov = True)
model.compile(optimizer=myOpt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
     


n = 88
x = np.expand_dims(x_test[n], axis=0)
result = model.predict(x)
print(result)
print(f'Распознанная цифра: {np.argmax(result)}')


