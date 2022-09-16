from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

import numpy as np


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# 数据向量化
train_d = vectorize(train_data)
test_d = vectorize(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 二分类问题常用损失函数二元交叉熵，和分类交叉熵

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 拆分验证集和训练集

x_val = train_d[:10000]
partial_x = train_d[10000:]

y_val = y_train[:10000]
partial_y = y_train[10000:]

history = model.fit(partial_x, partial_y, batch_size=512, epochs=20, validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='Training_loss')
plt.plot(epochs, val_loss, 'bo', label='validation_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# 清空图像
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'bo', label="validation accuracy")

plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.title('Training and validation accuracy')
plt.legend()

plt.show()
# 进行模型预测
print(model.predict(x_val))

# 主要是限制了分类和回归
