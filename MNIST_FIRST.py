from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

#搭建网络

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#数据预处理

train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

#准备标签

from keras.utils import to_categorical

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#拟合
network.fit(train_images,train_labels,batch_size=128,epochs=5)

test_loss,test_acc=network.evaluate(test_images,test_labels)
print('test_acc',test_acc)
