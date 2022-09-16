

from keras.datasets import reuters

(train_data,train_label),(test_data,test_label)=reuters.load_data(num_words=10000)

print(len(train_data))
#数据预处理
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    result = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        result[i,sequence]=1
    return result
# one_hot编码
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)


from keras.utils.np_utils import to_categorical

one_hot_train_label=to_categorical(train_label)
one_hot_test_label=to_categorical(test_label)
#模型构建
from keras import models
from keras import layers

model=models.Sequential()
#不理解为甚这块的特征值会达到10000
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#流出验证集
x_val=x_train[:1000]
partial_train=x_train[1000:]

train_val=one_hot_train_label[:1000]
partial_train_label=one_hot_train_label[1000:]

history=model.fit(partial_train,partial_train_label,batch_size=512,epochs=20,validation_data=(x_val,train_val))


print(history.history)
import matplotlib.pyplot as plt

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training loss and validation loss')
plt.legend()

plt.show()

plt.clf()
acc=history.history['acc']
val_acc=history.history['val_acc']

plt.plot(epochs,acc,'bo',label='accuracy')
plt.plot(epochs,val_acc,'b',label='validation_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy and validation accuracy')
plt.legend()
plt.show()


model1=models.Sequential()
model1.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model1.add(layers.Dense(64,activation='relu'))
model1.add(layers.Dense(46,activation='softmax'))

model1.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
model1.fit(partial_train,partial_train_label,epochs=9,batch_size=512,validation_data=(x_val,train_val))

results=model1.evaluate(x_test,one_hot_test_label)
print(results)


predictions=model1.predict(x_test)
print(predictions[0].shape)

# 还有一种处理标签的方法 就是转化为整形张量
# 唯一有所区别的使loss损失函数不同一个使categorical_crossentropy,sparse_categorical_crossentropy

