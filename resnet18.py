from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

import numpy as np
np.max(x_train[0])

x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
input_shape = (28, 28, 1)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

inputs = keras.Input(shape=(28, 28, 1))
con1 = layers.Conv2D(64,4,activation='relu',padding='same')(inputs)
pool1 = layers.MaxPooling2D((3,3),2)(con1)
l1con1 = layers.Conv2D(64,3,activation='relu',padding='same')(pool1)
l1con2 = layers.Conv2D(64,3,activation='relu',padding ='same')(l1con1)

merge1 = layers.concatenate([pool1,l1con2]) 
l1con3 = layers.Conv2D(64,3,activation='relu',padding ='same')(merge1)
l1con4 = layers.Conv2D(64,3,activation='relu',padding ='same')(l1con3)

merge2 = layers.concatenate([l1con2,l1con4])
l2con1 = layers.Conv2D(128,3,activation='relu',padding ='same')(merge2)
l2con2 = layers.Conv2D(128,3,activation='relu',padding ='same')(l2con1)

merge3 = layers.concatenate([l1con4,l2con2]) 
l2con3 = layers.Conv2D(128,3,activation='relu',padding ='same')(merge3)
l2con4 = layers.Conv2D(128,3,activation='relu',padding ='same')(l2con3)

merge4 = layers.concatenate([l2con4,l2con2]) 
l3con1 = layers.Conv2D(256,3,activation='relu',padding ='same')(merge4)
l3con2 = layers.Conv2D(256,3,activation='relu',padding ='same')(l3con1)

merge5 = layers.concatenate([l3con2,l2con4]) 
l3con3 = layers.Conv2D(256,3,activation='relu',padding ='same')(merge5)
l3con4 = layers.Conv2D(256,3,activation='relu',padding ='same')(l3con3)

merge6 = layers.concatenate([l3con2,l3con4]) 
l4con1 = layers.Conv2D(512,3,activation='relu',padding ='same')(merge6)
l4con2 = layers.Conv2D(512,3,activation='relu',padding ='same')(l4con1)

merge7 = layers.concatenate([l3con4,l4con2]) 
l4con3 = layers.Conv2D(256,3,activation='relu',padding ='same')(merge7)
l4con4 = layers.Conv2D(256,3,activation='relu',padding ='same')(l4con3)

fl1 = layers.Flatten()(l4con4)
outputs = layers.Dense(10,activation = 'softmax')(fl1)


model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

keras.utils.plot_model(model, "my_model.png", show_shapes=True)

batch_size = 128
num_classes = 10
epochs = 20

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")

model.save('mnist.h5')
print("Saving the model as mnist.h5")

dmodel = keras.models.load_model('mnist.h5')
dmodel.predict(x_train[4:5])

