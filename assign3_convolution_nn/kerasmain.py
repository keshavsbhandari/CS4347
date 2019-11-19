from __future__ import print_function
from configs import train_path, test_path
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

tensorboard_callback = keras.callbacks.TensorBoard(log_dir='foodsmallruns')

batch_size = 10
num_classes = 2
epochs = 10
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'archive')
model_name = 'kerasfoodsmall.h5'

model_path = os.path.join(save_dir, model_name)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=10,
    class_mode='categorical',
    shuffle=True,
    seed = 1)

test_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(128, 128),
        batch_size=10,
        class_mode='categorical')


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(128,128,3)))

model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

if os.path.exists(model_path):
    print("LOADING OLD MODEL")
    model.load_weights(model_path)

model.compile(loss= 'categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



model.fit_generator(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    callbacks=[tensorboard_callback])


model.save(model_path)
