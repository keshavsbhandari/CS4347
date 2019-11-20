from __future__ import print_function
from configs import train_path101 as train_path
from configs import test_path101 as test_path
import keras
from keras.preprocessing.image import ImageDataGenerator




import os

tensorboard_callback = keras.callbacks.TensorBoard(log_dir='food101runs')

batch_size = 10
num_classes = 10
epochs = 10
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'archive')
model_name = 'kerasfood101mobilenet.h5'
model_path = os.path.join(save_dir, model_name)


""""
DATA GENERATOR START FROM HERE
"""

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

""""
DATA GENERATOR ENDS HERE
"""


""""
MODEL BUILDING START FROM HERE
"""

# here we create a model with pretrained mobilenet
# since this mobilenet is designed for 1000 class problem we do not need last layer
# that is the reason why include_top is false, that is we removed top layer, that is last layer
# so we change the input shape to 128,128,3 , to train faster you can make this 64,64,3, but this will cost you accuracy loss

base_model = keras.applications.mobilenet.MobileNet(input_shape= (128,128,3),
                                                    alpha=1.0,
                                                    depth_multiplier=1,
                                                    dropout=1e-3,
                                                    include_top=False,
                                                    weights='imagenet',
                                                    input_tensor=None,
                                                    pooling='max')

# as we have removed the last layer or the top one which is needed for the prediction purpose
# now we have to add few more layers to make it work for our case
# we create an instance model
model = keras.models.Sequential()
# we add above pretrained imagenet model on this model
model.add(base_model)
# we add prediction layer to fit our case (note this layer is not trained this is new one)
model.add(keras.layers.Dense(num_classes))
# now we use activation function
model.add(keras.layers.Activation('softmax'))

# exit()
""""
MODEL BUILDING ENDS HERE
"""

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

if os.path.exists(model_path):
    print("LOADING OLD MODEL")
    model.load_weights(model_path)

model.compile(loss= 'categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



model.fit_generator(train_generator,
                    epochs=1,
                    validation_data=test_generator,
                    callbacks=[tensorboard_callback])


model.save(model_path)
