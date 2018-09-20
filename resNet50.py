#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:34:32 2018

@author: dmorton
"""

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

BATCH = 50
inputSize = 224'

## Data Augmentation.
trainGen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input)


## Don't augment test data.
testgen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

## Home directory path
path = './catdog/'

## Set training batches
trainBatches = trainGen.flow_from_directory(path+'train', target_size=(inputSize, inputSize),
                class_mode='categorical', shuffle=True, batch_size=BATCH)

## Set validation batches
valBatches = testgen.flow_from_directory(path+'valid', target_size=(inputSize, inputSize),
                class_mode='categorical', shuffle=False, batch_size=2 * BATCH)

#%%

## Load base ResNet50 model and freeze all the layers.
    
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False
#%%
    
## Add pooling and output layers.    
    
add_model = base_model.output
add_model = AveragePooling2D(pool_size=(7,7))(add_model)

add_model = Flatten()(add_model)
add_model = Dense(2, activation='sigmoid')(add_model)

model = Model(inputs=base_model.input, outputs=add_model)

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

#%%

## Pretrain final layer.

best_model_file = path + 'Res50-224x224.h5'

callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1, 
                               verbose=1, min_lr=1e-7),
             ModelCheckpoint(filepath=best_model_file, verbose=1,
                             save_best_only=True, save_weights_only=True, mode='auto')]



model.fit_generator(trainBatches, epochs=40, 
                    validation_data=valBatches,
                    callbacks=callbacks,
                    #workers=4,
                    verbose=1)

#%%

## Make all layers trainable and add relularization.

model.load_weights(best_model_file + '.hdf5')
for layer in model.layers:
    layer.W_regularizer = l2(1e-2)
    layer.trainable = True



model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1, 
                               verbose=1, min_lr=1e-7),
             ModelCheckpoint(filepath=best_model_file, verbose=1,
                             save_best_only=True, save_weights_only=True, mode='auto')]
#%%

## Train full Model.

model.fit_generator(trainBatches, epochs=40, 
                    validation_data=valBatches,
                    callbacks=callbacks,
                    #workers=4,
                    verbose=1)

#%%

## Build predictions from test data.

testBatches = testgen.flow_from_directory(path+'test', target_size=(inputSize, inputSize),
                shuffle=False, batch_size=2 * BATCH)

predictions = model.predict_generator(testBatches)

## Save predictions
pd.DataFrame(predictions, index = test_id).to_csv('./catdog/pred.csv', header = ['label'], index_label = 'id')