# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 15:39:31 2022

@author: user
"""
import tensorflow as tf

def denoisingAE():
    print("load model~~~~~~~~~~~~~~~~~~~~~~")
    model = tf.keras.models.Sequential([
        #Encoder
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(activation='relu', padding='same', filters = 16, kernel_size = 3),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(activation='relu', padding='same', filters = 8, kernel_size = 3),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(40,activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dense(10,activation='relu'),

        # #Decoder
        # tf.keras.layers.Dense(40,activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dense(392,activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Reshape((7,7,8)),
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        tf.keras.layers.Conv2D(activation='relu', padding='same', filters = 16, kernel_size = 3),
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        tf.keras.layers.Conv2D(activation='sigmoid', padding='same', filters = 1, kernel_size = 3),
    ])
    model.summary()

    return model
