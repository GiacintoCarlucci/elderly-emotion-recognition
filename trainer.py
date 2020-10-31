from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D
import tensorflow as tf
import datetime

def create_model(input_shape):
    print('\n--- CREATING MODEL ---\n')
    model = Sequential()

    model.add(Conv2D(16, (7, 7), input_shape=input_shape,activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(64, (3, 3),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(64, (3, 3),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(64, (3, 3),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SeparableConv2D(128, (3, 3),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(128, (3, 3),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(128, (3, 3),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    print('\n')
    model.summary()
    print('\n')

    print('\n--- DONE ---\n')
    return model

def train_model(model, train_batches, validation_batches, epochs):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x=train_batches, validation_data=validation_batches, callbacks=[tensorboard_callback], epochs=epochs, verbose=1)

    return model, history
