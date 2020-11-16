from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D
import tensorflow as tf
import datetime
from keras.models import load_model

def create_model(input_shape, final_nodes):
    print('\n--- CREATING MODEL ---\n')

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(final_nodes, activation='softmax'))

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
