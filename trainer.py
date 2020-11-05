from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D
import tensorflow as tf
import datetime

def create_model(input_shape, final_nodes):
    print('\n--- CREATING MODEL ---\n')

    vgg = VGG16(input_shape=[224,224] + [3], weights='imagenet', include_top=False)

    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    prediction = Dense(final_nodes, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)

    # model = Sequential()
    #
    # model.add(Conv2D(32, (5,5), activation="relu", input_shape=input_shape, padding='valid'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D())
    # # model.add(Conv2D(64, (5,5), activation="relu"))
    # # model.add(BatchNormalization())
    # # model.add(Conv2D(128, (3,3), activation="relu"))
    # # model.add(BatchNormalization())
    # # model.add(MaxPooling2D())
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(final_nodes))
    # model.add(Activation('softmax'))


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
