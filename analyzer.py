import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.models import Model

def get_x_y(model,batches):
    # Predicted values
    test_steps_per_epoch = np.math.ceil(batches.samples / batches.batch_size)
    predictions = model.predict(batches, steps=test_steps_per_epoch)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = batches.classes
    class_labels = list(batches.class_indices.keys())

    return true_classes, predicted_classes

def get_classification_report(model, validation_batches):
    print('\n--- PRINTING CLASSIFICATION REPORT ---\n')

    x,y = get_x_y(model,validation_batches)
    class_labels = list(validation_batches.class_indices.keys())
    report = classification_report(x, y, target_names=class_labels)

    print(report)
    print('\n--- DONE ---\n')

# Plot training & validation accuracy values
def plot_accuracy(history):
    print('\n--- PRINTING ACCURACY GRAPH ---\n')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    print('\n--- DONE ---\n')

# Plot training & validation loss values
def plot_loss(history):
    print('\n--- PRINTING LOSS GRAPH ---\n')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    print('\n--- DONE ---\n')

def plot_confusion_matrix(model,validation_batches):
    print('\n--- PRINTING CONFUSION MATRIX ---\n')
    x,y = get_x_y(model,validation_batches)

    cm = confusion_matrix(x, y)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    class_labels = list(validation_batches.class_indices.keys())

    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    print('\n--- DONE ---\n')

def display_activation(activations, row_size, col_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
    plt.show()
