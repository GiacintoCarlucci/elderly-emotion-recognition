# Keras Libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
# Plotting libraries
import matplotlib.pyplot as plt

# plots images in a form of a grid with 1 row and 10 columns
def plot_images(images_array):
    fig, axes = plt.subplots(1,10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_array, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def preprocess(train_path, validation_path, test_path, emotion_labels, batch_size):
    print('\n--- DATASET PREPROCESSING ---\n')
    # preprocessing for each path
    print('* preprocessing train_batches...')
    train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(96,96), classes=emotion_labels, batch_size=batch_size, color_mode='grayscale')
    print('* preprocessing validation_batches...')
    validation_batches = ImageDataGenerator().flow_from_directory(directory=validation_path, target_size=(96,96), classes=emotion_labels, batch_size=batch_size, color_mode='grayscale')
    print('* preprocessing test_batches...')
    test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(96,96), classes=emotion_labels, batch_size=batch_size, shuffle=False, color_mode='grayscale')
    print('\n')
    print('* batching train data...')
    imgs, labels = next(train_batches)
    print('* plotting training batch...')
    plot_images(imgs)
    print('* the labels of the plotted images are:')
    print(labels)
    print('\n--- DONE ---\n')

    return train_batches, validation_batches, test_batches
