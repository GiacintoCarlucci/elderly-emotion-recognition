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
    train_batches = augmentation().flow_from_directory(directory=train_path, target_size=(96,96), classes=emotion_labels, batch_size=batch_size, color_mode='grayscale')

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

# Creates an augmented dataset rotating, shifting or zooming initial images
# At the start of every epoch, transformations are applied to all the images with random parameters in the specified range.
# At every epoch, augmentation is applied again and again, thus due to the random parameters, unique images are generated.
def augmentation():
    return  ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
