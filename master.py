# custom libraries
from organizer import *
from preprocesser import *
from trainer import *
from analyzer import *


from keras.models import Model

# parameters
dataset_directories = ['train', 'validation', 'test']
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

train_path = 'data/emotions/train'
validation_path = 'data/emotions/validation'
test_path = 'data/emotions/test'

input_shape=(96,96,1)

epochs = 50
batch_size = 10


# prepare dataset directories
make_dataset_dirs(dataset_directories, emotion_labels)
split_raw_dataset(80, 15, 5, emotion_labels)

# preprocessing
train_batches, validation_batches, test_batches = preprocess(train_path, validation_path, test_path, emotion_labels, batch_size)

# training
final_nodes = len(emotion_labels)
model = create_model(input_shape, final_nodes)
model, history = train_model(model, train_batches, validation_batches, epochs)

# analizing results
get_classification_report(model, validation_batches)
plot_accuracy(history)
plot_loss(history)
plot_confusion_matrix(model,validation_batches)

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
imgs, labels = next(test_batches)
plt.imshow(imgs[0]);
activations = activation_model.predict(imgs)
# display_activation(activations, 8, 8, 1)
# display_activation(activations, 8, 8, 4)
# display_activation(activations, 8, 16, 6)
