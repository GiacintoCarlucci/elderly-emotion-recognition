# custom libraries
from organizer import *
from preprocesser import *
from trainer import *
from analyzer import *

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
split_raw_dataset(70, 20, 10, emotion_labels)

# preprocessing
train_batches, validation_batches, test_batches = preprocess(train_path, validation_path, test_path, emotion_labels, batch_size)

# training
model = create_model(input_shape)
model, history = train_model(model, train_batches, validation_batches, epochs)

# analizing results
get_classification_report(model, validation_batches)
plot_accuracy(history)
plot_loss(history)
plot_confusion_matrix(model,validation_batches)
