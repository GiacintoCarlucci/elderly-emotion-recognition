# OS libraries
import shutil, os

def make_dataset_dirs(dataset_directories, emotion_labels):
    print('\n--- CHECKING DATASET DIRECTORIES ---\n')
    print('for each %s directory' %dataset_directories)
    print('there must be %s subdirectory' %emotion_labels)
    print('\n')
    os.listdir()

    try:
        print('accessing ./data/emotions folder...')
        os.chdir('./data')

        if os.path.isdir('emotions') is False:
            os.mkdir('emotions')
        os.chdir('./emotions')

    except:
        print('error accessing ./data/emotions folder, quitting...')
        quit()

    for directory in dataset_directories:
        for emotion in emotion_labels:
                current_path = directory + '/' + emotion
                if os.path.isdir(current_path) is False:
                    os.makedirs(current_path)
                    print('* created ' + current_path)

    os.chdir('../../')
    print('\n--- DONE ---\n')

def split_raw_dataset(train_precentage, validation_percentage, test_percentage, emotion_labels):
    print('\n--- ORGANIZING RAW DATASET DIRECTORY ---\n')
    print('there will be the following organization for each label:')
    print('\n')
    print('- %s%% for training' %train_precentage)
    print('- %s%% for validation' %validation_percentage)
    print('- %s%% for testing' % test_percentage)
    print('\n')

    try:
        print('accessing ./data/raw_emotions folder...\n')
        os.chdir('data/raw_emotions')
    except:
        print('error accessing ./data/raw_emotions folder, quitting...\n')
        quit()

    print('\ndividing images by chosen percentages... \n')
    for emotion in emotion_labels:
        images = os.listdir("./" + emotion)
        length = len(images)
        emotion_train = int( (train_precentage*length)/100 )
        emotion_validation = int( (validation_percentage*length)/100 )
        emotion_test = int( (test_percentage*length)/100 )

        print('\n%s: \n\t total: %s \n\t train: %s \n\t validation: %s \n\t test: %s'
            %(emotion, length, emotion_train, emotion_validation, emotion_test))

        print('\nmoving to directories...')

        train_images = images[0:emotion_train]
        validation_images = images[emotion_train : emotion_train + emotion_validation]
        test_images = images[emotion_train + emotion_validation : emotion_train + emotion_validation + emotion_test]

        for image in train_images:
            shutil.move('./' + emotion + '/' + image , '../emotions/train/' + emotion)
        for image in validation_images:
            shutil.move('./' + emotion + '/' + image , '../emotions/validation/' + emotion)
        for image in test_images:
            shutil.move('./' + emotion + '/' + image , '../emotions/test/' + emotion)

        print('%s moved.' %emotion)

    os.chdir('../../')
    print('\n--- DONE ---\n')
