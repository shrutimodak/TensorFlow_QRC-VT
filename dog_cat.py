import os
from random import shuffle
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

#Training Parameters
LR = 1e-3
IMG_SIZE = 50
MODEL_NAME = 'dogs-vs-cats-4-layer-covnet'
TRAIN_DIR = 'C:/Users/modak/Anaconda3/envs/TensorFlow_QRC-VT/train/train'
TEST_DIR = 'C:/Users/modak/Anaconda3/envs/TensorFlow_QRC-VT/test/test'

# Data Preprocessing : Label Images, Convert to Grayscale, Resize and append to training set.

#Label Images : [1,0] = Cat & [0,1] = Dog

def create_label(image_name):
    word_label = image_name.split('.')[-3]
    word_label = word_label.split('/')[-1]
    #print(word_label)
    if word_label == "cat":
        return np.array([1, 0])
    elif word_label == "dog":
        return np.array([0, 1])


# Create a training and Testing List : With Image Data and Image labels.

def create_train_data():
    """Read image as 50x50 and grayscale"""
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)

    # Save processed data
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save('data/train.npy', training_data)

    return training_data

def create_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        test_data.append([np.array(img_data), img_num])
    shuffle(test_data)

    # Save processed data
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save('data/test.npy', test_data)

    return test_data


train_data = create_train_data()
#if training data already exists: Then Load the npy file.
#train_data = np.load('train_data.npy')

test_data = create_test_data()
# OR test_data = np.load('test_data.npy')

# Split the training data set into training samples and validation samples. ( You could take a 80-90% of total training set for training and remaining for sampling.
train = train_data[:-1000]
test = train_data[-1000:]

## Build the model
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

#Fully Connected Output Layer
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


#Assign Labels and Features for the model

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_test = [i[1] for i in test]

#Train the Model

if os.path.exists('C:/Users/H/Desktop/KaggleDogsvsCats/{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)

model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=3,
          validation_set=({'input': X_test}, {'targets': Y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)


# Check Results Visually: (If running on Jupyter notebook or load training and testing data and the model before executing this code)

fig=plt.figure(figsize=(16, 16))

for num, data in enumerate(test_data[25:]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(5, 5, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.savefig('%s-test-25.png' % MODEL_NAME)