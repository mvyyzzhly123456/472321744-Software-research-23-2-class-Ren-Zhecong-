
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.utils import plot_model
import cv2

np.random.seed(1337)  # for reproducibility



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad

from six.moves import cPickle as pickle

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


class eyesCNN():
    def __init__(self):
        pass

    def load_data(self, pickle_files):
        train_dataset = None
        train_labels = None
        test_dataset = None
        test_labels = None
        for i, pickle_file in enumerate(pickle_files):
            with open(pickle_file, 'rb') as f:
                save = pickle.load(f)
                if i == 0:
                    train_dataset = save['train_dataset']
                    train_labels = save['train_labels']
                    test_dataset = save['test_dataset']
                    test_labels = save['test_labels']
                else:
                    train_dataset = np.concatenate((train_dataset, save['train_dataset']))
                    train_labels = np.concatenate((train_labels, save['train_labels']))
                    test_dataset = np.concatenate((test_dataset, save['test_dataset']))
                    test_labels = np.concatenate((test_labels, save['test_labels']))
                del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

        X_train = train_dataset
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
        Y_train = train_labels
        X_test = test_dataset
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
        Y_test = test_labels
        return X_train, Y_train, X_test, Y_test

    def build_model(self, nb_classes, input_shape):
        model = Sequential()
        model.add(Convolution2D(32, (5, 5), padding='same', input_shape=input_shape,
                                data_format='channels_first'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Convolution2D(64, (5, 3), data_format='channels_first'), )
        #model.add(Activation('relu'))
        #model.add(Dropout(0.25))

        model.add(Convolution2D(64, (3, 3), padding='same', data_format='channels_first'))
        #model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dropout(0.5))
        #model.add(Dense(512))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('sigmoid'))
        return model


    def train(self, model, X_train, Y_train, X_test, Y_test, batch_size, epochs):
        sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test))
        model.save('eyesCNN.h5')

    def evaluate(self, X_test, Y_test, model):
        score = model.evaluate(X_test, Y_test, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

def train():
    cnn = eyesCNN()
    pickle_files = ['open_eyes.pickle', 'closed_eyes.pickle']
    train_dataset, train_labels, test_dataset, test_labels = cnn.load_data(pickle_files)
    _, img_channels, img_rows, img_cols = train_dataset.shape
    input_shape = (img_channels, img_rows, img_cols)
    nb_classes = 1
    model = cnn.build_model(nb_classes, input_shape)
    batch_size = 1000
    epochs = 3000
    cnn.train(model, train_dataset, train_labels, test_dataset, test_labels, batch_size, epochs)
    score = cnn.evaluate(test_dataset, test_labels, model)

def predict(model, x):
    score = model.predict(x, batch_size=1, verbose=1)
    print('score:', score)
    threshold = 0.5  # 设置阈值
    prediction = 'OPEN' if score >= threshold else 'CLOSE'
    print(prediction)
    return prediction

if __name__ == '__main__':
    # 训练代码
    train()

    # 预测一张图片
    model = load_model('eyesCNN.h5')
    im = cv2.imread('./Dataset/eyes/close/_ (1).png')
    im = cv2.resize(im, (50,50,3))
    im = np.dot(np.array(im, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 3, 1, 2))  # 转换维度顺序
    predict(model, im)