import matplotlib.pyplot as plt
import numpy as np
import os

from six.moves import cPickle as pickle
import cv2

dirs = [r'D:\pythonproject\DrowsyDriverDetection-master\Dataset\eyes\open\\']
dirs2 = [r'D:\pythonproject\DrowsyDriverDetection-master\Dataset\eyes\close\\']

# import glob
# file_list = glob.glob(os.path.join(r'D:\pythonproject\DrowsyDriverDetection-master\Dataset\eyes\open\\', '*'))
# print(len(file_list))

def generate_dataset():
    dataset = np.ndarray([10000, 50, 50, 1], dtype='float32')
    i = 0
    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                im = cv2.imread(dir + '/' + filename)
                im = cv2.resize(im, (50,50))
                im = np.dot(np.array(im, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
                dataset[i, :, :, :] = im[:, :, :]
                i += 1
    labels = np.ones([len(dataset), 1], dtype=int)
    return dataset, labels


def generate_dataset_closed():
    dataset = np.ndarray([10000, 50, 50, 1], dtype='float32')
    i = 0
    for dir in dirs2:
        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                im = cv2.imread(dir + '/' + filename)
                im = cv2.resize(im, (50,50))
                im = np.dot(np.array(im, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
                dataset[i, :, :, :] = im[:, :, :]
                i += 1
    labels = np.zeros([len(dataset), 1], dtype=int)
    return dataset, labels


dataset_open, labels_open = generate_dataset()
dataset_closed, labels_closed = generate_dataset_closed()
print("done")

split = int(len(dataset_closed) * 0.8)
train_dataset_closed = dataset_closed[:split]
train_labels_closed = labels_closed[:split]
test_dataset_closed = dataset_closed[split:]
test_labels_closed = labels_closed[split:]

pickle_file = 'closed_eyes.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset_closed,
        'train_labels': train_labels_closed,
        'test_dataset': test_dataset_closed,
        'test_labels': test_labels_closed,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

split = int(len(dataset_open) * 0.8)
train_dataset_open = dataset_open[:split]
train_labels_open = labels_open[:split]
test_dataset_open = dataset_open[split:]
test_labels_open = labels_open[split:]

pickle_file = 'open_eyes.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset_open,
        'train_labels': train_labels_open,
        'test_dataset': test_dataset_open,
        'test_labels': test_labels_open,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
