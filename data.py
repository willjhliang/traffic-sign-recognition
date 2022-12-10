
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import itertools

from constants import *


def load_data(datapath):
    """Loads images from files and performs basic pre-processing."""
    data = {}
    for k in range(K):
        data[k] = []

    for f in os.listdir(datapath):
        k = int(f[:3])  # Get label from filename
        img = Image.open(os.path.join(datapath, f)).convert('RGB')
        img = np.asarray(img) / 255  # Set pixel values to [0, 1]
        if len(data[k]) < class_size:
            data[k].append(img)
    for k in range(K):
        random.shuffle(data[k])
    
    return data


def consolidate_data(data):
    """Converts image-label data from map to numpy arrays."""
    X = []
    y = []
    for k in range(K):
        for i in data[k]:
            i = cv2.resize(i, (S, S))
            X.append(np.swapaxes(i, 0, -1))
            y.append(k)
    X = np.array(X)
    y = np.array(y)
    
    shuffled_indices = np.random.permutation(len(X))
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    X_flattened = np.reshape(X, (X.shape[0], -1))
    
    return X, X_flattened, y


def split_validation(X_train, y_train):
    """Splits training data into train and validation sets. Used in models below."""
    val_split = int(X_train.shape[0] * validation_ratio)
    X_train, X_val = X_train[val_split:], X_train[:val_split]
    y_train, y_val = y_train[val_split:], y_train[:val_split]
    return X_train, X_val, y_train, y_val


def visualize_data(train_data):
    """Visualizes the first image in each class."""
    _, axs = plt.subplots(4, 10, figsize=(10, 3))
    for k, (i, j) in itertools.zip_longest(range(K), list(itertools.product(range(4), range(10))), fillvalue=-1):
        axs[i, j].axis('off')
        if k >= 0:
            axs[i,j].imshow(train_data[k][0])


def compare_class_dist(data_1, data_2):
    """Compares the class distribution of two datasets."""
    class_dist_1 = [len(data_1[k]) for k in range(K)]
    class_dist_2 = [len(data_2[k]) for k in range(K)]

    _, axs = plt.subplots(1, 2, figsize=(8, 2.5))
    axs[0].bar(list(range(K)), class_dist_1)
    axs[1].bar(list(range(K)), class_dist_2);