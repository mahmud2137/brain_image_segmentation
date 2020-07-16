
from keras.layers import Lambda, Softmax, Conv2DTranspose, Reshape, Input, Dense, ReLU, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate, BatchNormalization
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import tensorflow as tf
# import utils
import json
import pandas as pd
from functools import partial
import h5py
from utils import load_nifti
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_sample_2d(df, n, input_shape, output_shape):
    """
    randomly sample patch images from DataFrame
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing name of image files
    n : int
        number of patches to extract
    input_shape : list
        shape of input patches to extract
    output_shape : list
        shape of output patches to extract
    Returns
    -------
    images : (n, n_channels, input_shape[0], input_shape[1], ...) ndarray
        input patches
    labels : (n, output_shape[0], output_shape[1], ...) ndarray
        label patches
    """
    N = len(df)
    if "weight" in list(df):
        weights = np.asarray(df["weight"])
        weights /= np.sum(weights)
        sub_indices = np.random.choice(N, n, replace=True, p=weights)
    else:
        sub_indices = np.random.choice(N, n, replace=True)
    image_files = df["image"][sub_indices]
    label_files = df["label"][sub_indices]
    images = []
    labels = []

    for image_file, label_file in zip(image_files, label_files):
    # for i in sub_indices:
    #     image_file = df["image"][i]
    #     label_file = df["label"][i]
        image = load_nifti(image_file)
        label = load_nifti(label_file).astype(np.int32)
        mask = np.int32(label > 0)
        slices = [slice(len_ // 2, -len_ // 2) for len_ in input_shape]
        mask[slices] *= 2
        indices = np.where(mask > 1.5)
        i = np.random.choice(len(indices[0]))
        input_slices = [
            slice(index[i] - len_ // 2, index[i] + len_ // 2)
            for index, len_ in zip(indices[:2], input_shape)
        ]
        input_slices.append(slice(indices[2][i],indices[2][i]+1))
        
        output_slices = [
            slice(index[i] - len_ // 2, index[i] + len_ // 2)
            for index, len_ in zip(indices[:2], output_shape)
        ]
        output_slices.append(slice(indices[2][i], indices[2][i]+1))
        image_patch = image[input_slices]
        label_patch = label[output_slices]
        image_patch = np.squeeze(image_patch)

        images.append(image_patch)
        labels.append(label_patch)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels



class NetModule():
    """
    Network Module
    input
    BatchNormalization, ReLU
    Conv 64, 3x3x3
    BatchNormalization, ReLU
    Conv 64, 3x3x3
    output
    """
    def __init__(self):
        self.bnorm1 = BatchNormalization()
        self.conv1 = Conv2D(64,3, padding = 'same', activation = 'relu')
        self.bnorm2 = BatchNormalization()
        self.conv2 = Conv2D(64,3, padding = 'same', activation = 'relu')

    def __call__(self,x):
        x = self.bnorm1(x)
        x = self.conv1(x)
        x = self.bnorm2(x)
        x = self.conv2(x)
        return x

class SegmentNet():
    def __init__(self, in_channels = 1, n_classes=4):
        self.conv1a = Conv2D(32, 3, padding = 'same', activation = 'relu')
        self.bnorm1a = BatchNormalization()
        self.conv1b = Conv2D(32, 3, padding = 'same', activation = 'relu')
        self.bnorm1b = BatchNormalization()
        self.conv1c = Conv2D(64, 3, padding = 'same', activation = 'relu')
        self.netmod1 = NetModule()
        self.netmod2 = NetModule()

        self.bnorm2a = BatchNormalization()
        self.conv2a = Conv2D(64, 3, padding = 'same', activation = 'relu')
        self.netmod3 = NetModule()
        self.netmod4 = NetModule()
        self.c2decov = Conv2DTranspose(64, 3, padding = 'same')
        self.conv2b = Conv2D(n_classes, 3, padding='same', activation = 'softmax')

    def __call__(self, x):
        h = self.conv1a(x)
        h = self.bnorm1a(h)
        h = self.conv1b(h)

        h = self.bnorm1b(h)
        h = self.conv1c(h)
        h = self.netmod1(h)
        h = self.netmod2(h)

        h = self.bnorm2a(h)
        h = self.conv2a(h)
        h = self.netmod3(h)
        h = self.netmod4(h)

        h = self.c2decov(h)
        h = self.conv2b(h)
        return h

if __name__ == '__main__':

    train_file = 'dataset_train.json'
    test_file = 'dataset_test.json'
    with open(train_file) as f:
        datasets_train = json.load(f)
    df_train = pd.DataFrame(datasets_train["data"])
    
    with open(test_file) as f:
        dataset_test = json.load(f)
    df_test = pd.DataFrame(dataset_test["data"])

    X_train, y_train = load_sample_2d(df_train, 3, input_shape=[80,80], output_shape=[80,80])
    print(X_train.shape)
    y_train = to_categorical(y_train)
    input_shape = X_train.shape[1:]
    print(input_shape)
    input_layer = Input(shape=input_shape)
    segnet = SegmentNet()
    seg_model = Model(inputs=input_layer, outputs=segnet(input_layer))
    seg_model.summary()
    seg_model.compile(optimizer='adam', loss='categorical_crossentropy')
    # seg_model.fit(X_train, y_train,
    #                 epochs=100)

    def data_gen(data, batch_size, input_shape=[80,80], output_shape=[80,80]):
        while True: 
            X, y = load_sample_2d(data, batch_size, input_shape, output_shape)
            y = to_categorical(y)
            yield X,y
    batch_size = 30
    hist = seg_model.fit_generator(data_gen(df_train, batch_size), steps_per_epoch=10, epochs=100, verbose = 2)
    seg_model.save_weights("model_weights/model_2d.h5")
