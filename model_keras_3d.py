
from keras.layers import Lambda, Softmax, Conv2DTranspose, Conv3DTranspose, Reshape, Input, Dense, ReLU, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate, BatchNormalization, Conv3D
from keras.layers import ReLU, add
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
from utils import load_nifti, load_sample, image_to_patches, patches_to_image
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
os.environ["CUDA_VISIBLE_DEVICES"]="1"


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
        self.conv1 = Conv3D(64,3, padding = 'same', activation = 'relu')
        self.bnorm2 = BatchNormalization()
        self.conv2 = Conv3D(64,3, padding = 'same', activation = 'relu')

    def __call__(self,x):
        x = self.bnorm1(x)
        x = self.conv1(x)
        x = self.bnorm2(x)
        x = self.conv2(x)
        return x

class SegmentNet():
    def __init__(self, in_channels = 2, n_classes=4):
        self.conv1a = Conv3D(32, 3, padding = 'same', activation = 'relu')
        self.bnorm1a = BatchNormalization()
        self.conv1b = Conv3D(32, 3, padding = 'same', activation = 'relu')
        self.c1decov = Conv3DTranspose(32, 3, padding='same')
        self.c1conv = Conv3D(n_classes, 3, padding='same', activation='softmax')

        self.bnorm1b = BatchNormalization()
        self.conv1c = Conv3D(64, 3, padding = 'same', activation = 'relu')
        self.netmod1 = NetModule()
        self.netmod2 = NetModule()
        self.c2decov = Conv3DTranspose(32, 3, padding='same')
        self.c2conv = Conv3D(n_classes, 3, padding='same', activation='softmax')

        self.bnorm2a = BatchNormalization()
        self.conv2a = Conv3D(64, 3, padding = 'same', activation = 'relu')
        self.netmod3 = NetModule()
        self.netmod4 = NetModule()
        self.c3decov = Conv3DTranspose(64, 3, padding = 'same')
        self.conv2b = Conv3D(n_classes, 3, padding='same', activation = 'softmax')

    def __call__(self, x):
        h = self.conv1a(x)
        h = self.bnorm1a(h)
        h = self.conv1b(h)
        c1 = self.c1decov(h)
        c1 = self.c1conv(c1)


        h = self.bnorm1b(h)
        h = self.conv1c(h)
        h = self.netmod1(h)
        h = self.netmod2(h)
        c2 = self.c2decov(h)
        c2 = self.c2conv(c2)


        h = self.bnorm2a(h)
        h = self.conv2a(h)
        h = self.netmod3(h)
        h = self.netmod4(h)

        c3 = self.c3decov(h)
        c3 = self.conv2b(h)

        c = add([c1,c2,c3])
        return [c1,c2,c3,c]

if __name__ == '__main__':

    train_file = 'dataset_train.json'
    test_file = 'dataset_test.json'
    with open(train_file) as f:
        datasets_train = json.load(f)
    df_train = pd.DataFrame(datasets_train["data"])
    
    with open(test_file) as f:
        dataset_test = json.load(f)
    df_test = pd.DataFrame(dataset_test["data"])

    img = load_nifti(df_train["image"][0])
    lbl = load_nifti(df_train["label"][0])
    patch_shape = [32,32,32]

    img_patches = image_to_patches(img, patch_shape)
    lbl_patches = image_to_patches(lbl, patch_shape)

    img_recon = patches_to_image(img_patches, img_shape=img.shape)



    X_train, y_train = load_sample(df_train, 3, input_shape=[32,32,32], output_shape=[32,32,32])
    print(X_train.shape)
    y_train = to_categorical(y_train)
    input_shape = X_train.shape[1:]
    print(input_shape)
    input_layer = Input(shape=input_shape)
    segnet = SegmentNet()
    seg_model = Model(inputs=input_layer, outputs=segnet(input_layer))
    seg_model.summary()
    plot_model(seg_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    seg_model.compile(optimizer='adam', loss='categorical_crossentropy')
    # seg_model.fit(X_train, y_train,
    #                 epochs=100)

    def data_gen(data, batch_size, input_shape=[32,32,32], output_shape=[32,32,32]):
        while True: 
            X, y = load_sample(data, batch_size, input_shape, output_shape)
            y = to_categorical(y)
            yield X,[y,y,y,y]
    batch_size = 16
    # hist = seg_model.fit_generator(data_gen(df_train, batch_size), steps_per_epoch=10, epochs=100, verbose = 2)
    # seg_model.save_weights("model_weights/model_3d.h5")
    seg_model.load_weights("model_weights/model_3d.h5")

    pred = seg_model.predict(img_patches)[-1]
    pred_arg = np.argmax(pred, axis=4)

    pred_whole_label = patches_to_image(pred_arg, img_shape=img.shape)
    
    plt.figure(figsize=(15,12))
    plt.imshow(pred_whole_label[115,:,:])
    plt.savefig('prediction.jpg')
    plt.show()
    

    plt.figure(figsize=(15,12))
    plt.imshow(img[115,:,:,0])
    plt.savefig('mri_image.jpg')
    plt.show()

    plt.figure(figsize=(15,12))
    plt.imshow(lbl[115,:,:])
    plt.savefig('true_label.jpg')
    plt.show()
    
    y_t = np.argmax(y_train[2], axis=3)
    plt.imshow(y_t[17,:,:])
    plt.show()
