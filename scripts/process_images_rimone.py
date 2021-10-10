from dual_IDG import DualImageDataGenerator
import numpy as np
import skimage
import skimage.transform
import skimage.exposure
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import h5py
import os
from skimage.color import rgb2hsv
from skimage.color import rgb2lab

h5f = h5py.File(os.path.join(os.path.dirname(os.getcwd()), 'data', 'hdf5_datasets', 'RIM_ONE_V3.hdf5'), 'r')


def get_images(suf):
    h5f = h5py.File(os.path.join(os.path.dirname(os.getcwd()), 'data', 'hdf5_datasets', 'RIM_ONE_V3.hdf5'), 'r')
    images = h5f['RIM-ONE v3/512 px/img_cropped'+suf]
    cups = h5f['RIM-ONE v3/512 px/cup_cropped'+suf]
    discs = h5f['RIM-ONE v3/512 px/disc_cropped'+suf]
    
    return images, cups, discs

def get_images_and_labels(suf):
    h5f = h5py.File(os.path.join(os.path.dirname(os.getcwd()), 'data', 'hdf5_datasets', 'RIM_ONE_V3.hdf5'), 'r')
    images = h5f['RIM-ONE v3/512 px/img_cropped'+suf]
    cups = h5f['RIM-ONE v3/512 px/cup_cropped'+suf]
    discs = h5f['RIM-ONE v3/512 px/disc_cropped'+suf]
    labels = np.array(h5f['RIM-ONE v3/512 px/is_ill'])
    return images, cups, discs, labels

train_idg = DualImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                   width_shift_range=0.05, height_shift_range=0.05,
                                   zoom_range=(0.8, 1.2),
                                   fill_mode='constant', cval=0.0)
test_idg = DualImageDataGenerator()

def convert_to_hsv_color(images):
    img_channel = []
    for i in (images):
        img_channel.append(rgb2hsv(i))
    return img_channel

def convert_to_lab_color(images):
    img_channel = []
    for i in (images):
        img_channel.append(rgb2lab(i))
    return img_channel

def convert_to_gray(images):
    img_channel = []
    for i in (images):
        gray = cv2.cvtColor(np.array(i, dtype=np.float32), cv2.COLOR_BGR2GRAY)
        img = np.zeros(i.shape)
        img[:,:,0] = gray
        img[:,:,1] = gray
        img[:,:,2] = gray
        
        img_channel.append(img)
    return img_channel


def convert_to_hsv(channel, images):
    img_channel = []
    for i in (images):
        hsv_img = rgb2hsv(i)
        img = np.zeros(hsv_img.shape)
        img[:,:,channel] = hsv_img[:,:,channel]
        img_channel.append(img)
    return img_channel

def convert_to_lab(channel, images):
    img_channel = []
    for i in (images):
        hsv_img = rgb2lab(i)
        img = np.zeros(hsv_img.shape)
        img[:,:,channel] = hsv_img[:,:,channel]
        img_channel.append(img)
    return img_channel

			
def get_color_channel(channel, images):
    img_channel = []
    for i in (images):
        img = np.zeros(i.shape)
        img[:,:,channel] = i[:,:,channel]
        img_channel.append(img)
    return img_channel

			
def preprocess(batch_X, batch_y, train_or_test='train'):    
    
    batch_X = batch_X / 255.0
    
    if train_or_test == 'train':
        batch_X, batch_y = next(train_idg.flow(batch_X, batch_y, batch_size=len(batch_X), shuffle=False))
    elif train_or_test == 'test':
        batch_X, batch_y = next(test_idg.flow(batch_X, batch_y, batch_size=len(batch_X), shuffle=False))
    batch_X = [skimage.exposure.equalize_adapthist(batch_X[i])
               for i in range(len(batch_X))]
    batch_X = np.array(batch_X)
    return batch_X, batch_y

            
def data_generator(X, y, train_idx, test_idx, resize_to=128, train_or_test='train', batch_size=3, return_orig=False, stationary=False):
    
    while True:
        if train_or_test == 'train':
            idx = np.random.choice(train_idx, size=batch_size)
        elif train_or_test == 'test':
            if stationary:
                idx = test_idx[:batch_size]
            else:
                idx = np.random.choice(test_idx, size=batch_size)
                
        batch_X = [skimage.transform.resize(X[i], (resize_to, resize_to)) for i in idx]
        batch_X = np.array(batch_X).copy()
        
        batch_y = [skimage.transform.resize(y[i], (resize_to, resize_to)) for i in idx]
        batch_y = np.array(batch_y).copy()
        
        if return_orig:
            batch_X_orig, batch_Y_orig = batch_X.copy(), batch_y.copy()
        
        batch_X, batch_y = preprocess(batch_X, batch_y, train_or_test)
        
        if not return_orig:
            yield batch_X, batch_y
        else:
            yield batch_X, batch_y, batch_X_orig, batch_Y_orig
            

def create_all_color_list(images, cups, discs, idx_list):
    
    allImages = []
    allCups = []
    allDiscs = []

    aux_images = []
    aux_cups = []
    aux_discs = []
    
    for idx in range(len(idx_list)):
        i = idx_list[idx]
        aux_images.append(images[i])
        aux_cups.append(cups[i])
        aux_discs.append(discs[i])
    
    allImages.extend(aux_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)
    
    hsv_images = convert_to_hsv_color(aux_images)

    allImages.extend(hsv_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    hsv_images = convert_to_hsv(0, aux_images)

    allImages.extend(hsv_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    hsv_images = convert_to_hsv(1, aux_images)

    allImages.extend(hsv_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    hsv_images = convert_to_hsv(2, aux_images)

    allImages.extend(hsv_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    lab_images = convert_to_lab_color(aux_images)

    allImages.extend(lab_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    lab_images = convert_to_lab(0, aux_images)

    allImages.extend(lab_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    lab_images = convert_to_lab(1, aux_images)

    allImages.extend(lab_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)
    
    lab_images = convert_to_lab(2, aux_images)

    allImages.extend(lab_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)
    
    rgb_images = convert_to_gray(aux_images)

    allImages.extend(rgb_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    rgb_images = get_color_channel(0, aux_images)

    allImages.extend(rgb_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    rgb_images = get_color_channel(1, aux_images)

    allImages.extend(rgb_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    rgb_images = get_color_channel(2, aux_images)

    allImages.extend(rgb_images)
    allCups.extend(aux_cups)
    allDiscs.extend(aux_discs)

    return allImages, allCups, allDiscs

