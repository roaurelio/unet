from dual_IDG import DualImageDataGenerator
import numpy as np
from tensorflow.keras import backend as K
import skimage
import skimage.transform
import skimage.exposure
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.color import rgb2lab

train_idx = np.arange(0, 50)
test_idx  = np.arange(0, 51)
K.set_image_data_format('channels_last')

train_idg = DualImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                   rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
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
        img_channel.append(rgb2lab(i)/255)
    return img_channel

def convert_to_gray(images):
    img_channel = []
    for i in (images):
        gray =  rgb2gray(i)
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
        lab_img = rgb2lab(i)/255
        img = np.zeros(lab_img.shape)
        img[:,:,channel] = lab_img[:,:,channel]
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
   #batch_X = batch_X / 255.0
    # the following line thresholds segmentation mask for DRISHTI-GS, since it contains averaged soft maps:
   # batch_y = batch_y >= 0.5
    
    if train_or_test == 'train':
        batch_X, batch_y = next(train_idg.flow(batch_X, batch_y, batch_size=len(batch_X), shuffle=False))
    elif train_or_test == 'test':
        batch_X, batch_y = next(test_idg.flow(batch_X, batch_y, batch_size=len(batch_X), shuffle=False))
    batch_X = [skimage.exposure.equalize_adapthist(batch_X[i])
               for i in range(len(batch_X))]
    batch_X = np.array(batch_X)
    return batch_X, batch_y


def data_generator(X, y, resize_to=128, train_or_test='train', batch_size=3, return_orig=False, stationary=False):
    """Gets random batch of data, 
    divides by 255,
    feeds it to DualImageDataGenerator."""
      
    while True:
        if train_or_test == 'train':
            idx = np.random.choice(train_idx, size=batch_size)
        elif train_or_test == 'validate':
            if stationary:
                idx = validate_idx[:batch_size]
            else:
                idx = np.random.choice(validate_idx, size=batch_size)
        elif train_or_test == 'test':
            if stationary:
                idx = test_idx[:batch_size]
            else:
                idx = np.random.choice(test_idx, size=batch_size)
                
        #batch_X = [X[i][disc_locations[i][0]:disc_locations[i][2], disc_locations[i][1]:disc_locations[i][3]] 
        #           for i in idx]
        #batch_X = [np.rollaxis(img, 2) for img in batch_X]

        batch_X = [skimage.transform.resize(X[i], (resize_to, resize_to))
                   for i in idx]
        batch_X = np.array(batch_X).copy()
        
        
        #batch_y = [y[i][disc_locations[i][0]:disc_locations[i][2], disc_locations[i][1]:disc_locations[i][3]] 
        #           for i in idx]
        #batch_y = [y[i][..., 0] for i in idx]

        batch_y = [skimage.transform.resize(y[i], (resize_to, resize_to))[..., None] for i in idx]
        batch_y = np.array(batch_y).copy()
                
        if return_orig:
            batch_X_orig, batch_Y_orig = batch_X.copy(), batch_y.copy()
                
        batch_X, batch_y = preprocess(batch_X, batch_y, train_or_test)
                        
        if not return_orig:
            yield batch_X, batch_y
        else:
            yield batch_X, batch_y, batch_X_orig, batch_Y_orig


			
def create_all_color_list(images, cups, discs):
    
    allImages = []
    allCups = []
    allDiscs = []
    
    print(len(images))
    
    allImages.extend(images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    hsv_images = convert_to_hsv_color(images)

    allImages.extend(hsv_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    hsv_images = convert_to_hsv(0, images)

    allImages.extend(hsv_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    hsv_images = convert_to_hsv(1, images)

    allImages.extend(hsv_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    hsv_images = convert_to_hsv(2, images)

    allImages.extend(hsv_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    lab_images = convert_to_lab_color(images)

    allImages.extend(lab_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    lab_images = convert_to_lab(0, images)

    allImages.extend(lab_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    lab_images = convert_to_lab(1, images)

    allImages.extend(lab_images)
    allCups.extend(cups)
    allDiscs.extend(discs)
    lab_images = convert_to_lab(2, images)

    allImages.extend(lab_images)
    allCups.extend(cups)
    allDiscs.extend(discs)
    
    rgb_images = convert_to_gray(images)

    allImages.extend(rgb_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    rgb_images = get_color_channel(0, images)

    allImages.extend(rgb_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    rgb_images = get_color_channel(1, images)

    allImages.extend(rgb_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    rgb_images = get_color_channel(2, images)

    allImages.extend(rgb_images)
    allCups.extend(cups)
    allDiscs.extend(discs)

    return allImages, allCups, allDiscs
