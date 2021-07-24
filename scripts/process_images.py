from dual_IDG import DualImageDataGenerator
import numpy as np
from tensorflow.keras import backend as K
import skimage
import skimage.transform
import skimage.exposure
import cv2

train_idx = np.arange(0, 50)
test_idx  = np.arange(0, 51)
K.set_image_data_format('channels_last')

train_idg = DualImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                   rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=(0.8, 1.2),
                                   fill_mode='constant', cval=0.0)
test_idg = DualImageDataGenerator()

def convert_to_hsv(channel, images):
    img_channel = []
    for i in (images):
        hsv_img = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
        img = np.zeros(hsv_img.shape)
        img[:,:,channel] = hsv_img[:,:,channel]
        img_channel.append(img)
    return img_channel

def convert_to_lab(channel, images):
    img_channel = []
    for i in (images):
        hsv_img = cv2.cvtColor(i, cv2.COLOR_BGR2LAB)
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
    # the following line thresholds segmentation mask for DRISHTI-GS, since it contains averaged soft maps:
    batch_y = batch_y >= 0.5
    
    if train_or_test == 'train':
        batch_X, batch_y = next(train_idg.flow(batch_X, batch_y, batch_size=len(batch_X), shuffle=False))
    elif train_or_test == 'test':
        batch_X, batch_y = next(test_idg.flow(batch_X, batch_y, batch_size=len(batch_X), shuffle=False))
    batch_X = [skimage.exposure.equalize_adapthist(batch_X[i])
               for i in range(len(batch_X))]
    batch_X = np.array(batch_X)
    return batch_X, batch_y


def data_generator(X, y, disc_locations, resize_to=128, train_or_test='train', batch_size=3, return_orig=False, stationary=False):
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
                
        batch_X = [X[i][disc_locations[i][0]:disc_locations[i][2], disc_locations[i][1]:disc_locations[i][3]] 
                   for i in idx]
        batch_X = [np.rollaxis(img, 2) for img in batch_X]

        batch_X = [skimage.transform.resize(np.rollaxis(img, 0, 3), (resize_to, resize_to))
                   for img in batch_X]
        batch_X = np.array(batch_X).copy()
        
        
        batch_y = [y[i][disc_locations[i][0]:disc_locations[i][2], disc_locations[i][1]:disc_locations[i][3]] 
                   for i in idx]
        batch_y = [img[..., 0] for img in batch_y]
        batch_y = [skimage.transform.resize(img, (resize_to, resize_to))[..., None] for img in batch_y]
        batch_y = np.array(batch_y).copy()
                
        if return_orig:
            batch_X_orig, batch_Y_orig = batch_X.copy(), batch_y.copy()
                
        batch_X, batch_y = preprocess(batch_X, batch_y, train_or_test)
                        
        if not return_orig:
            yield batch_X, batch_y
        else:
            yield batch_X, batch_y, batch_X_orig, batch_Y_orig


			
