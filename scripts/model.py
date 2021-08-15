from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, \
    Conv2D, MaxPooling2D, ZeroPadding2D, Input, Embedding, \
    Lambda, UpSampling2D, Cropping2D, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
import os
from process_images import *
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from datetime import datetime

NUM_EPOCHS = 500
SPE = 99
IMG_SIZE=128

def create_compile_model(img_size, lr):
    model = get_unet_light(img_rows=img_size, img_cols=img_size)
    model.compile(optimizer=SGD(learning_rate=lr, momentum=0.95),
                  loss=log_dice_loss,
                  metrics=[mean_IOU_gpu, dice_metric])
    return model


def mean_IOU_gpu(X, Y):
    """Computes mean Intersection-over-Union (IOU) for two arrays of binary images.
    Assuming X and Y are of shape (n_images, w, h)."""

    X_fl = K.clip(K.batch_flatten(X), 0., 1.)
    Y_fl = K.clip(K.batch_flatten(Y), 0., 1.)
    X_fl = K.cast(K.greater(X_fl, 0.5), 'float32')
    Y_fl = K.cast(K.greater(Y_fl, 0.5), 'float32')

    intersection = K.sum(X_fl * Y_fl, axis=1)
    union = K.sum(K.maximum(X_fl, Y_fl), axis=1)
    union = K.switch(K.equal(union, 0), K.ones_like(union), union)
    return K.mean(intersection / K.cast(union, 'float32'))

def mean_IOU_gpu_loss(X, Y):
    return -mean_IOU_gpu(X, Y)
	
def dice(y_true, y_pred):

    y_true_f = K.clip(K.batch_flatten(y_true), 0., 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), 0., 1.)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(intersection / union)

def dice_loss(y_true, y_pred):
    return -dice(y_true, y_pred)

def log_dice_loss(y_true, y_pred):
    return -K.log(dice(y_true, y_pred))

def dice_metric(y_true, y_pred):
    """An exact Dice score for binary tensors."""
    y_true_f = K.cast(K.greater(y_true, 0.5), 'float32')
    y_pred_f = K.cast(K.greater(y_pred, 0.5), 'float32')
    return dice(y_true_f, y_pred_f)
	
def get_unet_light(img_rows=256, img_cols=256):
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(conv9)

    return Model(inputs=inputs, outputs=conv10)



def predict(images, img_list, mask_list, model, img_size):
    pred_iou, pred_dice = [], []
    pred_result = []

    for i, img_no in enumerate(test_idx):
        print('image #{}'.format(img_no))
        img = images[img_no]
        batch_X = img_list[i:i + 1]
        batch_z = mask_list[i:i + 1]

        pred = (model.predict(batch_X)[0] > 0.5).astype(np.float64)
        pred = pred.reshape(img_size,img_size,)
        pred_result.append(pred)
        corr = (batch_z)[0, ..., 0]

        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(pred, cmap=plt.cm.Greys_r)
        ax.set_title('Predicted')
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(corr, cmap=plt.cm.Greys_r)
        ax.set_title('Correct')
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow((batch_X)[0])
        ax.set_title('Image')
        plt.show()

        cur_iou = K.eval(mean_IOU_gpu(pred[None, None, ...], corr[None, None, ...]))
        cur_dice = K.eval(dice(pred[None, None, ...], corr[None, None, ...]))
        print('IOU: {}\nDice: {}'.format(cur_iou, cur_dice))
        pred_iou.append(cur_iou)
        pred_dice.append(cur_dice)
        
    return pred_iou, pred_dice, pred_result


arch_name = "OD Cup, U-Net light on DRISHTI-GS 512 px cropped to OD 128 px fold 0, SGD, log_dice loss"
weights_folder_cup = os.path.join(os.path.dirname(os.getcwd()), 'models_weights',
                              '{},{}'.format(datetime.now().strftime('%d.%m,%H-%M'), arch_name))

arch_name = "OD Disc, U-Net light on DRISHTI-GS 512 px cropped to OD 128 px fold 0, SGD, log_dice loss"
weights_folder_disc = os.path.join(os.path.dirname(os.getcwd()), 'models_weights',
                              '{},{}'.format(datetime.now().strftime('%d.%m,%H-%M'), arch_name))

def folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def train(images, masks, disc_locations, path, model, epochs, X_valid, Y_valid, img_size, spe, weights_folder):
    return model.fit(data_generator(images, masks, disc_locations, img_size, train_or_test='train', batch_size=1), 
                              steps_per_epoch=spe,
                              max_queue_size=1,
                              validation_data=(X_valid, Y_valid),
                              epochs=epochs, verbose=1,                              
                              callbacks=[CSVLogger(os.path.join(folder(weights_folder), 'training_log_'+path+'.csv')),
                                         ModelCheckpoint(os.path.join(folder(weights_folder),
                                               'last_checkpoint_'+path+'.hdf5'),
                                               monitor='val_loss', mode='min', save_best_only=True, 
                                               save_weights_only=False, verbose=0)])
    
def train_cup(images, masks, disc_locations, path, model, epochs, X_valid, Y_valid, img_size, spe):
    return train(images, masks, disc_locations, path, model, epochs, X_valid, Y_valid, img_size, spe, weights_folder_cup)
      
def train_disc(images, masks, disc_locations, path, model, epochs, X_valid, Y_valid, img_size, spe):
    return train(images, masks, disc_locations, path, model, epochs, X_valid, Y_valid, img_size, spe, weights_folder_disc)
        