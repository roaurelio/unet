import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
 
def ellipseFitting(img):
    contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    ellipse = np.zeros(img.shape)
    diametro = []
    for ind, cont in enumerate(contours):
        (x,y),(MA,ma),angle = cv2.fitEllipse(cont)
        diametro.append((MA, ma))
        cv2.ellipse(ellipse,(int(x),int(y)),(int(MA/2), int(ma/2)),angle,0,360,(255,255,255),0)
    return ellipse, diametro

def obtain_ellipse(img):
    try:
        return ellipseFitting(img)
    except:
        kernel = np.ones((10,10),np.uint8)
        return ellipseFitting(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel))

def calculate_cdr(pred_cup, pred_disc, test_idx):
    cdrs = []
    diametros_cup = []
    diametros_disc = []
    for i, img_no in enumerate(test_idx):
        cup = pred_cup[i]
        disc = pred_disc[i]

        c = cv2.Canny(cup.astype(np.uint8), 1,1)
        d = cv2.Canny(disc.astype(np.uint8), 1,1)

        try:
            el_c, diam_c = obtain_ellipse(c)
            el_d, diam_d = obtain_ellipse(d)

            if len(diam_d) > 0 and len(diam_c) > 0:
                diametros_cup.append(max(diam_c))
                diametros_disc.append(max(diam_d))
            
                cdr = max(diam_c)[1]/max(diam_d)[1]
                cdrs.append(cdr)
                print('image #{} - cdr = {}'.format(img_no, cdr)) 
            else:
                diametros_cup.append((0,0))
                diametros_disc.append((0,0))
                cdrs.append(0)
        except ValueError as error:
            print(error)
            cdrs.append(0)
    return cdrs, diametros_cup, diametros_disc


def calculate_area(pred_cup, pred_disc, test_idx):
    areas = []
    for i in test_idx:
        cup = np.array(pred_cup[i], dtype='float').sum()
        disc = np.array(pred_disc[i], dtype='float').sum()
        if (disc > 0):
            areas.append(cup/disc)
        else:
            areas.append(0)
    return areas

def plot_results(result, epochs, arq):
    try:
        epoch = range(1, epochs + 1)
        fig = plt.figure(figsize=(15,6))

        ax = fig.add_subplot(1,3,1)
        ax.plot(epoch, result.history['dice_metric'])
        ax.plot(epoch, result.history['val_dice_metric'])
        ax.set_title('Dice do Modelo')
        ax.set_ylabel('Margem de acertos')
        ax.set_xlabel('Épocas')
        ax.legend(['Train','Val'], loc='upper left')

        ax = fig.add_subplot(1,3,2)
        ax.plot(epoch, result.history['mean_IOU_gpu'])
        ax.plot(epoch, result.history['val_mean_IOU_gpu'])
        ax.set_title('IOU do Modelo')
        ax.set_ylabel('Margem de acertos')
        ax.set_xlabel('Épocas')
        ax.legend(['Train','Val'], loc='upper left')

        ax = fig.add_subplot(1,3,3)
        ax.plot(epoch, result.history['loss'])
        ax.plot(epoch, result.history['val_loss'])
        ax.set_title('Margem de erro')
        ax.set_ylabel('Erros')
        ax.set_xlabel('Épocas')
        ax.legend(['Train','Val'], loc='upper left')
        plt.show()
        res = os.path.join(folder(path_results), '{}'.format(arq+'png'))
        plt.savefig(res, format='png')
    except:
        print('Erro ao gerar gráficos')
        
def root_path(name_folder):
    return folder(os.path.join(os.path.dirname(os.getcwd()), 'results',
                                  '{},{}'.format(datetime.now().strftime('%d.%m,%H-%M'), name_folder)))

def folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name
    
def save_diameters(diametros_cup, diametros_disc, root_path, file_name):
    cup = np.array(diametros_cup)
    disc = np.array(diametros_disc)
    df = pd.DataFrame(data={'cup - dm': cup[:,0], 'cup - dM': cup[:,1], 'disc - dm': disc[:,0], 'disc - dM': disc[:,1]})
    df.to_csv(os.path.join(root_path, file_name+'_diameters.csv'), decimal=',', sep='\t')
    return df
    
def create_table_result(pred_cup, pred_disc, test_idx, root_path, file_name):
    cdrs, diametros_cup, diametros_disc = calculate_cdr(pred_cup, pred_disc, test_idx)
    areas = calculate_area(pred_cup, pred_disc, test_idx)
    df = pd.DataFrame(data = {'cdr': cdrs, 'area': areas})
    df.to_csv(os.path.join(root_path, file_name+'_cdrs.csv'), decimal=',', sep='\t')
    df_diameters = save_diameters(diametros_cup, diametros_disc, root_path, file_name)
    return df, df_diameters


def save_iou_dice(iou, dice, root_path, arq):
    res = os.path.join(root_path, arq)

    with open(res, 'w') as arquivo:
        arquivo.write('iou: ' + str(iou).replace('.', ',')+'\n')
        arquivo.write('dice: ' + str(dice).replace('.', ','))
