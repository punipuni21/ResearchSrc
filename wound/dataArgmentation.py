# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:00:52 2020

@author: ShimaLab
"""

__author__ = 'Watanabe Ryotaro'
__version__ = '1.0'
__date__    = "2020/06/17 17:01"


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from glob import glob
import numpy as np
import os
import os.path
from tkinter import filedialog
from PIL import Image
import cv2


def augment(tra_imgs,times,condition,seed):
    """
    画像群が格納された4次元配列を基にデータ拡張する．
    
    """
    (n,h,w,c) =tra_imgs.shape
    aug_tra_img = np.zeros((n*times,h,w,c))
    
    idg = ImageDataGenerator(**condition)
    idgf=idg.flow(x=tra_imgs, seed=seed, batch_size=tra_imgs.shape[0], save_to_dir=None, save_prefix='', save_format='jpg')
    for i in range(times):
        print("{}/{}回目".format(i+1,times))
        aug_tra_img[i*n:(i+1)*n,:,:,:] = idgf.next()
    return aug_tra_img

#def write_condition(condition,file_read,dir_write):
#    """
#    拡張条件をサンプル画像で確認する
#    """
#    print("拡張条件毎の変化の出力開始")
#    no_condition = dict(
#            rotation_range=0,
#            width_shift_range=0,
#            height_shift_range=0,
#            shear_range=0,
#            zoom_range=[1,1],
#            horizontal_flip=False,
#            vertical_flip=False)
#    img = np.array(Image.open(file_read).resize((100,100))).reshape((1,100,100,3))
#    
#    for k,v in condition.items():
#        condition_copy = no_condition.copy()
#        condition_copy[k] = v
#        idg = ImageDataGenerator(**condition_copy)
#        idgf = idg.flow(img, seed=random.randint(1,100), batch_size=1, save_to_dir=None, save_prefix='', save_format='jpg')
#        row_imgs =[]
#        for r in range(3):
#            colum_imgs =[]
#            for c in range(3):
#                one_img = np.ones((102,102,3))*255
#                one_img[1:101,1:101,:] = idgf.next().reshape((100,100,3))
#                colum_imgs.append(one_img)
#            row_imgs.append(colum_imgs)
#        Image.fromarray(np.uint8(imedit.ims2np_tile(row_imgs))).save(dir_write + k +".jpg")
#    print("拡張条件毎の変化の出力完了")
#
#def process(times,condition,seed,imgArray,dir_write=None,file_write=None):
#    """
#    データセットの切り分け, 各種データをリサイズして配列化, データ拡張, .npyで保存, .jpgで保存 の五工程を記述
#    """
#    print("開始({}倍)".format(times))
#    
##    #1 データセットの切り分け
##    
##    #2 各種データをリサイズして配列化
##    tra_imgs = imedit.files2array_resized(tra_files,resize)
##    val_imgs = imedit.files2array_resized(val_files,resize)
##    tes_imgs = imedit.files2array_resized(tes_files,resize)
##    
#
#    #3 データ拡張
#    tra_outs = augment(tra_imgs,times,condition,seed)
#    
#    #4 .npyで保存
#    if file_write != None:
#        (tra_npy_file,val_npy_file,tes_npy_file) = file_write
#        np.save(tra_npy_file,np.uint8(tra_outs))
#        np.save(val_npy_file,np.uint8(val_imgs))
#        np.save(tes_npy_file,np.uint8(tes_imgs))
#    
#    #5 .jpgで保存
#    if dir_write != None:
#        for i,tra_out in enumerate(tra_outs):
#            new_file = dir_write + "tra_"  + str(i) + ".jpg"
#            Image.fromarray(np.uint8(tra_out)).save(new_file)
#            
#        for i,val_img in enumerate(val_imgs):
#            new_file = dir_write + "val_"  + str(i) + ".jpg"
#            Image.fromarray(np.uint8(val_img)).save(new_file)
#            
#        for i,tes_img in enumerate(tes_imgs):
#            new_file = dir_write + "tes_"  + str(i) + ".jpg"
#            Image.fromarray(np.uint8(tes_img)).save(new_file)
#        
#    print("完了")




def makeAugmentedImage(file_read,times,condition,seed):
    """
    入力画像とラベル画像の両方に対してデータ拡張
    """
    currentDir = os.path.abspath(os.path.dirname(__file__))
    for fileName in file_read:
        #入力画像に関してデータ拡張
            
        npyArray=np.load(fileName)#データ拡張前のデータ
#        dataType=fileName.split('/')[-2]#Train or Val or Test
        processingFileName=fileName.split('/')[-1].strip('.npy')
#        print(hog)
#        break
        print('now processing is ',processingFileName)
        ##出力用ディレクトリと保存用ファイル名の作成
        dirFileWrite=currentDir
        if not os.path.exists(currentDir+"\\dataAugment"):
            os.mkdir(currentDir+"\\dataAugment")
        #npzファイルの保存場所
        dirNpyFileWrite = dirFileWrite + '\\dataAugment\\' + processingFileName + 'Aug.npy'
            
        #Augment後の配列を保存
        npyAugArray = augment(npyArray,times,condition,seed).astype('uint8')
        np.save(dirNpyFileWrite,npyAugArray)
    
        

    
    
if __name__ == '__main__':
    
    print("読込ファイル")
    path_pwd = os.path.abspath(os.path.dirname(__file__))
    file_read = filedialog.askopenfilename(title="読込ファイルの選択",filetypes=[("",'bmp'),("",'jpg'),("",'png')],initialdir=path_pwd,multiple=True)
    
    times = 2
    condition = dict(
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=20,
            zoom_range=[0.8,1.2],
            horizontal_flip=False,
            vertical_flip=False)
    
    seed = 89
#    print("seed:",seed)
#    print("condition:",condition)
    makeAugmentedImage(file_read,times,condition,seed)
    
    print("Done !")

#    シード値どうする？
#    画像のタイル保存はしていない
    
    
    