# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:55:02 2020

@author: ShimaLab
"""

import os, sys
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from PIL import Image


# 画像を拡張する関数
def draw_images(generator, x, dir_name, index):
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir,
                       save_prefix=save_name, save_format='jpg',seed=89)

    # 1つの入力画像から何枚拡張するかを指定（今回は50枚）
    for i in range(10):
        bach = g.next()


if __name__ == "__main__":
    

    root_dir = "."
        
    
    conditions = dict(
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=20,
            zoom_range=[0.8,1.2],
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest')
    
    # ImageDataGeneratorを定義
    datagen = ImageDataGenerator(**conditions)
    seed = 89
    
        
     #カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
    image_dir = root_dir + "/train/"
    print(image_dir)
    files = glob.glob(image_dir + "/*.jpg")
    
    for idx,file in enumerate(files):
        
        img = load_img(file)
        imgname = file.split('\\')[-1]
        output_dir = root_dir+'/traaug/'
        
        img = np.expand_dims(img, axis=0)
        draw_images(datagen, img, output_dir, idx)
              
            
            