# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 02:42:03 2020

@author: ShimaLab
"""



import glob
import os
import numpy as np
from PIL import Image, ImageChops
import cv2
from tkinter import filedialog


def crop_to_square(image):
    size = min(image.size)       
    left, upper = (image.width - size) // 2, (image.height - size) // 2       
    right, bottom = (image.width + size) // 2, (image.height + size) // 2        
    return image.crop((left, upper, right, bottom))


if __name__ == '__main__': 

    path_pwd = os.path.abspath(os.path.dirname(__file__))
    
    
    dir_white = "whiteLabel"
    dir_cropped = "whiteLabelCropped/"
    
    paths_white = glob.glob(dir_white + "/*")#アノテーションのある画像
    

    
    filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_white))
    
    whiteLabel = list(map(lambda filename: "whiteLabel" + "/" + filename + ".jpg", filenames))#アノテーションのある画像ファイルたちの保存先
    
    cnt = 0
    for (imgfile,outname) in zip(whiteLabel,filenames):
        if(cnt%50==0):
            print(cnt)
        img = Image.open(imgfile)
        img = crop_to_square(img)
        img.save(dir_cropped+outname+".jpg")
        cnt += 1
    
    
  