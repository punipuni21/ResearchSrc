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
import sys
import os


def crop_to_square(image):
    size = min(image.size)       
    left, upper = (image.width - size) // 2, (image.height - size) // 2       
    right, bottom = (image.width + size) // 2, (image.height + size) // 2        
    return image.crop((left, upper, right, bottom))


if __name__ == '__main__': 
    
    
    print("読込ファイル")
    path_pwd = os.path.abspath(os.path.dirname(__file__))
    file_read = filedialog.askopenfilename(title="読込ファイルの選択",filetypes=[("",'bmp'),("",'jpg'),("",'png')],initialdir=path_pwd,multiple=True)
    
    
#
    dir_cropped = "whiteLabelCropped/"
#    
    
    if file_read == "":
        sys.exit(1)
    
    dir_write = file_read[0].rsplit('/', 1)[0]
    paths_white = glob.glob(dir_write + "/*")#アノテーションのある画像
    
    
    
    if dir_write[-1] != "/":
        dir_write = dir_write + "/"
    if not os.path.exists(dir_write+"cropped"):
        os.mkdir(dir_write+"cropped")
    
    dir_write+="cropped/"


    
    filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_white))
    
    cropped = list(map(lambda filename: dir_write + filename + ".jpg", filenames))#アノテーションのある画像ファイルたちの保存先
    
    img_np = np.zeros((len(file_read),256,256,3),dtype=np.uint8)
    
    cnt = 0
    for (imgfile,outname) in zip(file_read, cropped):
        if(cnt%50==0):
            print(cnt)
        img = Image.open(imgfile)
        if img.mode == "RGB":
            img = img.convert("P")
        img = img.resize((256,256))
        img = img.convert("RGB")
        img_np[cnt,:,:,:] = np.asarray(img,dtype=np.uint8)
        
        img.save(outname)
        cnt += 1
    
        
    np.save(file=dir_write+"out.npy",arr=np.uint8(img_np))
    
  