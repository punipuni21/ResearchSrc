# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:29:49 2020

@author: ShimaLab
"""
from PIL import Image
import os
import numpy as np
from tkinter import filedialog
from time import sleep
import sys


if __name__ == '__main__':
    print("リサイズ後の大きさ 何も入力しない場合400x400になります\n入力例>>>（高さ　幅）： 400 400")
    try:
        h,w = (int(i) for i in input().split())
        resize = (h,w)
    except ValueError:
        (h,w) = resize = (480, 480)
        #print(resize)
    print("読込ファイル")
    path_pwd = os.path.abspath(os.path.dirname(__file__))
    file_read = filedialog.askopenfilename(title="読込ファイルの選択",filetypes=[("",'bmp'),("",'jpg'),("",'png')],initialdir=path_pwd,multiple=True)
    
    
    if file_read == "":
        sys.exit(1)
    
    dir_write = file_read[0].rsplit('/', 1)[0]
    
    
    if dir_write[-1] != "/":
        dir_write = dir_write + "/"
    if not os.path.exists(dir_write+"npy_"+str(h)+'_'+str(w)):
        os.mkdir(dir_write+"npy_"+str(h)+'_'+str(w))
    dir_write+="npy_"+str(h)+'_'+str(w)+"/"


    
    print("{}に出力".format(dir_write))
    
    
    (h,w) = resize
    print(h," ",w)
    img_np = np.zeros((len(file_read),h,w,3),dtype=np.uint8)
    print(img_np.shape)
    print("Resize中")
    print("the number of images is",len(file_read))
    for i,file in enumerate(file_read):
        if(i%10==0):
            print("now processing is ",i)
        img = Image.open(file)
        #img = expand2square(img, (0, 0, 0))正方形にする場合
        img=img.resize((w,h))
#        print(img.size)25
        new_file = dir_write + file.rsplit('/', 1)[1]
        #new_file = dir_write + str(number) + '.jpg'
        #print(new_file)
        #print('file_num=',number)
        
        if img.mode != "RGB":
            img = img.convert("RGB")
#        print(img.size)
#        hoge = np.array(img,'uint8')
#        print(hoge.shape)
        img_np[i,:,:,:] = np.asarray(img,dtype=np.uint8)
#        img.save(new_file)
        
    np.save(file=dir_write+"out.npy",arr=np.uint8(img_np))