#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概要
"""
__author__ = 'Watanabe Ryotaro'
__version__ = '1.0'
__date__    = "2020/06/12 20:02"


from PIL import Image
import os
import numpy as np
from tkinter import filedialog
from time import sleep
import sys

#画像が正方形になるように余白に黒を敷き詰める
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

#画像のリサイズを行う
#リサイズしながらnpyファイルにデータを保存
#画像のサイズはどうする？？
def dir2resized_jpg(resize,file_read,dir_write):
    """
    画像の大きさを(480,480)に変え，画像を保存
    """ 
    (h,w) = resize
    img_np = np.zeros((len(file_read),h,w,3),dtype=np.uint8)
    for i,file in enumerate(file_read):
        img = Image.open(file)
        img = expand2square(img, (0, 0, 0))
        img=img.resize((h,w))
        print(img.size)
        number= int(file.rsplit('/')[-1].strip('.jpg').split('_')[-1])
        #new_file = dir_write + file.rsplit('/', 1)[1]
        new_file = dir_write + str(number) + '.jpg'
        print(new_file)
        print('file_num=',number)
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np[i,:,:,:] = np.asarray(img,dtype=np.uint8)
        img.save(new_file)
        
    np.save(file=dir_write+"out.npy",arr=np.uint8(img_np))
    

if __name__ == '__main__':
    print("リサイズ後の大きさ 何も入力しない場合400x400になります\n入力例>>>400 400")
    try:
        h,w = (int(i) for i in input().split())
        resize = (h,w)
    except ValueError:
        resize = (400, 400)
        print(resize)
    print("読込ファイル")
    path_pwd = os.path.abspath(os.path.dirname(__file__))
    file_read = filedialog.askopenfilename(title="読込ファイルの選択",filetypes=[("",'bmp'),("",'jpg'),("",'png')],initialdir=path_pwd,multiple=True)
    
    
    if file_read == "":
        sys.exit(1)
    
    dir_write = file_read[0].rsplit('/', 1)[0]
    
    
    if dir_write[-1] != "/":
        dir_write = dir_write + "/"
    if not os.path.exists(dir_write+"out"):
        os.mkdir(dir_write+"out")
    dir_write+="out/"


    
    dir2resized_jpg(resize,file_read,dir_write)
    
    print("{}に出力".format(dir_write))
    sleep(3)