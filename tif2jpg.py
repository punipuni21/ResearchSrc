# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:17:32 2020

@author: ShimaLab
"""

from PIL import Image
import numpy as np
import os
from tkinter import filedialog


#tiffファイルをカラー画像のjpeg画像に変換

if __name__ == '__main__':
    
  
    print("読込ファイル")
    path_pwd = os.path.abspath(os.path.dirname(__file__))
    file_read = filedialog.askopenfilename(title="読込ファイルの選択",filetypes=[("",'tif'),("",'bmp'),("",'jpg'),("",'png')],initialdir=path_pwd,multiple=True)
    
    #出力するファイルのディレクトリ
    dir_write = file_read[0].rsplit('/', 1)[0]
    if dir_write[-1] != "/":
        dir_write = dir_write + "/"
    if not os.path.exists(dir_write+"jpg"):
        os.mkdir(dir_write+"jpg")
    dir_write+="jpg/"

        
    for i,file in enumerate(file_read):
        
        img = Image.open(file)#(H,W)
        
        width, height = img.size  
        
        #numpy配列へ変換(H,W)→(H,W,3)に頑張って変換するため
        img_list = np.asarray(img)
        
        #出力するためのnumpy配列(H,W,3)
        imglist2=np.zeros((height,width,3),int).astype('uint8')
        
        
        msk=np.where(img_list==1)#病気の部分
        msk2=np.where(img_list==0)#（病気以外の部分）
    
    
        #背景以外は赤
        imglist2[msk[0],msk[1]]=[255,0,0]
#        imglist2[msk[0],msk[1],0]=255
#        imglist2[msk[0],msk[1],1]=0
#        imglist2[msk[0],msk[1],2]=0
        
        #背景は黒
        imglist2[msk2[0],msk2[1]]=[0,0,0]
#        imglist2[msk2[0],msk2[1],0]=0
#        imglist2[msk2[0],msk2[1],1]=0
#        imglist2[msk2[0],msk2[1],2]=0
    
        #画像の形式に変換
        pil_img=Image.fromarray(imglist2)
        
        #画像ごとの出力ファイル名
        file_type_tif=file.rsplit('/', 1)[1]
        file_type_jpg=file_type_tif.strip('.tif')+'.jpg'
        
        #出力場所は/out/fileName.jpg
        new_file = dir_write + file.rsplit('/', 1)[1].strip('.tif')+'.jpg'

        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img.save(new_file)
    print("Done!!")
    
    
    
    
    
    