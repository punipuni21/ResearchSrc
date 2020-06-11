#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概要
"""
__author__ = 'Watanabe Ryotaro'
__version__ = '1.0'
__date__    = "2020/06/12 20:02"

from glob import glob
import os
import numpy as np
from PIL import Image
from tkinter import filedialog
from time import sleep
import cv2


def _concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def tile(imgs_np,column_size,row_size=None,boarder='white',boarder_size=10,tile_width=800):
    """
    画像(4dy)をタイルにして1枚の画像(Image)として出力

    Examples
    --------
    >>>imgs_np = np.array([np.array(Image.open("a.jpg")),np.array(Image.open("b.jpg"))])
    >>>print(imgs_np.shape)
    (2,480,480,3)
    >>>img_tile_pil = write_img.tile(imgs_np=imgs_np,column_size=10)
    >>>img_tile_pil.save("tile.jpg")
    """

    (num,wid,hei,cha) = imgs_np.shape
    if column_size < 1:
        column_size = 1
    if row_size == None:
        row_size = (num-1)//column_size + 1
    if boarder=='white':
        w_boa = 255
    else:
        w_boa = 0
    if np.max(imgs_np)<=1:
        imgs_np = imgs_np * 255
    #print("write_imgs.tile (行{},列{})".format(row_size,column_size))
    row_imgs =[]
    for r in range(row_size):
        colum_imgs =[]
        for c in range(column_size):
            i = c + r * column_size
            if i < num:
                img_np = imgs_np[i,:,:,:]
                img_np_boa = np.ones((wid+2*boarder_size,hei+2*boarder_size,cha))*w_boa
                img_np_boa [boarder_size:wid+boarder_size,boarder_size:hei+boarder_size,:] = img_np
            else:
                img_np_boa = np.ones((wid+2*boarder_size,hei+2*boarder_size,cha))*w_boa
            colum_imgs.append(img_np_boa)
        row_imgs.append(colum_imgs)
    img_tile_pil = Image.fromarray(np.uint8(_concat_tile(row_imgs)))
    rate = tile_width/column_size
    img_tile_pil = img_tile_pil.resize((int(column_size*rate),int(row_size*rate)))
    return img_tile_pil

def main(imgs_np,file_write):
    #リサイズ
    (n,h,w,c) = imgs_np.shape
    ratio = tile_width/column_size/w
    resize = (int(h*ratio),int(w*ratio))
    imgs = np.zeros((n,resize[0],resize[1],c),dtype=np.uint8)

    for i,img in enumerate(imgs_np):
        imgs[i,:,:,:] = cv2.resize(img,resize)
    if len(file_read) > 0:
        img_tile_pil = tile(imgs,column_size=column_size,
                            boarder=boarder,boarder_size=boarder_size,
                            tile_width=tile_width)
        img_tile_pil.save(file_write)
        print("\nタイル画像出力完了")



if __name__ == '__main__':

    print("列数,画像幅,枠線サイズ,枠線色,を入力 defalut=(5,800,2,white)\n入力例>>>5 800 2 black")
    try:
        column_size,tile_width,boarder_size,boarder = (i for i in input().split())
        column_size = int(column_size)
        tile_width = int(tile_width)
        boarder_size = int(boarder_size)
        boarder = str(boarder)
    except ValueError:
        column_size = 5
        tile_width = 800
        boarder_size = 2
        boarder = 'white'

    print("読込ファイル")
    path_pwd = os.path.abspath(os.path.dirname(__file__))
    file_read = filedialog.askopenfilename(title="読込ファイルの選択",filetypes=[("",'jpg'),("",'png'),("",'npy')],initialdir=path_pwd,multiple=True)

    filenpy_read = [l for l in file_read if l.endswith('.npy')]
    if len(filenpy_read) > 0:
        for file in file_read:
            file_write = file.rsplit('.', 1)[0]
            file_write += "_tile.jpg"
            imgs_np = np.uint8(np.load(file))
            print("{}の読込完了".format(file))
            main(imgs_np,file_write)
    else:
        file_write = file_read[0].rsplit('.', 1)[0]
        file_write += "_tile.jpg"
        img_np = np.asarray(Image.open(file_read[0]))
        (h,w,c) = img_np.shape
        imgs_np = np.zeros((len(file_read),h,w,c))
        for i,file in enumerate(file_read):
             img = Image.open(file)
             if img.mode != "RGB":
                 img = img.convert("RGB")
             if img.size == (h,w):
                 imgs_np[i,:,:,:] = np.asarray(img,np.uint8)
             else:
                 print("{}の画像を最初の読込画像にリサイズ".format(file))
                 img = img.resize((h,w))
                 imgs_np[i,:,:,:] = np.asarray(img,np.uint8)
        print("画像群の読込完了")
        main(imgs_np,file_write)
    sleep(3)
