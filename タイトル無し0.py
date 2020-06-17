# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:53:57 2020

@author: ShimaLab
"""
from PIL import Image
import numpy as np
import os, os.path
from tkinter import filedialog
import sys
import pathlib



if __name__ == '__main__':
    
    npyFile='Haemorrhages.npy'
    hoge=np.load(npyFile)[16]
    
    out=np.zeros((512,512,3),'uint8')
    out[:,:,0]=np.where(hoge[:,:,0]>0,0,hoge[:,:,0])
    out[:,:,1]=np.where(hoge[:,:,0]>0,255,hoge[:,:,0])
    out[:,:,2]=np.where(hoge[:,:,0]>0,0,hoge[:,:,0])
    
    pil_img = Image.fromarray(out.astype(np.uint8))
    pil_img.save('save_pillow.jpg')      