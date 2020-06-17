# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:49:12 2020

@author: ShimaLab
"""

from PIL import Image
import numpy as np
import os, os.path
from tkinter import filedialog
import sys
import pathlib

from sklearn.model_selection import train_test_split



if __name__ == '__main__':
    
    
    currentDir = os.path.abspath(os.path.dirname(__file__))
        
    path_pwd_pathlib = pathlib.Path(os.getcwd())#カレントディレクトリの取得,strではない
    parentDir=str(path_pwd_pathlib.parent)#親ディレクトリの取得
    dirNpzFile=parentDir+'\\npyFiles\\'
  
    originImage=np.load(dirNpzFile+'OriginalImage.npy')
    whiteLabel=np.load(dirNpzFile+'whiteMask.npy')
    colorLabel=np.load(dirNpzFile+'colorMask.npy')
    idx = np.array(range(originImage.shape[0]))
    
    idxTrain, idxVal = train_test_split(idx ,test_size=0.3, random_state=81)
#    imageTrain, imageVal, whiteLabelTrain, whiteLabelVal, colorLabelTrain, colorLabelVal, idxTrain, idxVal \
#        = train_test_split(originImage, whiteLabel, colorLabel, idx ,test_size=0.3, random_state=81)
    
    idxTrain=sorted(idxTrain)
    idxVal=sorted(idxVal)
    
    
    #npyファイルの保存
    #現画像
    outOriginalImageTrain=originImage[idxTrain]
    outOriginalImageVal=originImage[idxVal]
    np.save(dirNpzFile+'trainOriginalImage.npy',outOriginalImageTrain)
    np.save(dirNpzFile+'valOriginalImage.npy',outOriginalImageVal)
    
    
    #白黒マスク
    outWhiteMaskImageTrain=whiteLabel[idxTrain]
    outWhiteMaskImageVal=whiteLabel[idxVal]
    np.save(dirNpzFile+'trainWhiteLabel.npy',outWhiteMaskImageTrain)
    np.save(dirNpzFile+'valWhiteLabel.npy',outWhiteMaskImageVal)
        
    
    #カラーマスク
    outColorMaskTrain=colorLabel[idxTrain]
    outColorMaskVal=colorLabel[idxVal]
    np.save(dirNpzFile+'trainColorLabel.npy',outColorMaskTrain)
    np.save(dirNpzFile+'valColorLabel.npy',outColorMaskVal)
    
    
    print("Done !")