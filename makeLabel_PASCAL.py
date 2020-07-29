# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:34:24 2020

@author: ShimaLab
"""


import glob
import os
import numpy as np
from PIL import Image, ImageChops
import cv2


if __name__ == '__main__':

    
    dir_segmented = "SegmentationClass"#アノテーションのある画像のディレクトリ
    
    paths_segmented = glob.glob(dir_segmented + "/*")#アノテーションのある画像
    
    filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
    
    whiteLabel = list(map(lambda filename: "whiteLabel" + "/" + filename + ".jpg", filenames))#アノテーションのある画像ファイルたちの保存先
    
    colorLabel = list(map(lambda filename: "whiteLabel" + "/" + filename + ".jpg", filenames))#アノテーションのある画像ファイルたちの保存先
    
    
    wPath = "whiteLabel/"
    cPath = "colorLabel/"
    black = (0,0,0)
    white = (255,255,255)
    
    cnt = 0
    
    for name, filename in zip(paths_segmented, filenames):
        if(cnt%50==0):
            print(cnt)
        img = Image.open(name)
#        img = np.asarray(img.convert("RGB"))
        img = img.convert("RGB")
        
        r, g, b = img.split()
        

        _r = r.point(lambda _: 1 if _ != black[0] else 0, mode="1")
        _g = g.point(lambda _: 1 if _ != black[1] else 0, mode="1")
        _b = b.point(lambda _: 1 if _ != black[2] else 0, mode="1")
        
        mask = ImageChops.logical_or(_r, _g)
        mask = ImageChops.logical_or(mask, _b)
        img.paste(Image.new("RGB", img.size, white), mask=mask)
        
        
        img.save(wPath+filename + ".jpg")

#        if(cnt==2):
#            break
        cnt += 1