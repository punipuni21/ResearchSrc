# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:11:26 2020

@author: ShimaLab
"""

import glob
import os
import numpy as np
from PIL import Image

if __name__ == '__main__':
    
    
    dir_original = "JPEGImages"#元画像のディレクトリ
    dir_segmented = "SegmentationClass"#アノテーションのある画像のディレクトリ
    dir_sgmentObj = "SegmentationObject"
    
    paths_original = glob.glob(dir_original + "/*")#元画像を全て取得
    paths_segmented = glob.glob(dir_segmented + "/*")#アノテーションのある画像
    paths_segOjb = glob.glob(dir_sgmentObj + "/*")#アノテーションのある画像
    
    dir_segmentTrainVal = "Original/"#保存先
    
    if len(paths_original) == 0 or len(paths_segmented) == 0:
        raise FileNotFoundError("Could not load images.")
    # 教師画像の拡張子を.pngに書き換えたものが読み込むべき入力画像のファイル名になります
    
    
    
    filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
    
    paths_original_ = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))#アノテーションのある画像ファイルたち
    path_tosave = list(map(lambda filename: dir_segmentTrainVal + filename + ".jpg", filenames))#アノテーションのある画像ファイルたちの保存先
   
    
    cnt = 0
    img = Image.open(paths_segmented[0])
    hh = np.asarray(img)
    imgg = img.convert('RGB')
    imgg.save('hoge.png')
    h = np.asarray(imgg)
    img2 = Image.open(paths_original[0])
    
    img3 = Image.open(paths_segOjb[0])
    
    
    
#    image = Image.open(file_path) #パスから画像１枚をロード
    print(img.mode,' ',img2.mode, ' ', img3.mode) #教師データを読み込んだ際，自動で"P"モード(パレットモード)になります
    for (dir_save, imgName) in zip(path_tosave, paths_original_,):
        if(cnt % 50 == 0):
            print("now = ",cnt)
        img = Image.open(imgName)#アノテーションのついている元画像を開く
        img2 = img.convert('RGB')#`画像のモードをRGBに変える
        img2.save(dir_save)
        cnt += 1
#        print(dir_save,' ',imgName)
      