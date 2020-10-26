# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:26:06 2020

@author: ShimaLab
"""



from PIL import Image
import os
import numpy as np
from tkinter import filedialog
from time import sleep
import sys



if __name__ == '__main__':
    
    
    print("読込ファイル")
    path_pwd = os.path.abspath(os.path.dirname(__file__))
    file_read = filedialog.askopenfilename(title="読込ファイルの選択",filetypes=[("",'bmp'),("",'jpg'),("",'png')],initialdir=path_pwd,multiple=True)
    
    for name in file_read:
        img = np.asarray(Image.open(name))
    