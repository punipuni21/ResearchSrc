# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:06:08 2020

@author: ShimaLab
"""

import pathlib
import os, os.path
import numpy as np

if __name__ == '__main__':

    p_sub = pathlib.Path(os.getcwd())
    par=str(p_sub.parent)
    
    hoge = np.array([1,2,3,4,5])
    print(hoge[hoge>3])
    print(np.where(hoge>3))
    hoge2=np.where(hoge>3,10,hoge)
    print(hoge2)