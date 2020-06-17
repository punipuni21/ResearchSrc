# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:12:34 2020

@author: ShimaLab
"""
from PIL import Image
import numpy as np
import os, os.path
from tkinter import filedialog
import sys
import pathlib

#--------------------------------------------------------------------------------------#
##ラベルを保存するためのディレクトリ
def makeDir(path_pwd):
    
    
    dirLabelWhite=path_pwd
    dirLabelColor=path_pwd
    
    ##ラベルを保存するディレクトリの指定
    if path_pwd[-1] != '\\':
        dirLabelWhite = path_pwd + '\\Label\\'
        dirLabelColor = path_pwd + '\\Label\\'
    if not os.path.exists(dirLabelWhite+"White"):
        os.mkdir(dirLabelWhite+"White")

    dirLabelWhite = dirLabelWhite + 'White\\'
    
    if not os.path.exists(dirLabelWhite+"LabelImage"):
        os.mkdir(dirLabelWhite+"LabelImage")
    
    if not os.path.exists(dirLabelColor+"Color"):
        os.mkdir(dirLabelColor+"Color")
    dirLabelColor = dirLabelColor + 'Color\\'
    
    if not os.path.exists(dirLabelColor+"LabelImage"):
        os.mkdir(dirLabelColor+"LabelImage")
    
    return (dirLabelWhite,dirLabelColor)

#--------------------------------------------------------------------------------------#
#病変のディレクトリにあるファイル数を取得
def searchtheNumberOfFiles(currentDir):
    
    theNumberOfFiles={}
    for dir_path in os.listdir(parentDir):
        target_dir = parentDir + "\\" + dir_path
        files = os.listdir(target_dir)
        count = len(files)
        theNumberOfFiles[dir_path]=count
    return theNumberOfFiles
#--------------------------------------------------------------------------------------#
def isLesionSizeSame(dirName, lesionList):
    
    #このプログラムはhoge\srcにあるので病変のサイズを取得するためには親ディレクトリ内で探すこと
    theNumberOfFiles=searchtheNumberOfFiles(parentDir)
  
    #各病変の個数を格納する変数
    theNumberOfEachLesionList=[]
    for LesionName in lesionList:
        theNumberOfEachLesionList.append(theNumberOfFiles[LesionName])
    
    
    theNumberOfEachLesion=theNumberOfEachLesionList[0]
#    theNumberOfEachLesionList[0]=8 #これはデバッグ用
    
    #病変のファイルの個数が全て同じでない場合は何かがおかしいのでプログラムを終了
    return (any(i != theNumberOfEachLesion for i in theNumberOfEachLesionList),theNumberOfEachLesion)







if __name__ == '__main__':
    
    
    #--------------------------------------------------------------------------------#
    #病変リスト
    lesionList=["Haemorrhages","HardExudates","Microaneurysms","SoftExudates"]
    #画像の情報
    Height,Width=(512,512)
    #--------------------------------------------------------------------------------#
    
    
    print("読込ファイル")
    currentDir = os.path.abspath(os.path.dirname(__file__))
        
    path_pwd_pathlib = pathlib.Path(os.getcwd())#カレントディレクトリの取得,strではない
    parentDir=str(path_pwd_pathlib.parent)#親ディレクトリの取得
    
    #ラベルを保存するディレクトリの作成（白黒ラベル，カラーラベル）
    (dirLabelWhite,dirLabelColor) = makeDir(path_pwd=parentDir)
    
    
    #病変の個数が全て同じか判定，同じでないならプログラム終了
    isEachLesionSizeSame, theNumberOfEachLesion=isLesionSizeSame(parentDir, lesionList)
    if(isEachLesionSizeSame):
        sys.exit("病変の個数が全て統一されていない，各病変のディレクトリを確認してください")
    
    #ここからマスクを生成するプログラム
    
    whiteMask=np.zeros((theNumberOfEachLesion,Height,Width,3),'uint8')#白黒マスク(RoleNet前段部用)
    colorMask=np.zeros((theNumberOfEachLesion,Height,Width,3),'uint8')#色マスク（RoleNet後段部用）
    colorDict={"Haemorrhages":np.array([255,0,0],'uint8'),"HardExudates":np.array([0,255,0],'uint8'),\
               "Microaneurysms":np.array([0,0,255],'uint8'),"SoftExudates":np.array([255,255,0],'uint8')}#赤，緑，青，黄
    
    
    for lesionName in lesionList:
        
        npyFile=parentDir+'\\npyFiles\\'+lesionName+'.npy'
        hoge=np.load(npyFile)
        
        for i in range(theNumberOfEachLesion):
            
            now=hoge[i]
#            print(colorDict[lesionName][0])
            #白黒画像は全て白でマスク
            
            whiteMask[i,:,:,0]=np.where(now[:,:,0]>0,255,whiteMask[i,:,:,0])
            whiteMask[i,:,:,1]=np.where(now[:,:,0]>0,255,whiteMask[i,:,:,1])
            whiteMask[i,:,:,2]=np.where(now[:,:,0]>0,255,whiteMask[i,:,:,2])
            
            colorMask[i,:,:,0]=np.where(now[:,:,0]>0,colorDict[lesionName][0],colorMask[i,:,:,0])
            colorMask[i,:,:,1]=np.where(now[:,:,0]>0,colorDict[lesionName][1],colorMask[i,:,:,1])
            colorMask[i,:,:,2]=np.where(now[:,:,0]>0,colorDict[lesionName][2],colorMask[i,:,:,2])

         

    ##保存
    np.save(dirLabelWhite+"whiteMask",whiteMask)
    np.save(dirLabelColor+"colorMask",colorMask)
    
    #ラベル付けはできている
    #画像の保存
    for idx in range(theNumberOfEachLesion):
        
        whiteLabelImage=Image.fromarray(whiteMask[idx].astype(np.uint8))
        
        colorLabelImage=Image.fromarray(colorMask[idx].astype(np.uint8))
        
        whiteLabelImage.save(dirLabelWhite+"LabelImage\\"+str(idx+1)+'.jpg')
        colorLabelImage.save(dirLabelColor+"LabelImage\\"+str(idx+1)+'.jpg')    
        
        
#        
#    print('Done !')
#    
#    
#    
#    
#    
#    
    
    
    
    
    
    
    
    
    
    
    