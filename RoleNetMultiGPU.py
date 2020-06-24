#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概要
"""
__author__ = 'Taiki Horiuchi'
__version__ = '1.0'
__date__    = "2019/07/03 16:43"


import path
from usecase.data import imedit
#from usecase.data.convert import ims2labs,labs2ims
from usecase.data.convert_label import labimg2labarg,labarg2labimg,read_dict
from usecase.model import argmax2evaluate
from usecase.model.gradcam import save_att_tile
import chainer
from chainer.datasets.tuple_dataset import TupleDataset
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.serializers import save_npz
from chainer import serializers
import chainer.functions as F
from chainer import Variable
from unet import UNet
from thinnet_wound_minimal import ThinNet_wound_minimal
from thinnet_wound_very_small import ThinNet_wound_very_small
from thinnet_wound_small import ThinNet_wound_small
from thinnet_wound import ThinNet_wound
import numpy as np
import cupy as cp
import csv
from copy import deepcopy
from usecase.data import gaussian

def write_csv(file, save_dict):
    save_row = {}

    with open(file,'w') as f:
        writer = csv.DictWriter(f, fieldnames=save_dict.keys(),delimiter=",",quotechar='"')
        writer.writeheader()

        k1 = list(save_dict.keys())[0]
        length = len(save_dict[k1])

        for i in range(length):
            for k, vs in save_dict.items():
                save_row[k] = vs[i]

            writer.writerow(save_row)

if __name__ == '__main__':
    #1 PATHの設定
    path.init()

    #2 ロード
    print("ロード")
    lab_name, lab_data = read_dict(path.FILE_LAB)
    calss_num = lab_data.shape[0]
    val_img = np.load(path.FILE_VAL_IMG)
    val_lab,_ = labimg2labarg(np.load(path.FILE_VAL_LAB),lab_data)
    tes_img = np.load(path.FILE_TES_IMG)[0:5,:,:,:]
    tes_lab,_ = labimg2labarg(np.load(path.FILE_TES_LAB),lab_data)
    tes_lab = tes_lab[0:5,:,:,:]
    tra_img = np.load(path.FILE_TRA_IMG)
    tra_lab,_ = labimg2labarg(np.load(path.FILE_TRA_LAB),lab_data)
        
    print("変換")
    ## uint8に変換 NHWC Channel last
    val_img = np.array(val_img,np.uint8)
    val_lab = np.array(val_lab,np.uint8)
    val_att = np.array(gaussian.main(val_lab)*255,np.uint8)
    tra_img = np.array(tra_img,np.uint8)
    tra_lab = np.array(tra_lab,np.uint8)
    tra_att = np.array(gaussian.main(tra_lab)*255,np.uint8)
    
    tes_img = np.array(tes_img,np.uint8)
    tes_img_ori = deepcopy(tes_img)
    tes_lab = np.array(tes_lab,np.uint8)
    
    att_index = 2
    att_img = tes_img[att_index,:,:,:]
    
    ## Channel first NHWC->NCHW
    tra_img = np.transpose(tra_img, [0, 3, 1, 2])
    tra_lab = np.transpose(tra_lab, [0, 3, 1, 2])
    tra_att = np.transpose(tra_att, [0, 3, 1, 2])
    val_img = np.transpose(val_img, [0, 3, 1, 2])
    val_lab = np.transpose(val_lab, [0, 3, 1, 2])
    val_att = np.transpose(val_att, [0, 3, 1, 2])
    tes_img = np.transpose(tes_img, [0, 3, 1, 2])
    tes_lab = np.transpose(tes_lab, [0, 3, 1, 2])
    
    train = TupleDataset(tra_img,tra_lab,tra_att)
    valid = TupleDataset(val_img,val_lab,val_att)
    test = TupleDataset(tes_img,tes_lab)
    
    # ハイパーパラメータの設定=============================================================
    batch_size = 2
    epoch_size = 32
    itr_size = int(tra_img.shape[0]/batch_size*epoch_size)
    gpu_device = 3
    class_size = tra_lab.shape[1]
    list_filename = ["unet","thinnet_wound_minimal","thinnet_wound_very_small","thinnet_wound_small","thinnet_wound_normal"]
    filename = list_filename[2]
    path.init(filename)
    #================================================================================
    #Single Memory
#    pool = cp.cuda.MemoryPool().malloc
#    cp.cuda.set_allocator(pool)
#    chainer.cuda.get_device(gpu_device).use()

    

    
    train_itr = SerialIterator(train, batch_size=batch_size, repeat=True, shuffle=False)
    
    
#    (tra_img,tra_lab,tra_att) = (None,None,None)
#    (val_img,val_lab,val_att) = (None,None,None)
#    (tes_img,tes_lab) = (None,None)
#    train = None
    
    #3 モデル生成
    print("モデル生成")
    if filename == list_filename[0]:
#        model = UNet(in_channels=3, out_channels=class_size)
        model_0 = UNet(in_channels=3, out_channels=class_size)
        model_1 = model_0.copy()
        model_0.to_gpu(0)
        model_1.to_gpu(1)
    
    elif filename == list_filename[1]:
        model = ThinNet_wound_minimal(in_channels=3, out_channels=class_size)
    elif  filename == list_filename[2]:
        model = ThinNet_wound_very_small(in_channels=3, out_channels=class_size)
    elif  filename == list_filename[3]:
        model = ThinNet_wound_small(in_channels=3, out_channels=class_size)
    else:
        model = ThinNet_wound(in_channels=3, out_channels=class_size)
    
    
    
    param_size = str(sum(p.data.size for p in model.params()))
    print("パラメータ数",param_size)
    file_param = path.DIR_MOD + model.__class__.__name__+".txt"
    with open(file_param, mode='w') as f:
        f.write(param_size)
#    serializers.load_npz(path.DIR_MOD+model.__class__.__name__+".npz", model)
#    model.to_gpu(gpu_device)





    #4 パラメータ最適化手法
    optimizer = Adam()
    optimizer.setup(model)
    
    
    
#    # ログ
#    results_train, results_valid = {}, {}
#    results_train['loss'], results_train['accuracy'] = [], []
#    results_valid['loss'], results_valid['accuracy'] = [], []
#    
#    
#    # ログ
#    results = {}
#    results['itreation'] = [0 for i in range(itr_size)]
#    results['tra_loss'], results['tra_acc'] = [0 for i in range(itr_size)], [0 for i in range(itr_size)]
#    results['val_loss'], results['val_acc'] = [0 for i in range(itr_size)], [0 for i in range(itr_size)]
#    
    
    results = {}
    results['itreation'] = []
    results['tra_loss'], results['tra_acc'] = [], []
    results['val_loss'], results['val_acc'] = [], []
    
    
    count = 0#イテレーション回数
    train_count = 0
    sum_accuracy = 0
    sum_loss = 0
    
    
    for epoch in range(epoch_size):
        print("学習開始 epoch",epoch)
        
        
        while True:
            # ミニバッチの取得
            batch_data = train_itr.next()
            
            #0番目
            
            
            bat_img0, bat_lab0, bat_lab_att0 = chainer.dataset.concat_examples(batch_data[:batch_size//2],0)
            bat_img0 = Variable(cp.array(bat_img0/255,dtype=cp.float32))
            bat_lab_arg0 = Variable(cp.array(cp.argmax(bat_lab0,axis=1),dtype=cp.int))
            bat_lab0 = Variable(cp.array(bat_lab0/255,dtype=cp.float32))
            
            #順方向の計算
            bat_pre0 = model_0.call_no_active(bat_img0)
            tra_loss0 = F.softmax_cross_entropy(x=bat_pre0,t=bat_lab_arg0,cache_score=True,enable_double_backprop=False)
            tra_acc0 = F.accuracy(F.softmax(bat_pre0), bat_lab_arg0)
            
            #エポックごとのaccuracyとかをまとめて保存
#            tra_loss0.to_cpu()
#            tra_acc0.to_cpu()
            
#            sum_loss += float(tra_loss0.array)*len(bat_img0)
#            sum_accuracy += float(tra_acc0.array) * len(bat_img0)
            
            model_0.cleargrads()
            tra_loss0.backward()
            
          
            #1番目
            bat_img1, bat_lab1, bat_lab_att1 = chainer.dataset.concat_examples(batch_data[batch_size//2:],1)
            bat_img1 = Variable(cp.array(bat_img1/255,dtype=cp.float32))
            bat_lab_arg1 = Variable(cp.array(cp.argmax(bat_lab1,axis=1),dtype=cp.int))
            bat_lab1 = Variable(cp.array(bat_lab1/255,dtype=cp.float32))
            
            bat_pre1 = model_1.call_no_active(bat_img1)
            tra_loss1 = F.softmax_cross_entropy(x=bat_pre1,t=bat_lab_arg1,cache_score=True,enable_double_backprop=False)
            tra_acc1 = F.accuracy(F.softmax(bat_pre1), bat_lab_arg1)
            
            
            #エポックごとのaccuracyとかをまとめて保存
#            tra_loss1.to_cpu()
#            tra_acc1.to_cpu()
            
#            sum_loss += float(tra_loss1.array)*len(bat_img1)
#            sum_accuracy += float(tra_acc1.array) * len(bat_img1)
            
            
            model_1.cleargrads()
            tra_loss1.backward()


            #勾配をまとめる
            model_0.addgrads(model_1)
            optimizer.update()
            model_1.copyparams(model_0)
            
            train_count += batch_size
            
            #ここで１つのミニバッチの学習が終了
            
            #全てのミニバッチの学習が終了したら
            if train_itr.is_new_epoch:
                
                #ここでエポックごとのAccuracyとLossを保存
                train_loss=float(tra_loss0.array)*len(bat_img0)+ float(tra_loss1.array)*len(bat_img1)
                train_acc=(float(tra_acc0.array) * len(bat_img0)+float(tra_acc1.array) * len(bat_img1))/batch_size
                # 検証用データに対する結果の確認
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    val_img, val_lab, val_att_lab = chainer.dataset.concat_examples(valid, 0)
                    val_img = Variable(cp.array(val_img/255,dtype=cp.float32))
                    val_lab_arg = Variable(cp.array(cp.argmax(val_lab,axis=1),dtype=cp.int))
                    val_lab = Variable(cp.array(val_lab/255,dtype=cp.float32))
                    
                    val_pre = model.call_no_active(val_img)
                    val_loss = F.softmax_cross_entropy(x=val_pre,t=val_lab_arg,cache_score=True,enable_double_backprop=False)
                    val_acc = F.accuracy(F.softmax(val_pre), val_lab_arg)
    
                # 注意：GPU で計算した結果はGPU上に存在するため、CPU上に転送します
                val_loss.to_cpu()
                val_acc.to_cpu()
    
                # 結果の表示
#                print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'
#                      'acc (train): {:.4f}, acc (valid): {:.4f}'.format(
#                    epoch, count, loss_train.array.mean(), loss_valid.array.mean(),
#                      acc_train.array.mean(), acc_valid.array.mean()))
    
                # 可視化用に保存
                results['tra_loss'].append(train_loss)
                results['tra_acc'].append(train_acc)
                results['val_loss'].append(val_loss.array)
                results['val_acc'].append(val_acc.array)
                
                
                train_count=0
                sum_accuracy = 0
                sum_loss = 0
                break
    

    #モデルの保存
    model.to_cpu()
    save_npz(path.DIR_MOD+model.__class__.__name__+".npz",model)
    model.to_gpu()
    
    # 変遷の保存
    write_csv(path.FILE_HIS,results)
       
    # テストデータによる評価
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', True):
        #Variable
        tes_img, tes_lab = chainer.dataset.concat_examples(test,gpu_device)
        tes_lab = Variable(cp.array(tes_lab/255,dtype=cp.float32))
        tes_img = Variable(cp.array(tes_img/255,dtype=cp.float32))
        tes_pre = model(tes_img)
        tes_pre_att = model.get_all_attention(tes_img[att_index:att_index+1,:,:,:])#F.expand_dims(tes_img[att_index,:,:,:],axis=0))
    #np
    tes_img = np.array(chainer.cuda.to_cpu(tes_img.data)*255,dtype=np.uint8)#0~255
    tes_pre = np.array(chainer.cuda.to_cpu(tes_pre.data),dtype=np.float32)#0~1
    tes_lab_argmax = np.argmax(np.array(chainer.cuda.to_cpu(tes_lab.data),dtype=np.float32),axis=1)#0~3
    tes_pre_argmax = np.argmax(tes_pre,axis=1)
    tes_lab_onehot = np.identity(calss_num)[tes_lab_argmax]#tensor(None,480,480,calss_num) 0-1
    tes_out_onehot = np.identity(calss_num)[tes_pre_argmax]#tensor(None,480,480,calss_num) 0-1
    
    save_att_tile(class_size,
                  lab_data,
                  att_img,
                  tes_pre_argmax[att_index,:,:],#tes_out_onehot[2,:,:,:],
                  tes_lab_argmax[att_index,:,:],#tes_lab_onehot[2,:,:,:],
                  tes_pre_att,
                  path.FILE_ATT,
                  " ")
    ## 評価値の算出と混同行列の出力
    argmax2evaluate.main(tes_lab_argmax,tes_pre_argmax,lab_name,
                         path.FILE_EVA,path.FILE_MAT_CSV, None,
                         axislabelsize=int(20),ticksize=int(14),valuesize=int(12))
        
    #8 出力結果を画像として保存
    print("出力画像保存")
    ## labとoutのone_hot
    
    ## labelやRGB画像に変換
    tes_lab_img = labarg2labimg(tes_lab_argmax, lab_data)#numpy(None,480,480,3) 0-255
    tes_out_img = labarg2labimg(tes_pre_argmax, lab_data)#numpy(None,480,480,3) 0-255

    ## タイル画像を生成
    tes_im_tile = imedit.tile(tes_img_ori,column_size=1,boarder_size=5,tile_width=200)#pil(None*200,200,3)
    tes_out_im_tile = imedit.tile(tes_out_img,column_size=1,boarder_size=5,tile_width=200)#pil(None*200,200,3)
    tes_lab_im_tile = imedit.tile(tes_lab_img,column_size=1,boarder_size=5,tile_width=200)#pil(None*200,200,3)

    ## タイル画像を横に連結
    im = imedit.concat_im_h(tes_im_tile, tes_out_im_tile)
    im = imedit.concat_im_h(im, tes_lab_im_tile)

    ## 連結されたタイル画像を保存
    im.save(path.FILE_RES_TIL)
     
    

    
    print("Done!")