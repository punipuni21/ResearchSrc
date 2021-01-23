#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
import copy, sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import confusion_matrix
import pandas as pd
from RoleNetFront import SeparableConv2d, RoleNetFront
from RoleNetBack import RoleNetBack
from RoleNet import RoleNet
from src import MyNet2, MyNet
from multiRoleNet import multiRoleNet
from multiRoleNet2 import multiRoleNet2


def sum_tensors(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

label_color_map = [
               (0, 0, 0),  # background
               (128, 0, 0), # aeroplane
               (0, 128, 0), # bicycle
               (128, 128, 0), # bird
               (0, 0, 128), # boat
               (128, 0, 128), # bottle
               (0, 128, 128), # bus
               (128, 128, 128), # car
               (64, 0, 0), # cat
               (192, 0, 0), # chair
               (64, 128, 0), # cow
               (192, 128, 0), # dining table
               (64, 0, 128), # dog
               (192, 0, 128), # horse
               (64, 128, 128), # motorbike
               (192, 128, 128), # person
               (0, 64, 0), # potted plant
               (128, 64, 0), # sheep
               (0, 192, 0), # sofa
               (128, 192, 0), # train
               (0, 64, 128) # tv/monitor
]

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out



class MyDataset(Dataset):

    IMG_EXTENSIONS = [".jpg",".JPG",".npy",".png"]


    def __init__(self, base_dir, dataset, transform=None):
        # data/img
        # data/labelみたいにしておきたい
        self.transform = transform
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = self._get_img_paths(base_dir + "/image/" + dataset + str("/"))
        self.lab_paths = self._get_lab_paths(base_dir + "/label/" + dataset + str("/"))
        self.label = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
        
    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in MyDataset.IMG_EXTENSIONS
        ]
        return img_paths


    def _get_lab_paths(self, lab_dir):
        """指定したディレクトリ内のラベル画像ファイルのパス一覧を取得する。
          返り値は入力画像とラベル画像
        """
        lab_dir = Path(lab_dir)
        lab_paths = [
            p for p in lab_dir.iterdir() if p.suffix in MyDataset.IMG_EXTENSIONS
        ]
        return lab_paths


    def __getitem__(self, index):
        # index 番目のサンプルが要求されたときに返す処理を実装
        # print(index)
        img_path = self.img_paths[index]
        lab_path = self.lab_paths[index]

        # 画像を読み込む。
        img = Image.open(img_path).resize((256,256))
        labtissue = Image.open(lab_path).resize((256,256))

        # 前処理がある場合は行う。
        # if self.transform is not None:
        #     img = np.asarray(img)
        #     img = torch.from_numpy(img).clone()


        img = np.asarray(img)
        img = torch.from_numpy(img.astype(np.float32)).clone()
        img = img.permute(2, 0, 1)
        img /= 255


        labtissue = np.asarray(labtissue).astype('uint8')
        labwound = copy.deepcopy(labtissue)
        # lab = np.where(lab == 255, 0, lab)
        # lab = np.identity(21)[lab]

        classLabtissue = np.zeros((labtissue.shape[0], labtissue.shape[1]))
        classLabwound = np.zeros((labwound.shape[0], labwound.shape[1]))
        
        for i in range(labtissue.shape[0]):
            for j in range(labtissue.shape[1]):
                mini = 1e9
                labId = 0
                for n in range(4):
                    cnt = 0
                    for k in range(3):
                        cnt += (labtissue[i, j, k] - self.label[n, k]) ** 2
                    if (cnt < mini):
                        mini = cnt
                        labId = n
                classLabtissue[i, j] = labId
                if(labId < 3):
                    classLabwound[i, j] = 0#wound
                else:
                    classLabwound[i, j] = 1#not wound
        labtissue = np.identity(4)[classLabtissue.astype('uint8')]
        labtissue = torch.from_numpy(labtissue.astype(np.float32)).clone()
        labtissue = labtissue.permute(2, 0, 1)
        
        labwound = np.identity(2)[classLabwound.astype('uint8')]
        labwound = torch.from_numpy(labwound.astype(np.float32)).clone()
        labwound = labwound.permute(2, 0, 1)

        return (img, labwound, labtissue)

    def __len__(self):
        # データセットのサンプル数が要求されたときに返す処理を実装
        return len(self.img_paths)


def calc_loss(pred, target, metrics, bce_weight=0.5):
   
    #predはsoftmaxをとっていない
    #pred:（woundの結果，tissueの結果）のタプル
    predwound ,predtissue = pred;
    targetwound ,targettissue = target;
        
    targetwound = torch.argmax(targetwound, dim=1)
    targettissue = torch.argmax(targettissue, dim=1)
    
    losswound = nn.cross_entropy(predwound,targetwound)
    losstissue = nn.cross_entropy(predtissue,targettissue)
    
    predwound = F.softmax(predwound,dim=1)
    predtissue = F.softmax(predtissue,dim=1)

    predwound = torch.argmax(predwound, dim=1)
    predtissue = torch.argmax(predtissue, dim=1)

    # print(pred.shape,target.shape)
    correctwound = (predwound == targetwound).sum().item()
    correcttissue = (predtissue == targettissue).sum().item()
    
    
    loss = losswound + losstissue
    
    
    
    metrics['lossall'] += loss.data.cpu().numpy() * targetwound.size(0)
#    metrics['correctall'] += correctwound+correcttissue
    
    metrics['losswound'] += losswound.data.cpu().numpy() * targetwound.size(0)
    metrics['correctwound'] += correctwound
    
    metrics['losstissue'] += losstissue.data.cpu().numpy() * targettissue.size(0)
    metrics['correcttissue'] += correcttissue
    
    
    
    predwound = predwound.to('cpu').detach().numpy().astype('uint32')
    targetwound = targetwound.to('cpu').detach().numpy().astype('uint32')
    
    
    mat_confusionwound = confusion_matrix(y_true=targetwound.flatten(), y_pred=predwound.flatten(),labels=[0,1])
    
    predtissue = predtissue.to('cpu').detach().numpy().astype('uint32')
    targettissue = targettissue.to('cpu').detach().numpy().astype('uint32')
    mat_confusiontissue = confusion_matrix(y_true=targettissue.flatten(), y_pred=predtissue.flatten() ,labels=[0,1,2,3])
    
    
    # mat_confusion = confusion_matrix(y_true=target.flatten(), y_pred=pred.flatten())
    # print(mat_confusion)
    # print(mat_confusion.shape)
    print("----------------------------------------")
    print(mat_confusionwound)
    print(mat_confusiontissue)
    print("----------------------------------------")
    
    true_positivewound = np.diag(mat_confusionwound)
    false_positivewound = np.sum(mat_confusionwound, 0) - true_positivewound
    false_negativewound = np.sum(mat_confusionwound, 1) - true_positivewound
    
    true_positivetissue = np.diag(mat_confusiontissue)
    false_positivetissue = np.sum(mat_confusiontissue, 0) - true_positivetissue
    false_negativetissue = np.sum(mat_confusiontissue, 1) - true_positivetissue
    # iou = (true_positive) / (true_positive + false_positive + false_negative+1)
    # print(true_positive)
    # print(false_negative)
    # print(false_positive)
    metrics["TPwound"] += true_positivewound
    metrics["FPwound"] += false_positivewound
    metrics["FNwound"] += false_negativewound
    
    metrics["TPtissue"] += true_positivetissue
    metrics["FPtissue"] += false_positivetissue
    metrics["FNtissue"] += false_negativetissue
   
    return loss




def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        if(k in ["TPwound","FPwound","FNwound","TPtissue","FPtissue","FNtissue"]):
            continue
        if(k in ["correctwound","correcttissue"]):
            outputs.append("{}: {:4f}".format("accuracy", metrics[k] / (epoch_samples*256*256)))            
        else:
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))




def train_model(model, optimizer, scheduler, dataloaders, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    model_path = 'model.pth'
    log = pd.DataFrame(index=[], 
    columns=["tralosswound","traaccwound","traIoUwound","vallosswound","valaccwound", "valIoUwound","teslosswound","tesaccwound", "tesIoUwound",
        "tralosstissue","traacctissue","traIoUtissue","vallosstissue","valacctissue", "valIoUtissue","teslosstissue","tesacctissue", "tesIoUtissue"])
    for epoch in range(num_epochs):
        tralosswound = 0.0
        traaccwound = 0.0
        traIoUwound = 0.0
        vallosswound=0.0
        valaccwound=0.0
        valIoUwound=0.0
        teslosswound = 0.0
        tesaccwound = 0.0
        tesIoUwound = 0.0
        
        tralosstissue = 0.0
        traacctissue = 0.0
        traIoUtissue = 0.0
        vallosstissue=0.0
        valacctissue=0.0
        valIoUtissue=0.0
        teslosstissue = 0.0
        tesacctissue = 0.0
        tesIoUtissue = 0.0
        
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'tes']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            cnt = 0
            for inputs, labelwound, labtissue in dataloaders[phase]:
                if(cnt%5==0):
                    print(cnt)
                # if(cnt%20==0):
                #     print(cnt)
                cnt += 1
                inputs = inputs.to(device)
                labelwound = labelwound.to(device)
                labtissue = labtissue.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, (labelwound, labtissue),  metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)
            if phase == "train":
                tralosswound = metrics['losswound'] / epoch_samples
                traaccwound = metrics['correctwound']/(epoch_samples*256*256)
                traIoUwound = np.mean(metrics["TPwound"]/(metrics["TPwound"]+metrics["FPwound"]+metrics["FNwound"]))
                
                tralosstissue = metrics['losstissue'] / epoch_samples
                traacctissue = metrics['correcttissue']/(epoch_samples*256*256)
                traIoUtissue = np.mean(metrics["TPtissue"]/(metrics["TPtissue"]+metrics["FPtissue"]+metrics["FNtissue"]))
                
                
            elif phase == "val":
                vallosswound = metrics['losswound'] / epoch_samples
                valaccwound = metrics['correctwound']/(epoch_samples*256*256)
                valIoUwound = np.mean(metrics["TPwound"]/(metrics["TPwound"]+metrics["FPwound"]+metrics["FNwound"]))
                
                vallosstissue = metrics['losstissue'] / epoch_samples
                valacctissue = metrics['correcttissue']/(epoch_samples*256*256)
                valIoUtissue = np.mean(metrics["TPtissue"]/(metrics["TPtissue"]+metrics["FPtissue"]+metrics["FNtissue"]))
            else:
                teslosswound = metrics['losswound'] / epoch_samples
                tesaccwound = metrics['correctwound']/(epoch_samples*256*256)
                tesIoUwound = np.mean(metrics["TPwound"]/(metrics["TPwound"]+metrics["FPwound"]+metrics["FNwound"]))
                
                teslosstissue = metrics['losstissue'] / epoch_samples
                tesacctissue = metrics['correcttissue']/(epoch_samples*256*256)
                tesIoUtissue = np.mean(metrics["TPtissue"]/(metrics["TPtissue"]+metrics["FPtissue"]+metrics["FNtissue"]))

                log = log.append({'tralosswound': tralosswound, 'traaccwound': traaccwound, 'traIoUwound': traIoUwound,
                            'vallosswound': vallosswound, "valaccwound": valaccwound, 'valIoUwound': valIoUwound, 
                            'teslosswound': teslosswound, "tesaccwound": tesaccwound, 'tesIoUwound': tesIoUwound,
                            
                            'tralosstissue': tralosstissue, 'traacctissue': traacctissue, 'traIoUtissue': traIoUtissue,
                            'vallosstissue': vallosstissue, "valacctissue": valacctissue, 'valIoUtissue': valIoUtissue, 
                            'teslosstissue': teslosstissue, "tesacctissue": tesacctissue, 'tesIoUtissue': tesIoUtissue}, ignore_index=True)
    
    
                log.to_csv("log.csv")
                # sys.exit()
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['lossall'] / (2*epoch_samples)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                torch.save(model.state_dict(), model_path)
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        scheduler.step()
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path)
    return model

if __name__ == "__main__":

    img_size = 256
    transform = transforms.Compose([transforms.ToTensor()])#多分使わない
    # Dataset を作成する。
    base_dir = "../data2"
    print("load the images and labels")
    train = MyDataset(base_dir, "traaug")
    val = MyDataset(base_dir, "val")
    tes = MyDataset(base_dir, "tes")
    batch_size = 4
    dataloaders = {
        'train': DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val, batch_size=3, shuffle=False, num_workers=0),
        'tes': DataLoader(tes, batch_size=6, shuffle=False, num_workers=0)
    }
    print("Loading finished")
    print(len(train))
    print(len(val))
    print(len(tes))
 
    device = torch.device('cpu')

#    model = ResNetUNet(n_class=4)
    model = multiRoleNet2(n_class=4)
    model = model.to(device)

    # RoleNetFront = RoleNetFront(in_channels=3, n_class=2)
    # model = RoleNetBack(in_channels=3, n_class=21)
    # model = RoleNet(RoleNetFront, RoleNetBack)
    # model = model.to(device)

    #これはモデルのパラメータを算出する処理
    summary(model, input_size=(3, 256, 256))
    print("start training")
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=1.0)
    model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=100)
    print("training finished")
    model_path = 'model.pth'
    print("save the best model")
    torch.save(model.state_dict(), model_path)
    print("finished!!!")
    # for data in dataloaders["train"]:
    #     img,lab = data
    #     print(img.shape,lab.shape)
    #     break
