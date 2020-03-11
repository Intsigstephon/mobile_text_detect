#coding=utf-8
#author: stephon
#time:  2020.03.05

import torch
import torchvision
import torch.nn as nn

import numpy as np 
import pandas as pd  
import cv2
from PIL import Image
from PIL import ImageDraw

import pathlib
import codecs

from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torchvision import utils
from torch.utils import data
from util import PIL_to_tensor
from util import tensor_to_PIL
from model import LBNet
from dataset import MobileTextDetectDataset
from collections import OrderedDict


def init_device():
    """
    run on CPU or GPU
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

class MyLoss(nn.Module):
    def __init__(self, beta= 50.):
        super(MyLoss, self).__init__()        
        self.beta = beta   #positvie loss increase

    def forward(self, predict, truth):
        """
        calucate the loss
        truth: 1 * 224 * 224(float, 0~1)
        predt: 1 * 224 * 224(float, or double or float16 0~1)
        """
        #print("before: {}".format(truth.size()))
        truth = truth.view(-1)
        #print("after: {}".format(truth.size()))
        weight = truth * 0.98  + 0.02   #(0,1)
        #print(weight)
        #print(torch.sum(weight).item()) 
        #exit()

        #get truth
        temp = truth.detach().cpu().numpy()
        #print(temp.shape)
        #print(np.sum(temp))
        #exit()

        # for x  in temp:
        #    if x > 0:
        #        print(x)
        #exit()

        predict = predict.view(-1)   #(0,1)
        #print(predict.detach().cpu().numpy()[:5])
        #print(torch.sum(predict).item())
        #exit()
        #print(predict.size())
        #exit()
        
        #cross-entropy loss
        loss = -torch.mean((truth * torch.log(predict + 0.000001) + (1 - truth) * torch.log(1 - predict + 0.000001)) * weight)
        #print("loss by parallel compute: {}".format(loss.item()))
        #exit()

        # loop
        # loss2 = 0.0
        # for i in range(len(truth)):
        #     loss2 += (truth[i] * torch.log(predict[i]) + (1- truth[i]) * torch.log(1 - predict[i])) * weight[i]
        # loss2 = -loss2 / len(truth)
        # print("loss2 by loop: {}".format(loss2.item()))
        # exit()

        return loss  

def load_part_pretrained_params(model):
    """
    load part parameters
    """
    return 1

def train(device, train_loader, model, criterion, optimizer, num_epoches, fine_tuning, checkpoint = None, valid_loader=None):
    """
    device
    train_loader
    model
    criterion
    optimizer
    num_epoches
    """
    if fine_tuning and checkpoint is not None:
        #1. load param
        #2. get start

        temp = checkpoint.split('_')[1]
        #print(temp)

        start = int(temp[5:]) + 1
        print("!!!!!!!Fine tunnig from {}, start from epoch {}".format(checkpoint, start))
        
        #if Just save weights
        #model.load_state_dict(torch.load(checkpoint))

        #if save net and weights
        #model.load_state_dict(torch.load(checkpoint).state_dict())

        #if defined the paralled model
        state_dict = torch.load(checkpoint)
        
        #print(state_dict.keys())
        #exit()

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+ k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print("parallel model parameters load succeed!!!!")
        #exit()

    else:
        print("train from scratch....")

        #load pretrained parameters
        state_dict = torch.load("./pretrained_mbv3/mbv3_small.pth.tar")['state_dict']   #pretrained 

        #remove module
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]                #remove module
            new_state_dict[name] = v
        #model.load_state_dict(new_state_dict)
        start = 0

        # 1. filter out unnecessary keys
        model_dict = model.state_dict()    
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}  #Just store the key-value: that boths exists

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # 3. freeze these parameters
            

    total_step = len(train_loader)

    #begin to loop
    for epoch in range(start, start + num_epoches):
        for i, (images, labels) in enumerate(train_loader):
            
            #show images
            #print(images.size())
            #print(labels.size())
            #print("oh, oh, oh.......")   

            #images = images.to(device)
            #labels = labels.to(device)

            images = images.cuda()
            labels = labels.cuda()

            # 正向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i) % 5 == 0:
                loss_valid = 0.0
                #calc test loss
                if valid_loader is not None:
                    print("calc loss on valid data, waiting.......")
                    for j, (imgs_valid, labs_valid) in enumerate(valid_loader):
                        imgs_valid = imgs_valid.to(device)
                        labs_valid = labs_valid.to(device)
                        opts_valid = model(imgs_valid)
                        loss_valid += criterion(opts_valid, labs_valid).item()
                    loss_valid /= j

                print("Epoch {} / {}, Step {} / {},  loss: {}, valid_loss: {}".format(epoch + 1, 
                num_epoches + start, 
                i, 
                total_step, 
                loss.item(),
                loss_valid))  

        model_dir = "./model/LBNet/"
        #torch.save(model.state_dict(), model_dir +  'mobile_text_detect_epoch{}_loss{.3f}.pth'.format(epoch, loss.item()))
        torch.save(model.state_dict(), model_dir + "model_epoch{}_loss{:.5f}.pth".format(epoch, loss.item()))

if __name__ == "__main__":

    root = "/home/bb6/stephon/MyDataSetDB/Vietnam/"
    transformer = transforms.Compose([ transforms.Resize((224, 224), interpolation=0),
                                       transforms.ToTensor()])                                  #to tensor will add a batch dim; Actually: B * C * H * W
    target_transformer = transforms.ToTensor()
    train_data = MobileTextDetectDataset(root,  datatxt = 'train.txt',  transform = transformer, target_transform=target_transformer)
    test_data  = MobileTextDetectDataset(root,  datatxt = 'test.txt',   transform = transformer, target_transform=target_transformer)    #should be same dims

    #print(len(train_data.imgpaths))
    #print(len(train_data.imglabels))
    #print(test_data)

    train_loader = data.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_loader  = data.DataLoader(dataset=test_data,  batch_size=64)
    
    #begin to train
    model = LBNet()
    #print(model)

    #use single GPU to train the model
    device = init_device()
    #model.to(device)
    #print(device)

    #use multiple GPUS to train
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)   #model has paralleled
    
    #move the model from CPU to GPU
    if torch.cuda.is_available():
        model.cuda()
    #exit()

    loss = MyLoss()
    # print(type(model.parameters()))
    # for key in model.parameters():
    #     print(key)
    # exit()
    
    #commonly used optimizer:
    #SGD, momentum, RmsProp, Adam;
    optimizer = torch.optim.SGD(model.parameters(),  lr=0.1, momentum=0.9)   #lr = 0.1; 稳步下降
    
    #begin to train
    num_epoches = 200
    fine_tunning = True
    train(device, train_loader, model, loss, optimizer, num_epoches, fine_tunning, "./model/LBNet/model_epoch236_loss0.00188.pth")    #first, freeze the dowmsampling part

