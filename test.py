#coding=utf-8
#author: stephon
#time: 2020.03.07

import os
import argparse

import torch
import torchvision
from  PIL import Image
from train import init_device
from torchvision import  transforms
from model import LBNet
def main():
    """
    test function
    1. single test
    2. evalute
    """
    #with no argparser
    imgpath = "../../MyDataSetDB/Vietnam/image/187719370.jpg"
    img = Image.open(imgpath).convert("RGB")

    #model
    model = LBNet()
    model.load_state_dict(torch.load("./model/LBNet/model_epoch96_loss0.00198.pth")) #2.4M
    device = init_device()
    print(device)
    model.to(device)

    #do not use BatchNorm or dropout
    model.eval()

    transformer = transforms.Compose([
        transforms.Resize((224, 224), interpolation=0),
        transforms.ToTensor()
    ])
    
    img_gpu = transformer(img)
    print(type(img_gpu))
    print(img_gpu.size())

    img_gpu = torch.unsqueeze(img_gpu, 0)
    print(type(img_gpu))
    print(img_gpu.size())
    print(img_gpu.device)
    img_gpu = img_gpu.to(device)
    print(img_gpu.device)
    #exit()

    #infer
    result = model(img_gpu)
    print(type(result))
    print(result.size())

    #get result from GPU to cpu
    #rslt = result.cpu().detach().numpy().squeeze(0)
    from util import tensor_to_PIL
    rslt = tensor_to_PIL(result)
    print(type(rslt))
    rslt.save("rslt.jpg")

    #do binary: thresh
    thresh = 128
    #print(rslt.size)
    size = rslt.size
    for i in range(size[0]):
        for j in range(size[1]):
            temp = rslt.getpixel((i, j))
            print("the value is: {}".format(temp))
            value = 255 if temp > thresh else 0
            rslt.putpixel((i, j), value)
    rslt.save("binary.jpg")

if __name__=="__main__":
    
    device = torch.device("cuda:0")
    print(type(device))
    print(device)
    #exit()

    main()


    #record:
    #1. Pick a model to test; Should use model.eval(): No BatchNorm or Dropout; it is OK.
    #2. optim:
    #3. loss:
    #4. learning_decay:
    #5. parallel: data parallel or model parallel?
    #6. how to transform to onnx format. 
    #7. use mnn framework to test the model performance.