#coding=utf-8
#author: stephon
#time：　2020.03.05

import cv2
import os
import torch
import torchvision
from PIL import Image
import numpy as np

from torchvision import transforms
from matplotlib import pyplot as plt
import os.path as osp

"""
switch between PIL, numpy(cv2), Tensor
"""

"""
numpy格式是使用cv2，也就是python-opencv库读取出来的图片格式;
需要注意的是用python-opencv读取出来的图片和使用PIL读取出来的图片数据略微不同，经测试用python-opencv读取出来的图片在训练时的效果比使用PIL读取出来的略差一些
conclusion: 推荐使用PIL
"""

#PIL to tensor
loader = transforms.Compose([
    transforms.ToTensor()]) 

#tensor to PIL
unloader = transforms.ToPILImage()

# 输入图片地址
# 返回tensor变量
def init_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

devie = init_device()

def image_loader(image_name):
  image = Image.open(image_name).convert('RGB')
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

# 输入PIL格式图片
# 返回tensor变量
def PIL_to_tensor(image):
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
  image = tensor.cpu().clone()
  image = image.squeeze(0)
  image = unloader(image)
  return image

# 直接展示tensor格式图片
def imshow(tensor, title=None):
  image = tensor.cpu().clone() # we clone the tensor to not do changes on it
  image = image.squeeze(0)     # remove the fake batch dimension
  image = unloader(image)
  plt.imshow(image)
  if title is not None:
      plt.title(title)
  plt.pause(0.001)             # pause a bit so that plots are updated

#直接保存tensor格式图片
def save_image(tensor, **para):
    """
    **para: dict(key-value)
    """
    dir = 'results'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    if not osp.exists(dir):
        os.makedirs(dir)

    #save to the results dir
    image.save('results_{}/s{}-c{}-l{}-e{}-sl{:4f}-cl{:4f}.jpg'
            .format(1, para['style_weight'], para['content_weight'], para['lr'], para['epoch'],
                para['style_loss'], para['content_loss']))


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>numpy 与 tensor<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#numpy转化为tensor
#numpy也就是OpenCv读取出来的格式: BGR
def toTensor(img):
  assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # to RGB;  H * W * C
  img = torch.from_numpy(img.transpose((2, 0, 1)))    # H * W * C to C * H * W
  return img.float().div(255).unsqueeze(0)            # 255也可以改为256, (0~1) and float

#tensor转化为numpy
def tensor_to_np(tensor):
  img = tensor.mul(255).byte()
  img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
  return img

#展示numpy格式图片
def show_from_cv(img, title=None):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.figure()
  plt.imshow(img)
  if title is not None:
    plt.title(title)
  plt.pause(0.001)

#展示tensor格式图片
def show_from_tensor(tensor, title=None):
  img = tensor.clone()
  img = tensor_to_np(img)
  plt.figure()
  plt.imshow(img)
  if title is not None:
    plt.title(title)
  plt.pause(0.001)

#将 N x H x W X C 的numpy格式图片转化为相应的tensor格式
def toTensor(img):
  img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
  return img.float().div(255).unsqueeze(0)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PIL.Image 与 cv2.imread互相转化<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#PIL.Image转换成OpenCV格式：
def Image2CV(path):
    img = Image.open(path).convert("RGB")       #.convert("RGB")可不要，默认打开就是RGB
    #img.show()

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #cv2.imshow("OpenCV", img)
    #cv2.waitKey()
    return img

def CV2Image(path):
    img = cv2.imread("plane.jpg", 1)  #BGR 
    #cv2.imshow("OpenCV", img)
    #cv2.waitKey(0)
    
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #image.show()
    return image

def isCV(img):
    """
    判断是否为cv2的图像格式
    """
    return isinstance(img, np.ndarray)

if __name__ == "__main__":

    #1. device
    device = init_device()
  

