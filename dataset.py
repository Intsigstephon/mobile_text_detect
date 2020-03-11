#coding=utf-8
#author: stephon
#time:  2020.03.05

import torch
import torchvision

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
from graphic import BresenhamLine
import cv2

"""
format: 
image_label:  aa.jpg, aa.txt; bb.jpg, bb.txt;  
train.txt: Imagename without postfix, for example aa, bb, cc(relative path)
text.txt:  same format as train()
*.txt:    x1,y1,x2,y2,x3,y3,x4,y4,####;.......
Include Indonesia, Vietnam, Mexico 
"""

#box type: AABB, RBOX, QUAD
#0: AABB
#1. RBOX
#2. QUAD

box_type = 0
def get_label_image(img, boxes, value=255):
    """
    img: PIL Image
    box: [[]]
    """
    #print(img.size)
    mask = Image.new("L", img.size, 0)

    if box_type == 0:
        for box in boxes:
            xmin = min(box[0], box[2], box[4], box[6])
            xmax = max(box[0], box[2], box[4], box[6])
            
            ymin = min(box[1], box[3], box[5], box[7])
            ymax = max(box[1], box[3], box[5], box[7])

            for i in range(xmin, xmax + 1):
                for j in range(ymin, ymax + 1):
                    mask.putpixel((i,j), value)
    
    if box_type == 2:
        for box in boxes:
            #two line abs(k) <=1
            print(box)
            up = {}
            points = BresenhamLine(box[0], box[1], box[2], box[3])
            for pt in points:
                mask.putpixel(pt, value)  
                up[pt[0]] = pt[1]

            down = dict()
            points = BresenhamLine(box[6], box[7], box[4], box[5])
            for pt in points:
                mask.putpixel(pt, value)
                down[pt[0]] = pt[1]
            
            #midlle rect
            start = max(box[0], box[6])
            end   = min(box[2], box[4])
            print(start, end)

            for i in range(start, end + 1):
                for j in range(up[i], down[i]):
                    mask.putpixel((i,j), value)
            #exit()

            #two vertical line
            points = BresenhamLine(box[3], box[2], box[5], box[4])
            for pt in points:
                mask.putpixel((pt[1], pt[0]), value)
            
            points = BresenhamLine(box[1], box[0], box[7], box[6])
            for pt in points:
                mask.putpixel((pt[1], pt[0]), value)

    return mask
    

class MobileTextDetectDataset(torch.utils.data.Dataset): 
    """
    build my dataset(Vietnam), used to do mobile text detection
    """
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        """
        do init: May include the dataset_type
        """
        super(MobileTextDetectDataset, self).__init__()
        self.root = root
        self.labeltxt = datatxt
        self.imgpaths = []
        self.imglabels = []

        #get all imgpaths
        with codecs.open(self.root + self.labeltxt, 'r', 'utf-8') as r:
            alllines = r.readlines()
            for line in alllines:
                line = line.rstrip()
                self.imgpaths.append(line)

        #get all imglabels
        for i in range(len(self.imgpaths)):
            imgpath = self.imgpaths[i]
            imgtxt = imgpath.replace('jpg', 'txt')
            with codecs.open(self.root + imgtxt, 'r', 'utf-8') as r:
                boxes = []
                for line in r.readlines():
                    line = line.rstrip()
                    box = [int(x) for x in line.split(',')[:8]]
                    boxes.append(box)
            self.imglabels.append(boxes)
        
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        index: image index
        """
        imgpath = self.imgpaths[index] 
        img = Image.open(self.root + imgpath).convert('RGB')   #convert to RGB
        #print(img.size)

        label = np.array(self.imglabels[index])
        
        #get label_image
        mask = get_label_image(img, label, 255)

        #label_mask.save("mask.jpg")

        if self.transform is not None:
            img  = self.transform(img)
            mask = self.transform(mask) 

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, mask
 
    def __len__(self): 
        return len(self.imgpaths)
 
if __name__ == "__main__":

    #show the backend
    print(torchvision.get_image_backend())

    #def parameters
    root = "/home/bb6/stephon/MyDataSetDB/Vietnam/"
    transformer = transforms.Compose([ transforms.Resize((224, 224), interpolation=0),          #should be 0: nearestneighbour
                                       transforms.ToTensor()])                                  #to tensor will add a batch dim; Actually: B * C * H * W
    target_transformer = transforms.ToTensor()
    train_data = MobileTextDetectDataset(root,  datatxt = 'train.txt',  transform = transformer, target_transform=target_transformer)
    test_data  = MobileTextDetectDataset(root,  datatxt = 'test.txt',   transform = transformer, target_transform=target_transformer)    #should be same dims

    print(len(train_data.imgpaths))
    print(len(train_data.imglabels))
    #print(test_data)

    train_loader = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    test_loader  = data.DataLoader(dataset=test_data,  batch_size=1)

    for batch_index, batch in enumerate(train_loader):
        print(batch_index)
        imgs, masks = batch
        print("img:>>>>>>>>")
        print("num: {}".format(len(imgs)))
        print(type(imgs))
        print(imgs.size())

        print("value is: {}".format(imgs[0,:, 150,150]))   #0,1

        #labels
        # print("label:>>>>>>>>")
        # print(type(labels))
        # print(len(labels))
        # print(labels.size())

        print("label mask:>>>>>>>>")
        print(type(masks))
        print(len(masks))
        print(masks.size())

        #show the value that > 0
        num = 0
        for i in range(224):
            for j in range(224):
                if masks[0,:, i,j] > 0.01:
                    #print("i: {}, j:{}, value is: {}".format(i, j, masks[0,:, i, j]))  #0,1
                    num +=1
        print("positive is: {}; percent:{:.3f}".format(num, num / 224.0 / 224.0))            
        exit()

        #label: tensor to list or numpy
        #labels = labels.cpu().clone().numpy().squeeze(0).squeeze(0)   

        #1. use batch_index
        if batch_index == 0:
            im = tensor_to_PIL(imgs)
            im.save("a.jpg")

            #draw lines
            # draw = ImageDraw.Draw(im)
            # for i in range(len(labels)):
            #     box = labels[i]
            #     print(box)
            #     draw.polygon([(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])], outline=(255, 0, 0))
            # im.save("a_box.jpg")

            #show the mask
            mask = tensor_to_PIL(masks)
            mask.save("a_mask.jpg")

            break





