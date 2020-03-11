#coding=utf-8
#author: stephon
#time: 2020.03.10

import torch
import  torchvision

model = torchvision.models.resnet18(pretrained=True)
model.eval()

#get example
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model/jit/model.pt")