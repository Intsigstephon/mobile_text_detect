#coding=utf-8
#author: stephon
#time : 2020.03.06

from mobilenetv3 import hswish, hsigmoid, SeModule, Block
from torch import nn
import torchvision
from torch.nn import init
import torch
import torch.nn.functional as F
from torchviz import make_dot


class better_upsampling(nn.Module):
    """
    双线性插值、转置卷积、上采样（unsampling）和上池化（unpooling
    可以进行优化，对应一个更好的上采样的方案;
    """
    def __init__(self, in_ch, out_ch, scale_factor):
        super(better_upsampling, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x):

        #do upsampling
        x = self.up(x)

        #same conv
        x = F.pad(x, (3 // 2, int(3 / 2),
                    3 // 2, int(3 / 2)))
        x = self.conv(x)

        return x

class UBlock(nn.Module):
    """
    Upsampling or Uppooling
    expand + depthwise + pointwise
    senet
    """
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, upsample):
        """
        3 * 3: in, expand, out, nolinear, stride
        """
        super(UBlock, self).__init__()
        self.se = semodule
        self.up = upsample

        #1 * 1: channel expand 24->96.
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)  #do normalization
        self.nolinear1 = nolinear    #nolinear

        #3 * 3 or 5 * 5: expand -> expand; use same conv; channel wise.  kernel: 3 * 3 * 1 * 1 * expand_size ; Actually no channel communication.
        #if stride > 1: then the feature h,w will decrease.
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        
        #1 * 1: 96->12;
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        #no linear

        #def a shortcut
        self.shortcut = nn.Sequential()
        if in_size != out_size:    #it means: the feature map is not changed
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),    #in_size != out_size, change no feature map.
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        #upsampling
        if self.up is not None:
            y = self.up(x)
        else:
            y = x

        #do conv1
        out = self.nolinear1(self.bn1(self.conv1(y)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.se != None:
            out = self.se(out)   
        
        #this is called Restnet
        #concanate: means feature fusion.
        out = out + self.shortcut(y) 

        return out

class LBNet(nn.Module):
    def __init__(self):
        """
        FCN, but use SE and Block
        conv and deconv
        input: 224 * 224; 
        if input size is not fixed, then the network can only train image_by_image(batch =1)
        size fixed: train parallel
        """
        super(LBNet, self).__init__()     #3 * 224 * 224

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)  #16 * 112 * 112
        self.bn1 = nn.BatchNorm2d(16)                 #16 * 112 * 112
        self.hs1 = hswish()               #16 * 112 * 112
        
        #bottlne neck
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),     #96 * 7 *７     
        )

        """
        实现从96 * 7 * 7 到１* 224 * 224的转化; 这样一个上采样的过程;
        逐步：　7, 14, 28, 56, 112, 224.
        """

        self.Ubneck = nn.Sequential(
            UBlock(5, 96, 576, 96, hswish(), SeModule(96), None),
            UBlock(5, 96, 576, 96, hswish(), SeModule(96), None),
            UBlock(5, 96, 288, 48, hswish(), SeModule(48), nn.Upsample(scale_factor= 2, mode='bilinear', align_corners=True)),
            UBlock(5, 48, 144, 48, hswish(), SeModule(48), None),
            UBlock(5, 48, 120, 40, hswish(), SeModule(40), None),
            UBlock(5, 40, 240, 40, hswish(), SeModule(40), None),
            UBlock(5, 40, 240, 40, hswish(), SeModule(40), None),
            UBlock(5, 40, 96, 24, hswish(),  SeModule(24), nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True)),
            UBlock(3, 24, 88, 24, nn.ReLU(inplace=True),   None, None),
            UBlock(3, 24, 72, 16, nn.ReLU(inplace=True),   None, nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True)),
            UBlock(3, 16, 16, 16, nn.ReLU(inplace=True),   SeModule(16), nn.Upsample(scale_factor = 2, mode="bilinear",align_corners=True)),  
            UBlock(3, 16, 16, 16, nn.ReLU(inplace=True),   SeModule(16), None),  
            UBlock(3, 16, 16, 4, nn.ReLU(inplace=True),    SeModule(4),  nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True)),   #upsmapling
            UBlock(3, 4, 16, 4, nn.ReLU(inplace=True),     SeModule(4), None),     
        )

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>frist version<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #deconv1
        # self.deup1 = nn.Upsample(scale_factor=2, mode='bilinear')    #96 * 14 * 14
        # self.deconv1 = nn.ConvTranspose2d(576, 96, kernel_size=1, stride=1, padding=0, bias=False)   #96 * 7 * 7
        # self.debn1 = nn.BatchNorm2d(96)
        # self.dehs1 = hswish()

        # #deconv2
        # self.deconv2 = nn.ConvTranspose2d(96, 48, kernel_size=1, stride=1, padding=0, bias=False)   #48 * 14 * 14
        # self.debn2 = nn.BatchNorm2d(48)
        # self.dehs2 = hswish()
        # self.deup2 = nn.Upsample(scale_factor=2)    #48 * 28 * 28

        # #deconv3
        # self.deconv3 = nn.ConvTranspose2d(48, 24, kernel_size=1, stride=1, padding=0, bias=False)   #24 * 28 * 28
        # self.debn3 = nn.BatchNorm2d(24)
        # self.dehs3 = hswish()
        # self.deup3 = nn.Upsample(scale_factor=2)    #24 * 56 * 56

        # #deconv4
        # self.deconv4 = nn.ConvTranspose2d(24, 12, kernel_size=1, stride=1, padding=0, bias=False)   #12 * 56 * 56
        # self.debn4 = nn.BatchNorm2d(12)
        # self.dehs4 = hswish()
        # self.deup4 = nn.Upsample(scale_factor=2)    #12 * 112 * 112

        # #deconv5
        # self.deconv5 = nn.ConvTranspose2d(12, 6, kernel_size=1, stride=1, padding=0, bias=False)   #6 * 112 * 112
        # self.debn5 = nn.BatchNorm2d(6)
        # self.dehs5 = hswish()
        # self.deup5 = nn.Upsample(scale_factor=2)   #6 * 224 * 224
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>frist version<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # #deconv6
        self.dconv2 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)   #1 * 224 * 224
        self.dhs2 = nn.Sigmoid()  #if use hsigmoid, then will be nan

        self.init_params()

    def init_params(self):
        """
        init param
        """        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        do forward: to generate [2, 1, 224, 224]
        """
        out = self.hs1(self.bn1(self.conv1(x)))

        #print(out.size())
        #print("33333")
        #exit()

        out = self.bneck(out)
        #print(out.size())
        #print("44444")

        #out = self.hs2(self.bn2(self.conv2(out)))
        #print(out.size())
        #print("55555")

        #do upsample
        #out = self.deup1(self.dehs1(self.debn1(self.deconv1(out))))
        #print(out.size())
        #out = self.deup2(self.dehs2(self.debn2(self.deconv2(out))))
        #print(out.size())
        #out = self.deup3(self.dehs3(self.debn3(self.deconv3(out))))
        #print(out.size())
        #out = self.deup4(self.dehs4(self.debn4(self.deconv4(out))))
        #print(out.size())
        #out = self.deup5(self.dehs5(self.debn5(self.deconv5(out))))
        #print(out.size())
        #out = self.dehs6(self.deconv6(out))
        #print(out.size())

        out = self.Ubneck(out)
        out = self.dhs2(self.dconv2(out))

        return out
    
def test():
    net = LBNet()
    print(net)
    
    x = torch.randn(2,3,224,224)   #Actually:  224 * 224 * 3, batch = 2
    y = net(x)
    print(y.size())

def visulize(model, x):
    """
    see the net architecture
    """
    model.eval()
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)])).view()

def print_model(model):
    """
    show the size of 
    1. each layer 
    2. num of parameters
    """
    k = 0
    for key, i in model.state_dict().items():
        print(key)
        l = 1
        print("\t\t结构:"  + str(list(i.size())))
        for j in i.size():
            l *= j
        print("\t\t参数和：" + str(l))
        k = k + l

    print("总参数数量和：" + str(k))

if __name__ == "__main__":
    #test()

    visulize(LBNet(), torch.randn(2,3,224,224))
    #print_model_arch_params(LBNet())