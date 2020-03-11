#coding=utf-8
#author: stephon
#time：　2020.03.05

"""
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True)
class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True)
difference between Conv2d and ConvTranspose2d
"""


"""
o = [ (i + 2p - k)/s ] +1 （1）
其中：
O : 为 output size
i: 为 input size
p: 为 padding size
k: 为kernel size
s: 为 stride size
[] 为下取整运算

"""