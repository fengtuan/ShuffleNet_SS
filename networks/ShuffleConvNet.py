#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:28:25 2020

@author: weiyang
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit


@torch.jit.script
def Mish(x):
    x = x * (torch.tanh(F.softplus(x)))
    return x

class MaskedBatchNorm1d(nn.Module):
    "Construct a BatchNorm1d module."
    def __init__(self, num_features, eps=1e-5,momentum=0.1):
        super(MaskedBatchNorm1d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features,1))
        self.bias = nn.Parameter(torch.zeros(num_features,1))
        self.eps = eps
        self.momentum=momentum
        self.register_buffer('running_mean',torch.zeros(num_features,1))
        self.register_buffer('running_var',torch.ones(num_features,1))
        self.C=num_features        
   
    def forward(self, x,masks):
        if self.training:
            m_x=torch.masked_select(x.transpose(0,1),masks.transpose(0,1)).view(self.C,-1).contiguous()
            var,mean=torch.var_mean(m_x,dim=1,keepdim=True)            
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mean.detach()
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*var.detach() 
        else:
            mean=self.running_mean
            var=self.running_var
        out=self.weight *( (x - mean) / (var+self.eps).sqrt()) + self.bias
        return torch.where(masks,out,torch.zeros(size=(1,),device=out.device)) 


def improved_channel_shuffle1(x):
    batchsize, num_channels, length = x.data.size()
    channels_per_group = num_channels // 2
    y=F.pad(x,(0,0,1,1),"constant",0)
    y = y.view(batchsize, 2,channels_per_group+1, length)
    y = torch.transpose(y, 1, 2).contiguous()
    y = y.view(batchsize, -1, length)[:,1:-1,:].contiguous()   
    return y 

def improved_channel_shuffle2(x, groups=2):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, length = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,channels_per_group, length)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, length)
    x1, x2 = x.chunk(2, dim=1)
    x=torch.cat([x2, x1], 1)
    return x

class basic_module(nn.Module):
    def __init__(self, num_channels,p):
        super(basic_module, self).__init__()
        branch_features =num_channels//2
        self.conv_1 = nn.Conv1d(branch_features, branch_features, 1, padding=0,bias=False)
        self.conv_2 = nn.Conv1d(branch_features, branch_features, 3, padding=1,bias=False,groups=branch_features)
        self.conv_3 = nn.Conv1d(branch_features, branch_features, 1, padding=0,bias=False)
        self.dropout = nn.Dropout(p)
        self.bn1 = MaskedBatchNorm1d(branch_features)
        self.bn2 = MaskedBatchNorm1d(branch_features)
        self.bn3 = MaskedBatchNorm1d(branch_features)   

  
    def forward(self, x,masks):
        x1, x2 = x.chunk(2, dim=1)
        branch2 = self.conv_1(x2)
        branch2 = Mish(branch2)
        branch2 = self.bn1(branch2,masks)      
        branch2 = self.bn2(self.conv_2(branch2),masks)
        branch2 = self.conv_3(branch2)
        branch2 = Mish(branch2)
        branch2 = self.bn3(branch2,masks)
        branch2 = self.dropout(branch2)
        out=torch.cat([x1, branch2], 1)
        out=improved_channel_shuffle1(out)
        return out  
class Normalized_FC(nn.Module):
    def __init__(self, num_features, num_classes, tau= 16):
        super(Normalized_FC, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.tau = tau

    def forward(self, x):
        out = F.linear(F.normalize(x), self.tau*F.normalize(self.weight))
        return out 

  
class ShuffleNet_SS(nn.Module):
    def __init__(self,num_features,num_classes,num_blocks,num_channels,use_norm=False,p=0.2):
        super(ShuffleNet_SS, self).__init__()
        self.num_blocks=num_blocks
        out_features=num_channels
        self.conv_1 = nn.Conv1d(num_features, num_channels, 3, padding=1,bias=False)
        self.conv_2 = nn.Conv1d(num_channels, out_features, 1,bias=False)
        self.bn1 = MaskedBatchNorm1d(num_channels)
        self.bn2 = MaskedBatchNorm1d(out_features)           
        self.intermediate_layers = nn.ModuleList(
            [basic_module(num_channels,p) for i in range(num_blocks)])
        if use_norm:
            self.fc = Normalized_FC(num_channels, num_classes)
        else:
            self.fc = nn.Linear(num_channels, num_classes)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)

    def forward(self, x,masks=None):        
        masks=masks.unsqueeze(dim=1)        
        out=self.conv_1(x)
        out= Mish(out)
        out= self.bn1(out,masks)
        out = self.dropout1(out)
        for i in range(self.num_blocks):
            out = self.intermediate_layers[i](out,masks)
        out=self.conv_2(out)
        out= Mish(out)
        out= self.bn2(out,masks)      
        out = out.transpose(1,2).contiguous()
        out = self.dropout2(out)        
        out= self.fc(out)
        return out

   
if __name__=='__main__':
    model=ShuffleNet_SS(20,8,9,256)
    print(model)  
