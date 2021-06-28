#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 16:47:26 2021

@author: weiyang
"""
import torch
from networks import ShuffleNet_SS
from  datasets import *
import sys
import os
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import utils
import time
import numpy as np
parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset',type=str,default='BC40_MSA_30')
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--model_path',type=str,default="model")
parser.add_argument('--loss_type',type=str,default="CrossEntropyLoss")
# parser.add_argument('--loss_type',type=str,default="LDAMLoss")
parser.add_argument('--num_class',type=int,default=3)
parser.add_argument('--num_blocks',type=int,default=10)
parser.add_argument('--num_channels',type=int,default=384)
parser.add_argument('--p',default = 0.2, type = float)
parser.add_argument('--m',default = 1.0, type = float)
args = parser.parse_args()


use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if args.num_class==8:
    isEightClass=True
else:
    isEightClass=False       
test_list=Load_H5PY('data/%s'%(args.dataset),isEightClass)
num_features=test_list[0].Profile.shape[1]
if args.loss_type=="LDAMLoss":     
    model=ShuffleNet_SS(num_features,args.num_class,num_blocks=args.num_blocks,num_channels=args.num_channels,use_norm=True,p=args.p).to(device) 
    model.load_state_dict(torch.load('%s/LDAM_model_%d.pth'%(args.model_path,args.num_class)))
else:
    model=ShuffleNet_SS(num_features,args.num_class,num_blocks=args.num_blocks,num_channels=args.num_channels,p=args.p).to(device)
    model.load_state_dict(torch.load('%s/CE_model_%d.pth'%(args.model_path,args.num_class)))

start_time = time.time()
if args.num_class==8:
    Q8,F1,accuracy=utils.eval_Q8(args,model,device,test_list,args.batch_size) 
    print("Eight-state predicton result:")
    print("  took {:.3f}s".format(time.time() - start_time))
    print('  Q8: %.2f %% on the dataset %s using %s for training'%(100*Q8,args.dataset,args.loss_type))
    print('  L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    print('  F1: %.2f %% on the dataset  %s'%(100*F1,args.dataset)) 
else:
    Q3,accuracy=utils.eval_Q3(args,model,device,test_list,args.batch_size)
    print("three-state predicton result:")
    print("  took {:.3f}s".format(time.time() - start_time))
    print('  Q3: %.2f %% on the dataset %s using %s for training'%(100*Q3,args.dataset,args.loss_type))
    print('  C: %.2f,E: %.2f,H: %.2f'%(accuracy[0],accuracy[1],accuracy[2]))
    
    
    
    