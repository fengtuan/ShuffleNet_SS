#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 22:54:36 2021

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
from loss import CrossEntropyLossWithMask,LDAMLossWithMask
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR
parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset',type=str,default='BC40_MSA_30')
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--maxEpochs', default =100, type = int)
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--lr', default = 1e-3, type = float, help = 'Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--model_path',type=str,default="ShuffleNet_Q3")
parser.add_argument('--res_dir',type=str,default="results_ShuffleNet_Q3")
parser.add_argument('--loss_type',type=str,default="CrossEntropyLoss")
#parser.add_argument('--loss_type',type=str,default="LDAMLoss")
parser.add_argument('--num_class',type=int,default=3)
parser.add_argument('--num_blocks',type=int,default=10)
parser.add_argument('--num_channels',type=int,default=384)
parser.add_argument('--p',default = 0.2, type = float)
parser.add_argument('--m',default = 1.0, type = float)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed) 
use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:     
    torch.cuda.manual_seed_all(args.seed)
if(os.path.isdir(args.model_path)==False ):
  os.mkdir(args.model_path)   
if(os.path.isdir(args.res_dir)==False ):
  os.mkdir(args.res_dir) 

print("Starting training...")
if args.num_class==8:
    isEightClass=True
else:
    isEightClass=False

       
train_list,valid_list,test_list = Load_DataSet(args.dataset,isEightClass)
num_features=train_list[0].Profile.shape[1]
cls_num_list=np.zeros(shape=(args.num_class),dtype=np.long)
for  i in range(len(train_list)):
    for j in range(train_list[i].ProteinLen):
        cls_num_list[train_list[i].SecondarySeq[j]]+=1
print(cls_num_list) 

if args.loss_type=="LDAMLoss":    
    model=ShuffleNet_SS(num_features,args.num_class,num_blocks=args.num_blocks,num_channels=args.num_channels,use_norm=True,p=args.p).cuda()
    criterion = LDAMLossWithMask(cls_num_list,max_m=args.m, weight=None).cuda()
    FileName=args.res_dir+'/'+args.loss_type+'_b_'+str(args.num_blocks)+'_c_'+str(args.num_channels)+'_m_'+str(args.m)+'_p_'+str(args.p)+'_'+time.strftime('%H-%M-%S',time.localtime(time.time()))+'_'+args.dataset+'.txt'
else:
    model=ShuffleNet_SS(num_features,args.num_class,num_blocks=args.num_blocks,num_channels=args.num_channels,p=args.p).cuda()
    criterion = CrossEntropyLossWithMask(weight=None)
    FileName=args.res_dir+'/'+args.loss_type+'_b_'+str(args.num_blocks)+'_c_'+str(args.num_channels)+'_p_'+str(args.p)+'_'+time.strftime('%H-%M-%S',time.localtime(time.time()))+'_'+args.dataset+'.txt' 
   
param_dict={}
for k,v in model.named_parameters():
    param_dict[k]=v
bn_params=[v for n,v in param_dict.items() if ('bn' in n or 'bias' in n)]
rest_params=[v for n,v in param_dict.items() if not ('bn' in n or 'bias' in n)]
optimizer = torch.optim.AdamW([{'params':bn_params,'weight_decay':0},
                              {'params':rest_params,'weight_decay':args.weight_decay}],
                             lr=args.lr,amsgrad=False)


#############################################################
f = open(FileName, 'w')
# early-stopping parameters
decrease_patience=5
low_increase_patience=5
improvement_threshold = 1.0005  # a relative improvement of this much is
best_accuracy = 0
best_epoch = 10
epoch = 0
Num_Of_decrease=0
Num_Of_Low_Increase=0
done_looping = False  
  
while (epoch < args.maxEpochs) and (not done_looping):
    epoch = epoch + 1
    start_time = time.time()  
    average_loss=utils.train(args,model,device,train_list,optimizer,criterion)
    # scheduler.step()
    print("{}th Epoch took {:.3f}s".format(epoch, time.time() - start_time))
    f.write("{}th Epoch took {:.3f}s\n".format(epoch, time.time() - start_time))
    print("  training loss:\t\t{:.3f}".format(average_loss))
    f.write("  training loss:\t\t{:.3f}\n".format(average_loss))

  
    Q3,accuracy=utils.eval_Q3(args,model,device,valid_list,args.batch_size) 
    print("  validation Q3 accuracy:\t\t{:.2f}".format(100*Q3))
    f.write("  validation Q3 accuracy:\t\t{:.2f}\n".format(100*Q3))    
    print('C: %.2f,E: %.2f,H: %.2f'%(accuracy[0],accuracy[1],accuracy[2]))
    f.write('C: %.2f,E: %.2f,H: %.2f\n'%(accuracy[0],accuracy[1],accuracy[2]))
    f.write('[%.2f,%.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2]))
    f.flush()
    # if we got the best validation Q3 accuracy until now
    if Q3 > best_accuracy:
      #improve patience if loss improvement is good enough
        if (Q3 > best_accuracy *improvement_threshold):
            Num_Of_Low_Increase= 0
        else:
            Num_Of_Low_Increase=Num_Of_Low_Increase+1
        best_accuracy = Q3
        best_epoch = epoch
        Num_Of_decrease=0
        torch.save(model.state_dict(),'%s/%d_epoch_model.pth'%(args.model_path,epoch))     
    else:
        Num_Of_decrease=Num_Of_decrease+1
    if (Num_Of_decrease>decrease_patience or Num_Of_Low_Increase>low_increase_patience):
        done_looping = True    
print('The validation accuracy %.2f %% of the best model in the %i th epoch' 
            %(100*best_accuracy,best_epoch))
f.write('The validation accuracy %.2f %% of the best model in the %i th epoch\n' 
            %(100*best_accuracy,best_epoch))
Eval_FileName=args.res_dir+'/'+args.dataset
model.load_state_dict(torch.load('%s/%d_epoch_model.pth'%(args.model_path,best_epoch)))
start_time = time.time() 
Q3,accuracy=utils.eval_Q3(args,model,device,test_list,args.batch_size) 
print("took {:.3f}s".format(time.time() - start_time))
print('Q3: %.2f %% on the dataset  %s'%(100*Q3,args.dataset))
print('C: %.2f,E: %.2f,H: %.2f'%(accuracy[0],accuracy[1],accuracy[2]))
f.write('C: %.2f,E: %.2f,H: %.2f\n'%(accuracy[0],accuracy[1],accuracy[2]))
f.write('[%.2f,%.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2]))
f.write('Q3:%.2f %%\n'%(100*Q3))
f.close() 
