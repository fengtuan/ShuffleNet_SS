#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:43:28 2020

@author: weiyang
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import itertools


def iterate_minibatches(ProteinLists, batchsize,shuffle=True):
    num_features=ProteinLists[0].Profile.shape[1]
    indices = np.arange(len(ProteinLists))
    if shuffle:        
        np.random.shuffle(indices)   
    maxLength=0
    inputs=torch.zeros(size=(batchsize,4096,num_features),dtype=torch.float32)
    masks=torch.zeros(size=(batchsize,4096),dtype=torch.bool)
    targets=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    for idx in range(len(ProteinLists)):
        if idx % batchsize==0:
            inputs.fill_(0)
            masks.fill_(False)
            targets.fill_(0)
            batch_idx=0          
            maxLength=0
        length=ProteinLists[indices[idx]].ProteinLen        
        masks[batch_idx,:length]=True
        inputs[batch_idx,:length,:]=ProteinLists[indices[idx]].Profile[:,:]
        targets[batch_idx,:length]=ProteinLists[indices[idx]].SecondarySeq[:]
        batch_idx+=1
        if length>maxLength:
                maxLength=length
        if (idx+1) % batchsize==0:
            yield inputs[:,:maxLength,:].transpose(1,2),targets[:,:maxLength],masks[:,:maxLength]
    if len(ProteinLists) % batchsize!=0:        
        yield inputs[:,:maxLength,:].transpose(1,2),targets[:,:maxLength],masks[:,:maxLength]


def train(args,model,device,train_list,optimizer,criterion):
    model.train()
    total_loss=0.0
    count=0
    for batch in iterate_minibatches(train_list,args.batch_size, shuffle=True):
        inputs, targets,masks = batch
        inputs=inputs.to(device)
        targets=targets.to(device)
        masks=masks.to(device)
        optimizer.zero_grad()
        outputs=model(inputs,masks)
        outputs=outputs.view(-1,outputs.size(2))
        loss=criterion(outputs,targets,masks)       
        total_loss += loss.item()
        count+=1
        loss.backward()
        optimizer.step()
    return total_loss/count 



def eval_Q8(args,model,device,eval_list,batch_size):
    model.eval()
    correct_num=0
    total_num=0 
    labels_true=np.array([])
    labels_pred=np.array([])
    with torch.no_grad():
        for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)            
            outputs=model(inputs,masks)      
            pred_labels=torch.argmax(outputs, dim=2) 
            correct_num+=torch.sum(torch.eq(pred_labels, targets)*masks)           
            total_num+=masks.sum() 
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]
                if L>0: 
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))
    Q8=correct_num / total_num.float()
    F1=f1_score(labels_true,labels_pred,average='macro',labels=np.unique(labels_pred)) 
    
    class_correct=list(0. for i in range(8))
    class_total=list(0. for i in range(8))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T']
    accuracy=[]
    for i in range(8):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))       
    return Q8.item(),F1,accuracy

def eval_Q3(args,model,device,eval_list,batch_size):
    model.eval()
    correct_num=0
    total_num=0 
    labels_true=np.array([])
    labels_pred=np.array([])
    with torch.no_grad():
        for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)            
            outputs=model(inputs,masks)      
            pred_labels=torch.argmax(outputs, dim=2) 
            correct_num+=torch.sum(torch.eq(pred_labels, targets)*masks)           
            total_num+=masks.sum() 
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]
                if L>0: 
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))
    Q3=correct_num / total_num.float()    
    class_correct=list(0. for i in range(3))
    class_total=list(0. for i in range(3))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
    accuracy=[]
    for i in range(3):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))        
    return Q3.item(),accuracy




