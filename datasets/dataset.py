# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 11:16:40 2017

@author: WeiYang
"""

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from enum import Enum
np.random.seed(0)

class ProteinNode:
    def __init__(self,ProteinID,ProteinLen,PrimarySeq,SecondarySeq,Profile):
        self.ProteinID=ProteinID
        self.ProteinLen=ProteinLen
        self.PrimarySeq=PrimarySeq        
        self.SecondarySeq=SecondarySeq
        self.Profile=Profile
 
def Load_H5PY(FilePath,isEightClass,dataset=None):
    SS=['L','B','E','G','I','H','S','T']
    SS8_Dict=dict(zip(SS,range(len(SS))))
    SS3_Dict={'L': 0, 'B': 1, 'E': 1, 'G': 2, 'I': 2, 'H': 2, 'S': 0, 'T': 0}
    Standard_AAS=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N','P','Q', 'R', 'S', 'T', 'V',  'W', 'Y']
    AA_Dict={}
    Non_Standard_AAS=list('BJOUXZ')
    for i,AA in enumerate(Standard_AAS):
        AA_Dict.update({AA:i})
    for i,AA in enumerate(Non_Standard_AAS):
        AA_Dict.update({AA:20})   
    f = h5py.File(FilePath+'.h5', 'r')
    
    if dataset=='CB433':
        idxs=f['NewCB513_filtered_idxs'][()]
    elif dataset=='CB513':
        idxs=f['OldCB513_filtered_idxs'][()]                    
    elif dataset=='CASP10':
        idxs=f['CASP10_filtered_idxs'][()]          
    elif dataset=='CASP11':
        idxs=f['CASP11_filtered_idxs'][()]           
    elif dataset=='CASP12':
        idxs=f['CASP12_filtered_idxs'][()]           
    elif dataset=='CASP13':
        idxs=f['CASP13_filtered_idxs'][()]
    elif dataset=='CASP14':
        idxs=f['CASP14_filtered_idxs'][()]
    elif dataset=='BC40_MSA_30':
        idxs=f['BC40_MSA_30_filtered_idxs'][()]         
    elif dataset=='PISCES_Data':
        idxs=f['random_idxs'][()]            
    else:
        NumOfSamples=len(f.keys())//4     
        idxs=range(NumOfSamples)    
    

    Data=[]
    for i in idxs:
        ProteinID=f['ID'+str(i)][()]
        #ProteinID=ProteinID.decode()
        PrimarySeq=f['PS'+str(i)][()]
        #PrimarySeq=PrimarySeq.decode()
        ProteinLen=len(PrimarySeq)
        PrimarySeq=[AA_Dict[e] for e in PrimarySeq]        
        SecondarySeq=f['SS'+str(i)][()]
        #SecondarySeq=SecondarySeq.decode()
        if isEightClass:
            SecondarySeq=[SS8_Dict[e] for e in SecondarySeq]
        else:
            SecondarySeq=[SS3_Dict[e] for e in SecondarySeq]
        PrimarySeq=torch.tensor(PrimarySeq,dtype=torch.long)
        SecondarySeq=torch.tensor(SecondarySeq,dtype=torch.long)
        Profile=torch.from_numpy(f['Profile'+str(i)][()])       
        one_hot_profile=F.one_hot(PrimarySeq,num_classes=21).float()
        Node=ProteinNode(ProteinID,ProteinLen,PrimarySeq,SecondarySeq,torch.cat([Profile,one_hot_profile],dim=1))
        Data.append(Node)       
    f.close()
    return Data



def BC40_MSA_30(isEightClass):  
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='BC40_MSA_30')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/BC40_MSA_30',isEightClass)
   return train_list,valid_list,test_list


def CB433(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CB433')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CB433',isEightClass)
   return train_list,valid_list,test_list

def CB513(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CB513')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)   
   test_list=Load_H5PY('data/CB513',isEightClass)
   return train_list,valid_list,test_list

def CASP10(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CASP10')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CASP10',isEightClass)
   return train_list,valid_list,test_list

def CASP11(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CASP11')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CASP11',isEightClass)
   return train_list,valid_list,test_list

def CASP12(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CASP12')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CASP12',isEightClass)
   return train_list,valid_list,test_list

def CASP13(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CASP13')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CASP13',isEightClass)
   return train_list,valid_list,test_list

def CASP14(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CASP14')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CASP14',isEightClass)
   return train_list,valid_list,test_list

def PISCES_Data(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='PISCES_Data')
   train_list=Data[:-1024]
   valid_list=Data[-1024:-512] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Data[-512:]
   test_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   return train_list,valid_list,test_list

def Load_DataSet(DataSet,isEightClass):
    if DataSet=='BC40_MSA_30':
        return BC40_MSA_30(isEightClass)
    elif  DataSet=='CASP13':
        return CASP13(isEightClass)
    elif  DataSet=='CASP14':
        return CASP14(isEightClass)
    elif DataSet=='CB433':
        return CB433(isEightClass)
    elif DataSet=='CB513':
        return CB513(isEightClass)                   
    elif DataSet=='CASP10':
        return CASP10(isEightClass)     
    elif DataSet=='CASP11':
        return CASP11(isEightClass)          
    elif DataSet=='CASP12':
        return CASP12(isEightClass)         
    elif DataSet=='CASP13':
        return CASP13(isEightClass)           
    elif DataSet=='PISCES_Data':
        return PISCES_Data(isEightClass)             
    else:
        pass

