import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LDAMLossWithMask_PSSP(nn.Module):
    
    def __init__(self, max_m=2.4, weight=None):
        super(LDAMLossWithMask_PSSP, self).__init__()
        margin_list=np.array([0.45357266, 1.,0.49222963,0.76696184,1.,0.43823621, 0.60325897,0.57481898])
        m_list = torch.cuda.FloatTensor(max_m*margin_list)
        self.m_list = m_list
        self.weight = weight

    def forward(self, x, target,mask):
        target=target.flatten()
        mask=mask.flatten()
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        batch_m = self.m_list[target].view(-1,1)
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        unreduction_loss=F.cross_entropy(output, target, weight=self.weight,reduction='none')
        if self.weight is not None:
            w=(self.weight[target]*mask).sum()
        else:
            w=float(mask.sum())
        loss=torch.sum(torch.masked_select(unreduction_loss,mask))/w 
        return loss

class LDAMLossWithMask(nn.Module):
    
    def __init__(self, cls_num_list, max_m=2.4, weight=None):
        super(LDAMLossWithMask, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list)) 
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.weight = weight

    def forward(self, x, target,mask):
        target=target.flatten()
        mask=mask.flatten()
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        batch_m = self.m_list[target].view(-1,1)
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        unreduction_loss=F.cross_entropy(output, target, weight=self.weight,reduction='none')
        if self.weight is not None:
            w=(self.weight[target]*mask).sum()
        else:
            w=float(mask.sum())
        loss=torch.sum(torch.masked_select(unreduction_loss,mask))/w 
        return loss

class CrossEntropyLossWithMask(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLossWithMask, self).__init__()
        self.weight = weight

    def forward(self, x, target,mask):
        target=target.flatten()
        mask=mask.flatten()        
        unreduction_loss=F.cross_entropy(x, target,weight=self.weight,reduction='none')
        if self.weight is not None:
            w=(self.weight[target]*mask).sum()
        else:
            w=float(mask.sum())
        loss=torch.sum(torch.masked_select(unreduction_loss,mask))/w         
        return loss
    
