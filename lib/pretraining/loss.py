import torch
import torch.nn as nn
from random import randint
from pytorch_adapt.layers.diversity_loss import DiversityLoss

class PretrainingLoss(nn.Module):
    '''
        class PretrainingLoss
        
        Loss to be applied during the pretraining phase
        
        Inputs:
            hp: HParams; the class of hyperparameters
        Outputs:
            None
    '''    
    def __init__(self,hp):
        super(PretrainingLoss,self).__init__()
        self.hp=hp
        self.sim=nn.CosineSimilarity()
        self.softmax=nn.Softmax(-1)
        if not hp.simplified_pretraining:
            self.diversity_loss=DiversityLoss()
    def contrastive_loss(self,logits,target):
        '''
            Computes the contrastive loss
            
            Input:
                logits: tensor; the input logits
                target: tensor; ground truth
            Output:
                loss: tensor; the contrastive loss
        '''
        loss=self.sim(logits,target)
        loss=loss/self.hp.temperature_loss
        loss=torch.exp(loss)
        distractors=[]
        for u in loss:
            distractors.append([])
            for i in range(len(u)):
                perm=torch.randperm(loss.shape[1]-1)
                t=perm[perm==i]
                perm=perm[perm!=i]
                perm=torch.split(perm,self.hp.distractors_K)[0]
                perm=torch.cat([perm,t])
                perm=u[perm]
                perm=torch.sum(perm)
                distractors[-1].append(perm)
        distractors=torch.FloatTensor(distractors).to(self.hp.device)
        loss=loss/distractors
        loss=torch.mean(loss,-1)
        loss=torch.mean(loss)
        loss=-torch.log(loss)
        return loss
    def forward(self,logits,target):
        '''
            Computes the loss
            
            Input:
                logits: tensor; the input logits
                target: tensor; ground truth
            Output:
                loss: tensor; the loss
        '''
        loss=self.contrastive_loss(logits,target)
        if not self.hp.simplified_pretraining:
            loss=loss+self.hp.alpha_loss*self.diversity_loss(target)
        return loss