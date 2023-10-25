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
    # def contrastive_loss(self,logits,target):
    #     #print(x.shape)
    #     loss=self.sim(logits.squeeze(),target.squeeze())
    #     loss=loss/self.hp.temperature_loss
    #     loss=torch.exp(loss)
    #     distractors=[]
    #     distractors=torch.randperm(loss.shape[0]-1)
    #     distractors=torch.split(distractors,self.hp.distractors_K)[0]
    #     distractors=loss[distractors]
    #     distractors=torch.sum(distractors)
    #             # print(perm)
    #     # perms=torch.stack(perms)
    #     loss=loss/distractors
    #     # loss=self.softmax(loss)
    #     loss=-torch.log(loss)
    #     # print(loss.shape)
    #     loss=torch.mean(loss,-1)
    #     loss=torch.mean(loss)
    #     return loss
    def contrastive_loss(self,logits,target):
        '''
            Computes the contrastive loss
            
            Input:
                logits: tensor; the input logits
                target: tensor; ground truth
            Output:
                loss: tensor; the contrastive loss
        '''
        #print(x.shape)
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
                # print(perm)
                perm=torch.split(perm,self.hp.distractors_K)[0]
                # print(perm,t)
                perm=torch.cat([perm,t])
                # print(perm)
                perm=u[perm]
                # perm.dtype=torch.int64
                # print(loss[u].shape,perm.shape)
                perm=torch.sum(perm)
                distractors[-1].append(perm)
                # print(perm)
        # perms=torch.stack(perms)
        distractors=torch.FloatTensor(distractors).to(self.hp.device)
        loss=loss/distractors
        # loss=self.softmax(loss)
        # print(loss.shape)
        loss=torch.mean(loss,-1)
        loss=torch.mean(loss)
        loss=-torch.log(loss)
        return loss
    # def diversity_loss(self,logits,target):
        # pass
    def forward(self,logits,target):
        '''
            Computes the loss
            
            Input:
                logits: tensor; the input logits
                target: tensor; ground truth
            Output:
                loss: tensor; the loss
        '''
        # x=target.ge(1,)
        # target=torch.masked_select(target, target)
        # logits=torch.split(logits,len(target))[0]
        loss=self.contrastive_loss(logits,target)
        if not self.hp.simplified_pretraining:
            loss=loss+self.hp.alpha_loss*self.diversity_loss(target)
        return loss