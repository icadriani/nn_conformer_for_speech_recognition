import torch
import torch.nn as nn
from torch.autograd import Variable

class RelativePositionalEmbeddings(nn.Module):
    '''
        This class initializes the positional embeddings
        
        Inputs:
            shape: tuple; embeddings shape
            relative: bool; If the embeddings must be learned
    '''
    def __init__(self,shape,relative=True):
        super(RelativePositionalEmbeddings,self).__init__()
        self.shape=shape
        # self.dmodel=dmodel
        self.relative=relative
    def forward(self):
        '''
            Initialization of the positional embeddings
            
            Inputs:
                None
            Outputs:
                rel_pos_emb: tensor; positional embeddings
        '''
        # print(x.shape)
        inv_freq=1/(10000**(torch.arange(0,self.shape[-1],2)/self.shape[-1]))
        pos_inv_freq=torch.outer(torch.arange(0,self.shape[-2]),inv_freq)
        rel_pos_emb=torch.cat([pos_inv_freq.sin(),pos_inv_freq.cos()],dim=-1)
        rel_pos_emb=rel_pos_emb.expand(self.shape)
        if self.relative:
            rel_pos_emb.requires_grad=True
        # print(rel_pos_emb.requires_grad)
        # rel_pos_emb=Variable(pos_emb,requires_grad=True)
        return rel_pos_emb#.unsqueeze(1)