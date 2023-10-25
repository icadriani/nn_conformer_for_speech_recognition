import torch
import torch.nn as nn
from torchaudio.models import Conformer
from lib.convsubsampling import ConvSubSampling
import random
from math import floor
import numpy as np
from torchvision.models import convnext_tiny
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2ConformerConfig

class ASRNN(nn.Module):
    '''
        class ASRNN
        
        The main model used for the automatic speech recognition in this project
        
        Inputs:
            hp: HParams; the class of hyperparameters
        Outputs:
            None
    '''
    def __init__(self,hp):
        super(ASRNN,self).__init__()
        
        self.hp=hp
        
        self.conv_sub_sampling=ConvSubSampling(hp,hp.pretraining_insize,hp.conv_sub_2_nodes)
        self.standard_linear=nn.Linear(self.conv_sub_sampling.out_size,hp.standard_linear_nodes*hp.max_len)
        self.conformers=Conformer(hp.standard_linear_nodes,hp.mhsa_num_heads,hp.conformer_ff1_linear1_nodes,hp.n_conformers,hp.conformer_depthwise_conv_kernel,hp.dropout)
        self.projection_fc=nn.Linear(hp.standard_linear_nodes,hp.projection_out_size)
        self.swish=nn.SiLU()
        self.projection_batch_norm=nn.BatchNorm1d(hp.projection_out_size)
        self.projection_fc_1=nn.Linear(hp.projection_out_size,hp.projection_out_size)
        self.swish=nn.SiLU()
        self.projection_batch_norm=nn.BatchNorm1d(hp.projection_out_size)
        decoder_nodes=hp.standard_decoder_nodes//2 if hp.standard_decoder_bidirectional else hp.standard_decoder_nodes
        decoder_dropout=hp.dropout if hp.standard_decoder_layers>1 else 0.0
        self.lstm=nn.LSTM(hp.projection_out_size,hp.standard_decoder_nodes,bidirectional=hp.standard_decoder_bidirectional,num_layers=hp.standard_decoder_layers,dropout=decoder_dropout)
        self.dropout=nn.Dropout(hp.dropout)
        decoder_out_size=hp.standard_decoder_nodes*2 if hp.standard_decoder_bidirectional else hp.standard_decoder_nodes
        self.final_fc=nn.Linear(decoder_out_size,hp.ntokens)
        self.beta=torch.rand(1).to(hp.device)
        self.beta.requires_grad=True
        self.log_softmax=nn.LogSoftmax(-1)
        
        
    def predict(self,x):
        '''
            Retrevies the most probable labels
            
            Inputs:
                x: tensor; input probabilities
            Outputs:
                x: tensor; predicted labels
        '''
        x=torch.argmax(x,-1)
        return x
    
    def conformer_blocks(self,x):
        '''
            Applies n conformer blocks. This is embedded in the Conformer from torchaudio. This can be used when having individual layers, for example if the conformer layer is written from scratch.
            
            Inputs:
                x: nn.ModuleList; list of conformer layers
            Outputs:
                x: tensor; output tensor after applying this module
        '''
        for c in self.conformers:
            x=c(x)
        return x
        
    def projection_block(self,x,finetuning=False):
        '''
            Projection block module
            
            Inputs:
                x: tensor; input tensor
                finetuning: bool; whether the model is called during training in finetuning phase
            Outputs:
                x: tensor; output tensor after applying this module
        '''
        if finetuning:
            x=self.projection_fc_1(x)
        else:
            x=self.projection_fc(x)
        x=self.swish(x)
        x=self.projection_batch_norm(x)
        return x
        
    def time_warping(self,x,tau):
        '''
            The time warping module
            
            Inputs:
                x: tensor; the input tensor
                tau; tensor; the input lengths
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        Wt=[]
        for u in range(x.shape[0]):
            Wt.append([])
            w=random.randint(-self.hp.warping_param_W,self.hp.warping_param_W)
            if tau[u]<2*self.hp.warping_param_W:
                w0=self.hp.warping_param_W
            else:
                w0=random.randint(self.hp.warping_param_W,tau[u]-self.hp.warping_param_W-1)
            for t in range(int(tau[u])):
                if t<=w0:
                    wt=int(((w0+w)/w0)*t)
                else:
                    wt=torch.div(((tau[u]-1-w0-w)*t+(tau[u]-1)*w),(tau[u]-1-w0), rounding_mode='floor').item()
                Wt[u].append(wt)
            Wt[u]+=list(range(len(Wt[u]),x.shape[2]))
        x_warp=[]
        x=x.cpu().numpy()
        Wt=torch.LongTensor(np.array(Wt))
        for b in range(x.shape[0]):
            x_warp.append([])
            for f in range(x.shape[1]):
                xw=x[b][f][Wt[b]]
                x_warp[b].append(xw)
        x_warp=torch.FloatTensor(np.array(x_warp)).to(self.hp.device)
        return x_warp
        
    def frequency_masking(self,x):
        '''
            The frequency masking module
            
            Inputs:
                x: tensor; the input tensor
                tau; tensor; the input lengths
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        v=x.shape[1]
        f=random.randint(0,self.hp.frequency_mask_param_F)
        f0=random.randint(0,v-self.hp.frequency_mask_param_F)
        mask=[[[False]*x.shape[-1]]*v]*x.shape[0]
        mask[:][f0:f0+f]=[[True]*x.shape[-1]]*f
        mask=torch.BoolTensor(mask).to(self.hp.device)
        x=torch.masked_fill(x, mask, value=self.hp.mask_value)
        return x
        
    def time_masking(self,x,tau):
        '''
            The time masking module 
            
            Inputs:
                x: tensor; the input tensor
                tau; tensor; the input lengths
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        mask=[]
        for u in range(x.shape[0]):
            if u<len(tau) and self.hp.adaptive_size:
                T=floor(self.hp.ps*tau[u])
            else:
                T=self.hp.time_mask_param_T
            t=random.randint(0,T)
            t0=random.randint(0,max(tau[u]-T,tau[u]))
            mask.append([[False]*x.shape[2]]*x.shape[1])
            mask[u][:][t0:t0+t]=[True]*t
        mask=torch.BoolTensor(mask).to(self.hp.device)
        x=torch.masked_fill(x, mask, value=self.hp.mask_value)
        return x
    
    def SpecAugment(self,x,tau):
        '''
            The SpecAugment module
            
            Inputs:
                x: tensor; the input tensor
                tau; tensor; the input lengths
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        x=x.squeeze()
        if len(list(x.shape))<3:
            x=x.unsqueeze(0)
        for _ in range(self.hp.warping_ntimes):
            x=self.time_warping(x,tau)
        for _ in range(self.hp.frequency_mask_ntimes):
            x=self.frequency_masking(x)
        Mt=self.hp.time_multiplicity
        if self.hp.adaptive_multiplicity:
            Mt=min(Mt,floor(self.hp.pm))
        for _ in range(Mt):
            x=self.time_masking(x,tau)
        return x.unsqueeze(1)
    def encoder(self,x,input_lens,SpecAugment=False,finetuning=False):
        '''
            The model's encoder
            
            Inputs:
                x: tensor; input tensor
                SpecAugment: bool; whether SpecAugment must be performed. Default False
                finetuning: bool; whether fine-tuning is being performed. Default False
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        if SpecAugment:
            x=self.SpecAugment(x,input_lens)
        x=self.conv_sub_sampling(x)
        x=x.flatten(1)
        x=self.standard_linear(x)
        x=x.view((self.hp.batch_size,self.hp.max_len,x.shape[1]//self.hp.max_len))
        x=self.dropout(x)
        input_lens_nonzero=input_lens.gt(0)
        input_lens=torch.masked_select(input_lens,input_lens_nonzero)
        x=x[:input_lens.shape[0],:int(torch.max(input_lens).item()),:]
        x,output_lens=self.conformers(x,input_lens)
        x=nn.functional.pad(x,(0,0,0,self.hp.max_len-x.shape[1],0,self.hp.batch_size-input_lens.shape[0]))
        x=x.flatten(0,1)
        x=self.dropout(x)
        x=self.projection_block(x)
        if finetuning:
            x=self.projection_block(x,True)
        return x,output_lens
         
    def decoder(self,y):
        '''
            The model's decoder
            
            Inputs:
                y: tensor; input tensor
            Outputs:
                y: tensor; output tensor, the input tensor after the application of the module layers
        '''
        y=self.lstm(y)[0]
        y=self.decoder_fc(y)
        y=self.relu(y)
        y=y.view((self.hp.max_len*self.hp.batch_size,self.hp.ntokens))
        return y
    def forward(self,x,input_lens,SpecAugment=False,lm=None,finetuning=False):
        '''
            The application of the model on the input tensor
            
            Inputs:
                x: tensor; input tensor
                input_lens; tensor; the lengths of the input mels
                SpecAugment: bool; whether SpecAugment must be performed. Default False
                lm: language model. Default None
                finetuning: bool; whether fine-tuning is being performed. Default False
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        x=x.unsqueeze(1)
        x,output_lens=self.encoder(x,input_lens,SpecAugment,finetuning=finetuning)
        x=self.lstm(x)[0]
        x=self.dropout(x)
        x=self.final_fc(x)
        x=x.view((self.hp.batch_size,self.hp.max_len,self.hp.ntokens))
        x=self.log_softmax(x)
        if lm is not None:
            x=x+lm(self.hp.ngram,torch.argmax(x,-1))
        return x, output_lens
        
    