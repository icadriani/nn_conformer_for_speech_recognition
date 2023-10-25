import torch
import torch.nn as nn
# from lib.conformer import Conformer
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
        
        # preconvnet_fc_in_size=hp.n_mels*hp.max_len
        # self.covnet_rows=hp.n_mels
        # self.covnet_cols=hp.max_len//3
        # preconvnet_fc_out_size=3*self.covnet_rows*self.covnet_cols
        # print(preconvnet_fc_insize_in_size,preconvnet_fc_insize_out_size)
        # self.preconvnet_fc=nn.Linear(preconvnet_fc_in_size,preconvnet_fc_out_size)
        # self.convnet=convnext_tiny(weights=True)
        self.conv_sub_sampling=ConvSubSampling(hp,hp.pretraining_insize,hp.conv_sub_2_nodes)
        #self.conformer=Conformer(hp,hp.pretraining_linear_nodes,hp.pretraining_out_size)
        # self.standard_linear=nn.Linear(1000,hp.standard_linear_nodes)
        # self.standard_linear=nn.Linear(1000,hp.standard_linear_nodes*hp.max_len)
        self.standard_linear=nn.Linear(self.conv_sub_sampling.out_size,hp.standard_linear_nodes*hp.max_len)
        #print(self.conv_sub_sampling.out_size)
        #self.linear_quantization=nn.Linear(self.conv_sub_sampling.out_size,hp.target_context_vectors_size)
        # self.conformers=nn.ModuleList([Conformer(hp,hp.standard_linear_nodes)]*hp.n_conformers)
        self.conformers=Conformer(hp.standard_linear_nodes,hp.mhsa_num_heads,hp.conformer_ff1_linear1_nodes,hp.n_conformers,hp.conformer_depthwise_conv_kernel,hp.dropout)
        # configuration = Wav2Vec2ConformerConfig()
        # self.conformers=Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
        # self.conformers=Conformer(hp.standard_linear_nodes,num_layers=hp.n_conformers,dropout=hp.dropout)
        #self.mask=None
        self.projection_fc=nn.Linear(hp.standard_linear_nodes,hp.projection_out_size)
        self.swish=nn.SiLU()
        # self.projection_batch_norm=nn.BatchNorm1d(hp.projection_out_size//hp.ntokens)
        # if hp.batch_size==1:
        # hp.projection_out_size=hp.projection_out_size//hp.max_len
        self.projection_batch_norm=nn.BatchNorm1d(hp.projection_out_size)
        self.projection_fc_1=nn.Linear(hp.projection_out_size,hp.projection_out_size)
        self.swish=nn.SiLU()
        # self.projection_batch_norm=nn.BatchNorm1d(hp.projection_out_size//hp.ntokens)
        # if hp.batch_size==1:
        # hp.projection_out_size=hp.projection_out_size//hp.max_len
        self.projection_batch_norm=nn.BatchNorm1d(hp.projection_out_size)
        decoder_nodes=hp.standard_decoder_nodes//2 if hp.standard_decoder_bidirectional else hp.standard_decoder_nodes
        decoder_dropout=hp.dropout if hp.standard_decoder_layers>1 else 0.0
        #decoder_nodes=hp.target_context_vectors_size//2 if hp.pretraining_decoder_bidirectional else hp.target_context_vectors_size
        # if pretrained_embeddings is None:
        #     self.yembedding=nn.Embedding(hp.ntokens,hp.embedding_dim)
        # else:
        #     self.yembedding=nn.Embedding.from_pretrained(pretrained_embeddings)
        #     hp.embedding_dim=pretrained_embeddings.shape[-1]
        self.lstm=nn.LSTM(hp.projection_out_size,hp.standard_decoder_nodes,bidirectional=hp.standard_decoder_bidirectional,num_layers=hp.standard_decoder_layers,dropout=decoder_dropout)
        self.dropout=nn.Dropout(hp.dropout)
        decoder_out_size=hp.standard_decoder_nodes*2 if hp.standard_decoder_bidirectional else hp.standard_decoder_nodes
        # self.layer_norm=nn.LayerNorm(hp.ntokens)
        # decoder_out_size=decoder_nodes*2 if hp.standard_decoder_bidirectional else decoder_nodes
        # decoder_out_size*=hp.ntokens
        # decoder_out_size*=hp.max_len
        # self.decoder_batch_norm=nn.BatchNorm1d(hp.embedding_dim)
        # decoder_out_size*=hp.max_len
        # self.decoder_fc=nn.Linear(decoder_out_size,hp.ntokens)
        # final_fc_in_size=decoder_out_size*hp.ntokens+hp.projection_out_size
        # print(final_fc_in_size)
        # if hp.batch_size>1:
        #     final_fc_in_size//=hp.max_len
        # print(final_fc_in_size)
        self.final_fc=nn.Linear(decoder_out_size,hp.ntokens)
        # self.final_fc=nn.Linear(hp.projection_out_size,hp.ntokens)
        # self.final_fc2=nn.Linear(1,hp.ntokens)
        # self.lm_fc=nn.Linear(hp.ntokens*2,hp.ntokens)
        # self.tanh=nn.Tanh()
        self.beta=torch.rand(1).to(hp.device)
        self.beta.requires_grad=True
        # self.relu=nn.ReLU()
        self.log_softmax=nn.LogSoftmax(-1)
        
        # print(type(hp.max_value))
        # self.embeddings=nn.Embedding(hp.max_value+1,hp.embedding_dim)
        # self.pos_embeddings=nn.Embedding(hp.n_mels+1,hp.embedding_dim)
        
    def predict(self,x):
        '''
            Retrevies the most probable labels
            
            Inputs:
                x: tensor; input probabilities
            Outputs:
                x: tensor; predicted labels
        '''
        # x=self.softmax(x)
        # print(x.shape)
        # print(x)
        #     mask
        # print(torch.max(x,-1))
        x=torch.argmax(x,-1)
        # print(x.shape)
        # x=x.numpy().tolist()
        return x
    
    def conformer_blocks(self,x):
        '''
            Applies n conformer blocks. This is embedded in the Conformer from torchaudio. This can be used when having individual layers, for example if the conformer layer is written from scratch.
            
            Inputs:
                x: nn.ModuleList; list of conformer layers
            Outputs:
                x: tensor; output tensor after applying this module
        '''
        #x=self.conformer(x)
        for c in self.conformers:
            # print(x.shape)
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
        # print(x.shape)
        # x=torch.flatten(x,1)
        # x=torch.flatten(x,1)
        # print(x.shape)
        if finetuning:
            x=self.projection_fc_1(x)
        else:
            x=self.projection_fc(x)
        # y=self.predict(x)
        # x=x.flatten(0,1)
        # print(x.shape)
        # print(x.shape)
        # print(x.shape)
        # if self.hp.batch_size==1:
        # else:
        #     x=x.view((x.shape[0],x.shape[1]//self.hp.ntoken,*self.hp.ntokens))
        x=self.swish(x)
        # print(x.shape)
        # x=x.view((self.hp.batch_size*self.hp.max_len,x.shape[1]//self.hp.max_len))
        # print(x.shape)
        # x=x.unsqueeze(-1)
        # print(x.shape)
        x=self.projection_batch_norm(x)
        # x=x.view((self.hp.max_len*self.hp.batch_size,self.hp.ntokens))
        # x=x.squeeze().unsqueeze(0)
        # if self.hp.batch_size>1:
        #     x=x.view((x.shape[0]*x.shape[1]//self.hp.max_len,self.hp.max_len))
        # x=x.unsqueeze(1)
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
        # print(tau)
        # tau=x.shape[-1]
        Wt=[]
        for u in range(x.shape[0]):
            Wt.append([])
            w=random.randint(-self.hp.warping_param_W,self.hp.warping_param_W)
            if tau[u]<2*self.hp.warping_param_W:
                w0=self.hp.warping_param_W
            else:
                w0=random.randint(self.hp.warping_param_W,tau[u]-self.hp.warping_param_W-1)
            # print(tau[u])
            for t in range(int(tau[u])):
                if t<=w0:
                    wt=int(((w0+w)/w0)*t)
                else:
                    wt=torch.div(((tau[u]-1-w0-w)*t+(tau[u]-1)*w),(tau[u]-1-w0), rounding_mode='floor').item()
                # wt=round(wt)
                # print(wt,type(wt))
                Wt[u].append(wt)
                # print(tau[u])
            # print()
            # print(Wt[u])
            Wt[u]+=list(range(len(Wt[u]),x.shape[2]))
            # print(len(Wt[u]),x.shape[2],tau[u])
        x_warp=[]
        # print(Wt)
        x=x.cpu().numpy()#.tolist()
        Wt=torch.LongTensor(np.array(Wt))#.to(self.hp.device)
        for b in range(x.shape[0]):
            x_warp.append([])
            for f in range(x.shape[1]):
                # print(Wt[b])
                xw=x[b][f][Wt[b]]
                x_warp[b].append(xw)
        # print(x_warp)
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
        # f=round()
        mask=[[[False]*x.shape[-1]]*v]*x.shape[0]
        mask[:][f0:f0+f]=[[True]*x.shape[-1]]*f
        # print(mask[0])
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
        # tau=x.shape[2]
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
        # print(x.shape)
        # tau=x['tau']
        if SpecAugment:
            x=self.SpecAugment(x,input_lens)
        # elif type(x)==dict:
        #     x=x['mels']
        # x=x.squeeze()
        # print(x.shape)
        x=self.conv_sub_sampling(x)
        # x=x.flatten(1)
        # x=self.preconvnet_fc(x)
        # print(x)
        # x=x.view(-1,3,self.covnet_rows,self.covnet_cols)
        # print(x.shape)
        # x=self.convnet(x)
        x=x.flatten(1)
        # print(x)
        # print(x.shape)
        x=self.standard_linear(x)
        x=x.view((self.hp.batch_size,self.hp.max_len,x.shape[1]//self.hp.max_len))
        x=self.dropout(x)
        # x=x.unsqueeze(1)
        # print(x)
        # print(x.shape)
        # x=x.squeeze()
        # lengths=torch.LongTensor([x.shape[1]]*x.shape[0]).to(self.hp.device)
        # print(input_lens.shape)
        # print(x.shape)
        # shape=x.shape
        input_lens_nonzero=input_lens.gt(0)
        input_lens=torch.masked_select(input_lens,input_lens_nonzero)
        x=x[:input_lens.shape[0],:int(torch.max(input_lens).item()),:]
        # x=x.flatten(1)
        x,output_lens=self.conformers(x,input_lens)
        # x=x.logits
        # if input_lens[-1].item()==0:
        #     print(x)
        # print(x.shape)
        # print(self.hp.batch_size,input_lens.shape[0],self.hp.batch_size-input_lens.shape[0])
        x=nn.functional.pad(x,(0,0,0,self.hp.max_len-x.shape[1],0,self.hp.batch_size-input_lens.shape[0]))
        # print(x.shape)
        # print(x[0])
        x=x.flatten(0,1)
        x=self.dropout(x)
        # print(x.shape)
        # x=self.conformer_blocks(x)
        # print(x.shape)
        # if self.hp.use_standard_projection or finetuning:
        x=self.projection_block(x)
        if finetuning:
            x=self.projection_block(x,True)
        return x,output_lens
    # def merge_embeddings(self,emb,mask):
        # emb=emb.cpu()
        # emb=emb.numpy().tolist()
        # merged=[]
        # for s in range(mask.shape[0]):
        #     merged.append([])
        #     for x in range(mask.shape[1]):
        #         if mask[s][x]:
         
    def decoder(self,y):
        '''
            The model's decoder
            
            Inputs:
                y: tensor; input tensor
            Outputs:
                y: tensor; output tensor, the input tensor after the application of the module layers
        '''
        # print(y.shape,y[0].shape)
        # y=self.predict(y)
        # shape=[1]+list(y[0].shape)
        # y=torch.cat([torch.zeros(shape,dtype=torch.long).to(self.hp.device),y[:-1]])
        # # print(y.shape)
        # mask=y.eq(self.hp.space_idx)
        # y=y.masked_fill_(mask=mask,value=0)
        # print(y.shape)
        # y=self.yembedding(y)
        # print(y.shape)
        # y=y.flatten(1)
        # print(y.shape)
        # print(y.shape)
        # y=y.flatten(0,1)
        # y=y.unsqueeze(1)
        # print(y.shape)
        # y=self.decoder_batch_norm(y)
        y=self.lstm(y)[0]
        # print(y.shape)
        # y=y.flatten(0,1)
        # print(y.shape)
        y=self.decoder_fc(y)
        y=self.relu(y)
        # y=y.flatten()
        y=y.view((self.hp.max_len*self.hp.batch_size,self.hp.ntokens))
        return y
    def forward(self,x,input_lens,SpecAugment=False,lm=None,finetuning=False):#,OutAugment=0.0):
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
        # x=x.squeeze()
        x=x.unsqueeze(1)
        # print(torch.max(x.round().int()))
        # y=self.embeddings((x*self.hp.max_value).round().int())
        # # print(x.shape)
        # x=x.unsqueeze(-1).repeat(1,1,1,self.hp.embedding_dim).squeeze()
        # # # print(x.shape)
        # x=x+y
        # x=torch.mean(x,-2)
        # pos=torch.arange(1,self.hp.n_mels+1).to(self.hp.device)#.unsqueeze(0)
        # pos=self.pos_embeddings(pos)
        # # print(pos.shape)
        # # pos=pos.unsqueeze(-1)
        # pos=pos.repeat(self.hp.batch_size,1,self.hp.max_len,1)#.unsqueeze(1)
        # # print(pos[0])
        # # print(x.shape,pos.shape)
        # # print(pos.shape)
        # x=x+pos
        # x=self.pos_embeddings(x)
        # print(x['tau'].shape)
        # tau=torch.max(x['tau'],dim=-1)[0]
        # print(tau.shape)
        x,output_lens=self.encoder(x,input_lens,SpecAugment,finetuning=finetuning)
        # print(x.shape)
        # print(x)
        # y=self.decoder(x)
        # print(x.shape,y.shape)
        # x=torch.cat([x,y],dim=-1)
        # x=x+y
        x=self.lstm(x)[0]
        # x=self.decoder_norm(x)
        # x=x.view((x.shape[0]*x.shape[1]//self.hp.ntokens,self.hp.ntokens))
        # print(x.shape)
        # x=x.view(self.hp.batch_size*self.hp.max_len,x.shape[1]//self.hp.max_len)
        # x=x.flatten(0,1)
        x=self.dropout(x)
        x=self.final_fc(x)
        # x=self.tanh(x)
        # x=x.flatten().unsqueeze(1)
        # x=self.final_fc2(x)
        # print(x.shape)
        x=x.view((self.hp.batch_size,self.hp.max_len,self.hp.ntokens))
        x=self.log_softmax(x)
        if lm is not None:
            # x=torch.cat([x,lm],dim=-1)
            # print(torch.max(torch.max(torch.max(x,-1)[0],-1)[0],-1)[0].item())
            # print(torch.max(torch.max(torch.max(lm,-1)[0],-1)[0],-1)[0].item())
            # print('-------------------')
            # print(x.shape)
            # print(x)
            x=x+lm(self.hp.ngram,torch.argmax(x,-1))
            # x=self.layer_norm(x)
            # x=self.log_softmax(x)
            # x=self.lm_fc(x)
            # x=self.log_softmax(x)
        # x=nn.functional.log_softmax(x,dim=-1)
        # # print(x.shape)
        # print(torch.max(torch.max(torch.max(x,-1)[0],-1)[0],-1)[0].item())
        # print(tuple(x.shape[:-1]))
        # if OutAugment>0:
        #     mask_idx=torch.randint(self.hp.space_idx,x.shape[-1],x.shape[:-1]).to(self.hp.device)
        #     mask_prob=torch.rand(x.shape[:-1]).to(self.hp.device)
        #     mask=mask_prob.ge(0.5*OutAugment)
        #     mask_prob.requires_grad=True
        #     mask_idx=mask_idx.masked_fill_(mask,value=self.hp.space_idx)
        #     mask_prob=torch.rand(x.shape[:-1]).to(self.hp.device)
        #     mask_prob.requires_grad=True
        #     # print(mask_prob)
        #     # print(mask_prob)
        #     mask=mask_prob.le(OutAugment)
        #     mask_prob=mask_prob.clone().masked_fill_(mask,value=100)
        #     mask=mask_prob.le(1)
        #     mask_prob=mask_prob.clone().masked_fill_(mask,value=0)
        #     # print(mask_prob)
        #     mask_prob=mask_prob.unsqueeze(-1).expand(x.shape)
        #     mask_idx=nn.functional.one_hot(mask_idx,x.shape[-1])
        #     # print(mask_idx)
        #     # print(mask_prob)
        #     # print(mask_idx.shape,mask_prob.shape)
        #     mask_=mask_prob*mask_idx
        #     # mask=mask_prob.eq(0)
        #     # mask_=mask_.masked_fill_(mask,value=1)
        #     x=x+mask_
        # print(mask[0][0].cpu().numpy().tolist())
        # print(mask.shape)
        # x=x.transpose(0,1)
        return x, output_lens
        
    