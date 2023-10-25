import torch
import torch.nn as nn
from lib.relativepositionalembeddings import RelativePositionalEmbeddings
from math import sqrt

class LM(nn.Module):
    '''
        class LM
        
        This is the language model
        
        Inputs:
            hp: HParams; the class of hyperparameters
            input_vocab_size: int; the size of the input vocabulary
            output_vocab_size: int; the size of the output vocabulary
        Outputs:
            None
    '''
    def __init__(self,hp,input_vocab_size,output_vocab_size):
        super(LM,self).__init__()
        self.hp=hp
        self.input_embeddings=nn.Embedding(input_vocab_size,hp.input_embedding_size)
        self.input_rel_pos_emb=RelativePositionalEmbeddings((hp.batch_size*hp.lm_max_len,input_vocab_size,hp.input_embedding_size),hp.rel_pos_emb)()
        # self.input_rel_pos_emb=RelativePositionalEmbeddings((hp.batch_size*hp.lm_max_len,output_vocab_size,hp.output_embedding_size),hp.rel_pos_emb)()
        self.output_embeddings=nn.Embedding(output_vocab_size,hp.output_embedding_size)
        self.output_rel_pos_emb=RelativePositionalEmbeddings((hp.batch_size*hp.lm_max_len,output_vocab_size,hp.output_embedding_size),hp.rel_pos_emb)()
        self.input_mhas=nn.ModuleList([nn.MultiheadAttention(hp.input_embedding_size,hp.lm_in_mhsa_num_heads,batch_first=True)]*hp.lm_in_N)
        self.output_mhas=nn.ModuleList([nn.MultiheadAttention(hp.output_embedding_size,hp.lm_out_mhsa_num_heads,batch_first=True)]*hp.lm_out_N)
        self.masked_output_mhas=nn.ModuleList([nn.MultiheadAttention(hp.output_embedding_size,hp.lm_masked_out_mhsa_num_heads,batch_first=True)]*hp.lm_out_N)
        self.input_mhas_layernorm=nn.ModuleList([nn.LayerNorm(hp.input_embedding_size)]*hp.lm_in_N)
        self.output_mhas_layernorm=nn.ModuleList([nn.LayerNorm(hp.output_embedding_size)]*hp.lm_out_N)
        self.masked_output_mhas_layernorm=nn.ModuleList([nn.LayerNorm(hp.output_embedding_size)]*hp.lm_out_N)
        self.inner_input_fcs=nn.ModuleList([nn.Linear(hp.input_embedding_size,hp.lm_innner_input_nodes)]*hp.lm_in_N)
        self.inner_output_fcs=nn.ModuleList([nn.Linear(hp.output_embedding_size,hp.lm_innner_output_nodes)]*hp.lm_out_N)
        self.input_fcs=nn.ModuleList([nn.Linear(hp.lm_innner_input_nodes,hp.input_embedding_size)]*hp.lm_in_N)
        self.output_fcs=nn.ModuleList([nn.Linear(hp.lm_innner_output_nodes,hp.output_embedding_size)]*hp.lm_out_N)
        self.input_fcs_layernorm=nn.ModuleList([nn.LayerNorm(hp.input_embedding_size)]*hp.lm_in_N)
        self.output_fcs_layernorm=nn.ModuleList([nn.LayerNorm(hp.output_embedding_size)]*hp.lm_out_N)
        # self.linear=nn.Linear(hp.output_embedding_size*output_vocab_size*hp.lm_max_len,output_vocab_size)
        self.linear=nn.Linear(hp.output_embedding_size*output_vocab_size,output_vocab_size)
        # print(hp.output_embedding_size*output_vocab_size*hp.batch_size*hp.lm_max_len)
        # print(hp.output_embedding_size,output_vocab_size,hp.batch_size*hp.lm_max_len)
        self.relu=nn.ReLU()
        self.mask=self.get_mask(output_vocab_size)#,hp.output_embedding_size)
    def get_mask(self,shape_0):
        '''
            Retrives the mask given a length
            
            Inputs:
                shape_0: int; the length of the mask
            Outputs:
                mask: tensor; the computed mask
        '''
        mask=[]
        for i in range(shape_0):
            mask_i=[True]*min((i+1),shape_0)
            mask_i=mask_i+[True]*(shape_0-len(mask_i))
            mask.append(mask_i)
        mask=torch.Tensor(mask).to(self.hp.device)
        return mask
    def encoder(self,x):
        '''
            The model's encoded runned n times
            
            Inputs:
                x: tensor; input tensor
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        for i in range(self.hp.lm_in_N):
            # print(x.shape)
            x=x+self.input_mhas[i](x,x,x)[0]
            x=self.input_mhas_layernorm[i](x)
            ffn=self.inner_input_fcs[i](x)
            ffn=self.relu(ffn)
            x=x+self.input_fcs[i](ffn)
            x=self.input_fcs_layernorm[i](x)
        return x
    def decoder(self,x,enc):
        '''
            The model's decoder runned n times
            
            Inputs:
                x: tensor; input tensor
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        for i in range(self.hp.lm_out_N):
            ## add mask
            # print(x.shape)
            x=x+self.masked_output_mhas[i](x,x,x,attn_mask=self.mask)[0]
            x=self.masked_output_mhas_layernorm[i](x)
            x=x+self.output_mhas[i](x,enc,enc)[0]
            x=self.output_mhas_layernorm[i](x)
            ffn=self.inner_output_fcs[i](x)
            ffn=self.relu(ffn)
            x=x+self.output_fcs[i](ffn)
            x=self.output_fcs_layernorm[i](x)
        return x
    def forward(self,inbatch,outbatch):
        '''
            The application of the language model on the input tensor
            
            Inputs:
                x: tensor; input tensor
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        # self.input_embeddings.weight*=sqrt(self.hp.input_embedding_size)
        # print(inbatch.shape)
        inbatch=self.input_embeddings(inbatch)
        # print(inbatch.shape)
        inbatch=inbatch-self.input_rel_pos_emb
        # inbatch=torch.flatten(inbatch,0,1)
        # inbatch=inbatch.unsqueeze(1)
        enc=self.encoder(inbatch)
        # self.output_embeddings.weight*=sqrt(self.hp.output_embedding_size)
        outbatch=self.output_embeddings(outbatch)
        outbatch=outbatch+self.output_rel_pos_emb
        inbatch=outbatch.unsqueeze(1)
        # print(outbatch.shape)
        # outbatch=torch.flatten(outbatch,0,1)
        x=self.decoder(outbatch,enc)
        # print(x.shape)
        # x=torch.flatten(x)
        # print(x.shape)
        # x=x.view(self.hp.batch_size,1,x.shape[0]//self.hp.batch_size)
        x=x.flatten(1).unsqueeze(1)
        x=self.linear(x)
        return x.squeeze()