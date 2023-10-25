import torch
import torch.nn as nn
# from lib.conformer import Conformer
from torchaudio.models import Conformer
from lib.convsubsampling import ConvSubSampling
from torchvision.models import convnext_tiny

class PreTrainingNN(nn.Module):
    def __init__(self,hp):
        super(PreTrainingNN,self).__init__()
        
        self.hp=hp
        
        # self.conv_sub_sampling=ConvSubSampling(hp,hp.pretraining_insize,hp.encoded_features_out_size)
        self.covnet_rows=hp.n_mels
        self.covnet_cols=hp.max_len//3
        # preconvnet_fc_out_size=3*self.covnet_rows*self.covnet_cols
        # print(preconvnet_fc_insize_in_size,preconvnet_fc_insize_out_size)
        # self.preconvnet_fc=nn.Linear(preconvnet_fc_in_size,preconvnet_fc_out_size)
        self.convnet=convnext_tiny(weights=True)
        #self.conformer=Conformer(hp,hp.pretraining_linear_nodes,hp.pretraining_out_size)
        self.fc=nn.Linear(1000,hp.standard_linear_nodes*hp.max_len)
        #print(self.conv_sub_sampling.out_size)
        self.linear_quantization=nn.Linear(1000,hp.target_context_vectors_size*hp.max_len)
        # self.conformers=nn.ModuleList([Conformer(hp,hp.pretraining_linear_nodes)]+[Conformer(hp,hp.pretraining_linear_nodes) for _ in range(hp.n_conformers-1)])
        self.conformers=Conformer(hp.standard_linear_nodes,hp.mhsa_num_heads,hp.conformer_ff1_linear1_nodes,hp.n_conformers,hp.conformer_depthwise_conv_kernel,hp.dropout)
        decoder_dropout=hp.dropout if hp.pretraining_decoder_layers>1 else 0.0
        # decoder_nodes=hp.target_context_vectors_size//2 if hp.pretraining_decoder_bidirectional else hp.target_context_vectors_size
        self.decoder=nn.LSTM(hp.standard_linear_nodes,hp.decoder_nodes,bidirectional=hp.pretraining_decoder_bidirectional,num_layers=hp.pretraining_decoder_layers,dropout=decoder_dropout)
        #decoder_out_size=hp.pretraining_out_size*2 if hp.pretraining_decoder_bidirectional else hp.pretraining_out_size
        self.mask=None
    
    def conformer_blocks(self,x):
        #x=self.conformer(x)
        for c in self.conformers:
            x=c(x)
        return x
    
    def masking(self,x,change_mask=False):
        # if change_mask or self.mask is None:
        self.mask=x.ge(self.hp.mask_probability)
        x=torch.masked_fill(x, self.mask, value=self.hp.mask_value)
        #print(x.shape)
        #x=x.view((self.hp.batch_size,1,x.shape[0]//self.hp.batch_size))
        return x
    
    def quantization(self,x):
        # print(x.shape)
        x=self.linear_quantization(x)
        x=x.view((self.hp.batch_size,self.hp.max_len,x.shape[1]//self.hp.max_len))
        # print(x.shape)
        if not self.hp.simplified_pretraining:
            x=nn.functional.gumbel_softmax(x, tau=self.hp.temperature_tau)
        return x

    def forward(self,x,input_lens,change_mask=False):
        # encoded_features=self.conv_sub_sampling(x)
        encoded_features=x.view(-1,3,self.covnet_rows,self.covnet_cols)
        # print(x.shape)
        encoded_features=self.convnet(encoded_features)
        encoded_features=encoded_features.flatten(1)
        #print(encoded_features.shape)
        target_context_vectors=self.quantization(encoded_features)
        x=self.masking(encoded_features,change_mask)
        #print(x.shape)
        x=self.fc(x)
        x=x.view((self.hp.batch_size,self.hp.max_len,x.shape[1]//self.hp.max_len))
        # x=self.conformer_blocks(x)
        input_lens_nonzero=input_lens.gt(0)
        input_lens=torch.masked_select(input_lens,input_lens_nonzero)
        x=x[:input_lens.shape[0],:int(torch.max(input_lens).item()),:]
        # print(x.shape)
        # print(x)
        x,output_lengths=self.conformers(x,input_lens)
        # print(x.shape)
        x=nn.functional.pad(x,(0,0,0,self.hp.max_len-x.shape[1],0,self.hp.batch_size-input_lens.shape[0]))
        # x=nn.functional.pad(x,(0,0,0,self.hp.max_len-x.shape[1]))
        # print(x)
        # print('helloooo')
        context_vectors_from_masked_features=self.decoder(x)[0]
        # print(context_vectors_from_masked_features.shape)
        return context_vectors_from_masked_features,target_context_vectors
        
        
    