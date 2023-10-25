import torch
import torch.nn as nn
from torchaudio.models import Conformer
from lib.convsubsampling import ConvSubSampling
from torchvision.models import convnext_tiny

class PreTrainingNN(nn.Module):
    '''
        class PreTrainingNN
        
        The pretraining model
        
        Inputs:
            hp: HParams; the class of hyperparameters
        Outputs:
            None
    '''
    def __init__(self,hp):
        super(PreTrainingNN,self).__init__()
        
        self.hp=hp
        
        self.conv_sub_sampling=ConvSubSampling(hp,hp.pretraining_insize,hp.encoded_features_out_size)
        self.fc=nn.Linear(self.conv_sub_sampling.out_size,hp.standard_linear_nodes*hp.max_len)
        self.linear_quantization=nn.Linear(self.conv_sub_sampling.out_size,hp.target_context_vectors_size*hp.max_len)
        self.conformers=Conformer(hp.standard_linear_nodes,hp.mhsa_num_heads,hp.conformer_ff1_linear1_nodes,hp.n_conformers,hp.conformer_depthwise_conv_kernel,hp.dropout)
        decoder_dropout=hp.dropout if hp.pretraining_decoder_layers>1 else 0.0
        self.decoder=nn.LSTM(hp.standard_linear_nodes,hp.decoder_nodes,bidirectional=hp.pretraining_decoder_bidirectional,num_layers=hp.pretraining_decoder_layers,dropout=decoder_dropout)
        self.mask=None
    
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
    
    def masking(self,x):
        '''
            Mask the input tensor
            
            Inputs:
                x: input tensor
            Outputs:
                x: output tensor  
        '''
        self.mask=x.ge(self.hp.mask_probability)
        x=torch.masked_fill(x, self.mask, value=self.hp.mask_value)
        return x
    
    def quantization(self,x):
        '''
            This is the quantization module which in the simplified case is just a linear layer
            
            Inputs:
                x: input tensor
            Outputs:
                x: output tensor 
        '''
        x=self.linear_quantization(x)
        x=x.view((self.hp.batch_size,self.hp.max_len,x.shape[1]//self.hp.max_len))
        if not self.hp.simplified_pretraining:
            x=nn.functional.gumbel_softmax(x, tau=self.hp.temperature_tau)
        return x

    def forward(self,x,input_lens):
        '''
            Runs the model on the input tensor
            
            Inputs:
                x: input tensor
                input_lens: the input lengths
            Outputs:
                context_vectors_from_masked_features: tensor; context vectors from masked features
                target_context_vectors: tensor; target context vectors
        '''
        encoded_features=self.conv_sub_sampling(x)
        encoded_features=encoded_features.flatten(1)
        target_context_vectors=self.quantization(encoded_features)
        x=self.masking(encoded_features)
        x=self.fc(x)
        x=x.view((self.hp.batch_size,self.hp.max_len,x.shape[1]//self.hp.max_len))
        input_lens_nonzero=input_lens.gt(0)
        input_lens=torch.masked_select(input_lens,input_lens_nonzero)
        x=x[:input_lens.shape[0],:int(torch.max(input_lens).item()),:]
        x,output_lengths=self.conformers(x,input_lens)
        x=nn.functional.pad(x,(0,0,0,self.hp.max_len-x.shape[1],0,self.hp.batch_size-input_lens.shape[0]))
        context_vectors_from_masked_features=self.decoder(x)[0]
        return context_vectors_from_masked_features,target_context_vectors
        
        
    