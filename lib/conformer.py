import torch
import torch.nn as nn
from lib.relativepositionalembeddings import RelativePositionalEmbeddings

class Conformer(nn.Module):
    '''
        The conformer model has described in the paper
        
        Inputs:
            hp: HParams; the class of hyperparameters
            conformer_size=
    '''
    def __init__(self,hp,conformer_size):
        super(Conformer,self).__init__()
        
        self.hp=hp
        
        self.last_layernorm=nn.LayerNorm(conformer_size)
        
        # activations
        self.swish=nn.SiLU()
        self.glu=nn.GLU()
        
        self.dropout=nn.Dropout(hp.conformer_dropout)
        
        # multihead self attention
        self.conformer_size=conformer_size
        if hp.rel_att:
            self.rel_pos_emb=RelativePositionalEmbeddings((hp.batch_size*hp.max_len,conformer_size),hp.rel_pos_emb)().to(hp.device)
            # self.rel_pos_emb=self.rel_pos_emb()
            self.mhsa_layernorm=nn.LayerNorm(conformer_size)
            self.mhsa=nn.MultiheadAttention(conformer_size,hp.mhsa_num_heads,batch_first=True)
            # self.key=torch.randn((hp.batch_size,1,conformer_size),requires_grad=True).to(hp.device)
            # self.value=torch.randn((hp.batch_size,1,conformer_size),requires_grad=True).to(hp.device)
            #self.key=nn.Embedding(hp.n_mels,conformer_size)
            #self.value=nn.Embedding(hp.n_mels,conformer_size)
        
        # convolution module
        
        self.conv_layernorm=nn.LayerNorm(conformer_size)
        self.pointwise_conv1=nn.Conv1d(1,hp.conformer_pointwise_conv1_nodes,hp.conformer_pointwise_conv1_kernel,stride=1)
        self.depthwise_conv=nn.Conv1d(hp.conformer_pointwise_conv1_nodes,hp.conformer_depthwise_conv_nodes,hp.conformer_depthwise_conv_kernel,stride=hp.conformer_depthwise_conv_stride)
        self.conv_batch_norm=nn.BatchNorm1d(hp.conformer_depthwise_conv_nodes)
        
        conv_kers=[hp.conformer_pointwise_conv1_kernel,hp.conformer_depthwise_conv_kernel,hp.conformer_pointwise_conv2_kernel]
        conv_strides=[1,hp.conformer_depthwise_conv_stride,1]
        #pool_kers=[hp.pool1_kernel,hp.pool2_kernel,hp.pool3_kernel,hp.pool4_kernel,hp.pool5_kernel]
        conv_last_size=conformer_size
        #print(conv_last_size)
        for i in range(len(conv_kers)):
            conv_last_size=(conv_last_size-conv_kers[i]+conv_strides[i])//(conv_strides[i])
            #print(i,conv_last_size)
            if i==0:
                conv_last_size//=2
        
        # print(conv_last_size)
        conformer_ff2_linear2_nodes=conformer_size//conv_last_size
        
        #conv_mod_out_size=conv_last_size*conformer_size
        self.pointwise_conv2=nn.Conv1d(hp.conformer_depthwise_conv_nodes,conformer_ff2_linear2_nodes,hp.conformer_pointwise_conv2_kernel,stride=1)
        
        # feed forward module
        
        self.ff1_layernorm=nn.LayerNorm(conformer_size)
        self.ff1_linear1=nn.Linear(conformer_size,hp.conformer_ff1_linear1_nodes)
        self.ff1_linear2=nn.Linear(hp.conformer_ff1_linear1_nodes,conformer_size)
        
        self.ff2_layernorm=nn.LayerNorm(conformer_size)
        self.ff2_linear1=nn.Linear(conformer_size,hp.conformer_ff2_linear1_nodes)
        self.ff2_linear2=nn.Linear(hp.conformer_ff2_linear1_nodes,conformer_size)
        
    def feed_forward_module_1(self,x):
        '''
            First Feed Forward Module
            
            Inputs:
                x: tensor; input tensor
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        x=self.ff1_layernorm(x)
        x=self.ff1_linear1(x)
        x=self.swish(x)
        x=self.dropout(x)
        x=self.ff1_linear2(x)
        x=self.dropout(x)
        return x
        
    def feed_forward_module_2(self,x):
        '''
            Second Feed Forward Module
            
            Inputs:
                x: tensor; input tensor
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        x=self.ff2_layernorm(x)
        x=self.ff2_linear1(x)
        x=self.swish(x)
        x=self.dropout(x)
        x=self.ff2_linear2(x)
        x=self.dropout(x)
        return x
        
    def multihead_self_attention_module(self,x):
        '''
            Multihead self Attention Module
            
            Inputs:
                x: tensor; input tensor
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        # relpos=self.positional_embeddings(x)
        x=x+self.rel_pos_emb
        # print(x.shape)
        x=self.mhsa_layernorm(x)
        # print(x.shape)
        x,_=self.mhsa(x,x,x)
        # print(x.shape)
        x=self.dropout(x)
        return x
        
    def convolution_module(self,x):
        '''
            Convolution Module
            
            Inputs:
                x: tensor; input tensor
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        # x=x.unsqueeze(2)
        #print()
        # print(x.shape)
        x=self.conv_layernorm(x)
        # print(x.shape)
        x=x.unsqueeze(1)
        x=self.pointwise_conv1(x)
        # print(x.shape)
        x=self.glu(x)
        #print('glu',x.shape)
        x=self.depthwise_conv(x)
        # print(x.shape)
        x=self.conv_batch_norm(x)
        # print(x.shape)
        x=self.swish(x)
        x=self.pointwise_conv2(x)
        x=self.dropout(x)
        # print(x.shape)
        x=torch.flatten(x,1)#.unsqueeze(1)
        # print(x.shape)
        # print('hi')
        return x
        
    def forward(self,x):
        '''
            The application of the model on the input tensor
            
            Inputs:
                x: tensor; input tensor
            Outputs:
                x: tensor; output tensor, the input tensor after the application of the module layers
        '''
        # print(x.shape)
        if len(list(x.shape))!=2:# or x.shape[1]!=self.conformer_size or x.shape[0]!=self.hp.batch_size:
            raise RuntimeError('Expected input size is of shape '+str([self.hp.batch_size,self.conformer_size]))
        x=x+self.feed_forward_module_1(x)/2
        # print(x.shape)
        if self.hp.rel_att:
            x=x+self.multihead_self_attention_module(x)
        x=x+self.convolution_module(x)
        # print(x.shape)
        x=x.flatten(1)
        x=x+self.feed_forward_module_2(x)/2
        x=self.last_layernorm(x)
        return x