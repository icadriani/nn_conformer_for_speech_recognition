import torch
import torch.nn as nn
from lib.conformer import Conformer

class ConvSubSampling(nn.Module):
    '''
        class ConvSubSampling
        
        This the Convolution Subsampling module. 
        
        Inputs:
            hp: HParams; The class of Hyperparameters
            in_size: int; the size of the input I (*,I,*,*)
            out_nodes: The number of the output nodes (The nodes of the last convolution)
    '''
    def __init__(self,hp,in_size,out_nodes):
        super(ConvSubSampling,self).__init__()
        
        self.hp=hp
        
        self.conv_sub_1=nn.Conv2d(in_size,hp.conv_sub_1_nodes,hp.conv_sub_1_kernel,stride=hp.conv_sub_1_stride)
        self.conv_sub_2=nn.Conv2d(hp.conv_sub_1_nodes,out_nodes,hp.conv_sub_2_kernel,stride=hp.conv_sub_2_stride)
        
        conv_kers=[hp.conv_sub_1_kernel,hp.conv_sub_2_kernel]
        conv_strides=[hp.conv_sub_1_stride,hp.conv_sub_2_stride]
        convh=hp.input_rows
        convw=hp.input_cols
        for i in range(len(conv_kers)):
            convi_stride_h,convi_stride_w=list(conv_strides[i])
            convh=(convh-conv_kers[i]+convi_stride_h)//(convi_stride_h)
            convw=(convw-conv_kers[i]+convi_stride_w)//(convi_stride_w)
        self.out_size=out_nodes*convh*convw
        
    def forward(self,x):
        '''
            When called, this model is runned on the given batch.
            
            Inputs:
                x: tensor; the tensor containing the input batch
            Outputs:
                x: tensor; the output tensor of the model. The result of applying the model on the given input batch.
        '''
        x=self.conv_sub_1(x)
        x=self.conv_sub_2(x)
        return x
        
        
    