import os
import torch
from torch.utils.tensorboard import SummaryWriter
from math import inf
# import torch_xla.core.xla_model as xm
# xm.xla_device()

class HParams():
    def __init__(self,base_dir):
        self.base_dir=base_dir
        self.data_dir=os.path.join(base_dir,'data')
        self.model_dir=os.path.join(base_dir,'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.plots_dir=os.path.join(base_dir,'results')
        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)
        self.writer_path=os.path.join(self.plots_dir,'tensorboard')
        if os.path.exists(self.writer_path):
            os.system('rm -r '+self.writer_path)
        self.writer=SummaryWriter(self.writer_path)
        self.pretrained_model_path=os.path.join(self.model_dir,'pretrained_weights.pth')
        self.standard_model_path=os.path.join(self.model_dir,'standard_weights.pth')
        self.lm_model_path=os.path.join(self.model_dir,'lm_weights.pth')
        self.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.beta1=0.9
        self.beta=0.9
        self.ngram=2
        self.scale_parameter=False
        self.relative_step=False
        self.lr=1e-3
        self.pretraining_lr=1e-3
        self.batch_size=32
        self.ntokens=2048
        self.unk_tol=0.3
        self.epochs=5
        self.pretraining_epochs=2000
        # self.n_fft=1024
        self.hop_length=256
        self.n_mels=40
        self.n_conformers=10
        self.dropout=0.025
        self.pretraining_insize=1
        self.pretraining_convsub_out_size=256
        self.conv_sub_1_nodes=512
        self.conv_sub_1_kernel=32
        self.conv_sub_1_stride=(3,3)
        self.conv_sub_2_nodes=16
        self.conv_sub_2_kernel=13
        self.conv_sub_2_stride=(3,3)
        self.mask_probability=0.065
        self.mask_value=0
        self.target_context_vectors_size=320
        self.pretraining_linear_nodes=self.target_context_vectors_size
        self.conformer_ff1_linear1_nodes=1024
        #self.conformer_ff1_linear2_nodes=64
        self.conformer_ff2_linear1_nodes=1024
        self.conformer_dropout=0.5
        self.mhsa_num_heads=8
        self.conformer_pointwise_conv1_nodes=1024
        self.conformer_pointwise_conv1_kernel=1
        self.conformer_depthwise_conv_nodes=512
        self.conformer_depthwise_conv_kernel=33 # 3 for pytorch conformer
        self.conformer_depthwise_conv_stride=2
        self.conformer_pointwise_conv2_nodes=256
        self.conformer_pointwise_conv2_kernel=1
        self.pretraining_conformer_out_size=64
        self.pretraining_decoder_bidirectional=True
        self.pretraining_decoder_layers=1
        self.conformer_size=1024
        #self.pretraining_out_size=640
        self.encoded_features_out_size=128
        self.mask_change_every_n_steps=10
        # choose among 'train-clean-100', 'train-clean-360' and 'train-other-500'
        self.standard_train_type=['train-clean-360','train-other-500'] #'train-clean-100'
        # self.standard_train_type='train-clean-100'
        # choose among '10min', '1h', '10h'
        self.librilight_subset='10h'
        self.decoder_nodes=self.target_context_vectors_size//2 if self.pretraining_decoder_bidirectional else self.target_context_vectors_size
        self.standard_linear_nodes=512 #self.pretraining_linear_nodes
        self.standard_decoder_layers=1 #self.pretraining_decoder_layers
        self.standard_decoder_bidirectional=True #self.pretraining_decoder_bidirectional
        self.standard_decoder_nodes=2048 #self.decoder_nodes
        self.simplified_pretraining=False
        self.alpha_loss=0.1
        self.temperature_loss=0.1
        self.distractors_K=50
        self.temperature_tau=2
        self.warping_param_W=80
        self.warping_ntimes=1
        self.frequency_mask_param_F=27
        self.frequency_mask_ntimes=2
        self.time_multiplicity=10
        self.pm=0.05
        self.ps=0.05
        self.time_mask_param_T=100
        self.lm_ntokens=256
        self.adaptive_multiplicity=True
        self.adaptive_size=True
        self.wpm=True
        self.do_pretraining=False
        self.load_pretraining=True
        self.nst=False
        self.lm=False
        self.train_lm=False
        self.rel_att=True
        self.pretrained_conformer=False
        self.rel_pos_emb=self.rel_att
        self.ft_epochs=2
        self.ft_train_epochs=1
        self.lm_in_N=4
        self.lm_out_N=4
        self.input_embedding_size=320
        self.output_embedding_size=self.input_embedding_size
        self.lm_in_mhsa_num_heads=8
        self.lm_out_mhsa_num_heads=8
        self.lm_masked_out_mhsa_num_heads=8
        self.lm_innner_input_nodes=512
        self.lm_innner_output_nodes=512
        # self.lm_out_size=512
        self.lm_epochs=3
        self.lm_max_len=20
        self.read_mels=True
        self.projection_out_size=1024 #self.ntokens
        self.embedding_dim=300
        self.decoder_fc_nodes=self.projection_out_size
        self.max_target_len=None #50
        self.max_len=100 #50
    def set_max_len(self,max_len):
        self.max_len=max_len
        # self.projection_out_size*=self.max_len
    def set_target_max_len(self,max_len):
        self.max_target_len=max_len
        # self.projection_out_size=self.ntokens
    def set_vocab_len(self,l):
        # self.projection_out_size//=self.ntokens
        self.ntokens=l
        # self.projection_out_size*=self.ntokens
        # self.decoder_fc_nodes=l
    def set_standard_out_size(self,standard_out_size):
        self.standard_out_size=standard_out_size
    def set_input_dim(self,input_rows,input_cols):
        self.input_rows=input_rows
        self.input_cols=input_cols
    def set_space_index(self,space_idx):
        self.space_idx=space_idx
    def set_blank_index(self,blank_idx):
        self.blank_idx=blank_idx    
        