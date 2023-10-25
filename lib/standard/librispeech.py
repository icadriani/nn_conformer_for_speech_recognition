import json
import librosa
import os
from tqdm import tqdm
from colorama import Fore
from random import shuffle
import numpy as np
import torch
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset
from lib.standard.myvocab import myVocab
from lib.standard.wordpiecemodel import WPM
from math import floor
from functools import reduce
from itertools import product
# from torchtext.transforms import SpmTokenizerTransform
np.seterr(divide = 'ignore') 

class LibriSpeech(Dataset):
    '''
        class LibriSpeech
        
        This classs handles the ononymous dataset from torch
        
        Inputs:
            hp: HParams; class of hyperparameters
    '''
    def __init__(self,hp):
        super(LibriSpeech,self).__init__()
        # self.hp.base_dir=base_dir
        # self.data_dir=os.path.join(base_dir,'data')
        self.hp=hp
        self.drive_librispeech_dir=os.path.join(hp.data_dir,'LibriSpeech')
        self.librispeech_dir='LibriSpeech'
        if not os.path.exists(self.librispeech_dir):
            os.mkdir(self.librispeech_dir)
        self.lm_dir=os.path.join(self.drive_librispeech_dir,'LibriSpeech_LM_data')
        self.lm_corpus_dir=os.path.join(self.lm_dir,'librispeech-lm-corpus','corpus')
        #self.small_dir=os.path.join(self.librispeech_dir,'small_segmented')
        self.data=self.get_data()
        hp.set_input_dim(self.input_rows,self.input_cols)
        hp.set_vocab_len(len(self.vocab.vocab))
        # print(speech.max_len,speech.max_og_len)
        hp.set_max_len(self.max_len)
        hp.set_target_max_len(self.max_target_len)
        hp.set_standard_out_size(len(self.vocab.vocab)*self.max_target_len)
        hp.set_space_index(self.vocab.vocab[self.vocab.space_token])
        hp.set_blank_index(self.vocab.vocab[self.vocab.blank_token])
        # if hp.lm:
        #     self.lm=None
        #     self.get_lm(hp.ngram)
        #self.make_dataset()
    def get_data(self):
        '''
            Gathers the data and builds the vocabulary.
            
            Inputs:
                None
            Outputs:
                data: dict; the data where the keys are the sets names such as train, dev-clean ecc.
        '''
        data={}
        # if os.path.exists(self.drive_librispeech_dir):
        #     to_copy=[self.hp.standard_train_type] if type(self.hp.standard_train_type)!=list else self.hp.standard_train_type
        #     to_copy+=['dev-clean','dev-other','test-clean','test-other']
        #     if not os.path.exists(self.librispeech_dir):
        #         os.mkdir(self.librispeech_dir)
        #     for x in to_copy:
        #         f=os.path.join(self.drive_librispeech_dir,x)
        #         t=os.path.join(self.librispeech_dir,'LibriSpeech',x)
        #         os.system('cp -r '+f+' '+t)
        if type(self.hp.standard_train_type)==list:
            data['train']=[]
            for x in self.hp.standard_train_type:
                data['train'].append(LIBRISPEECH(self.librispeech_dir,x,download=not os.path.exists(os.path.join(self.librispeech_dir,x))))
                f=os.path.join(self.librispeech_dir,x+'.tar.gz')
                os.system('rm '+f)
            # for x in self.hp.standard_train_type:
            #     tmp=LIBRISPEECH(self.librispeech_dir,x,download=not os.path.exists(os.path.join(self.librispeech_dir,x)))
            #     data['train']+=[y[:3] for y in tmp]
        else:
            data['train']=LIBRISPEECH(self.librispeech_dir,self.hp.standard_train_type,download=not os.path.exists(os.path.join(self.librispeech_dir,self.hp.standard_train_type)))
            f=os.path.join(self.librispeech_dir,self.hp.standard_train_type+'.tar.gz')
            os.system('rm '+f)
        # print(len(data['train']))
        # print(type(data['train']))
        # print(set([x[1] for x in data['train']]))
        # data['train']=[[list(data['train'][i]) for i in range(1000)]]
        # data['train']=[[list(x[i]) for i in range(3000)] for x in data['train']]
        # print([x for x in data['train'] if len(x)==0])
        # for i in range(50):
        #     print(train[i][2])
        data['dev-clean']=LIBRISPEECH(self.librispeech_dir,'dev-clean',download=not os.path.exists(os.path.join(self.librispeech_dir,'dev-clean')))
        # data['dev-clean']=[list(data['dev-clean'][i][:3]) for i in range(len(data['dev-clean']))]
        data['dev-other']=LIBRISPEECH(self.librispeech_dir,'dev-other',download=not os.path.exists(os.path.join(self.librispeech_dir,'dev-other')))
        # data['dev-other']=[list(data['dev-other'][i][:3]) for i in range(len(data['dev-other']))]
        data['test-clean']=LIBRISPEECH(self.librispeech_dir,'test-clean',download=not os.path.exists(os.path.join(self.librispeech_dir,'test-clean')))
        data['test-other']=LIBRISPEECH(self.librispeech_dir,'test-other',download=not os.path.exists(os.path.join(self.librispeech_dir,'test-other')))
        # data={k:[[data[k][i] for i in range(500)]] for k in data}#len(data['train']))]
        
        to_rm=[os.path.join(self.librispeech_dir,x) for x in os.listdir(self.librispeech_dir) if os.path.isfile(os.path.join(self.librispeech_dir,x))]
        to_rm+=[os.path.join(self.librispeech_dir,'LibriSpeech',x) for x in os.listdir(os.path.join(self.librispeech_dir,'LibriSpeech')) if os.path.isfile(os.path.join(self.librispeech_dir,'LibriSpeech',x))]
        # print(to_rm,os.listdir(self.librispeech_dir),os.listdir(os.path.join(self.librispeech_dir,'LibriSpeech')))
        for x in to_rm:
            os.system('rm '+x)
            
        # for k in data:
        #     if k!='train':
        #         data[k]=list(data[k])
        # drive_mels=os.path.join(self.drive_librispeech_dir,'mels')
        # if type(self.hp.standard_train_type)==list:
        #     train_mels_dirs=[os.path.join(drive_mels,x) for x in self.hp.standard_train_type]
        # else:
        #     train_mels_dirs=[os.path.join(drive_mels,self.hp.standard_train_type)]
        # mels_dirs=[os.path.join(drive_mels,x) for x in ['dev-clean','dev-other','test-clean','test-other']]
        # mels=drive_mels.replace(self.drive_librispeech_dir,self.librispeech_dir)
        # if not os.path.exists(mels):
        #     os.mkdir(mels)
        # for x in mels_dirs:
        #     os.system('cp -r '+x+' '+mels)
        # for x in train_mels_dirs:
        #     os.system('cp -r '+x+'* '+os.path.join(mels,'train'))
        # for x in train_mels_dirs:
            
        #     for y in 
        # cp -r /content/drive/MyDrive/nn22/data/LibriSpeech/mels/ /content/LibriSpeech/mels/train/
        if self.hp.wpm:
            self.vocab=WPM(self.hp.base_dir,data['train'],ntokens=self.hp.ntokens,unk_tol=self.hp.unk_tol)
            # self.vocab=WPM(self.hp.base_dir,ntokens=self.hp.ntokens)
        else:
            self.vocab=myVocab(self.hp.base_dir,data['train'],ntokens=self.hp.ntokens)
            # self.vocab=myVocab(self.hp.base_dir,ntokens=self.hp.ntokens)
        # data={'train':list(train),'dev-clean':dev_clean,'dev-other':dev_other,'test-clean':test_clean,'test-other':test_other}
        # data['train'],max_train_len=self.process_data(train,'train',vocab)
        # data['dev-clean'],max_dev_clean_len=self.process_data(dev_clean,'dev-clean',vocab)
        # data['dev-other'],max_dev_other_len=self.process_data(dev_other,'dev-other',vocab)
        # data['test-clean'],max_test_clean_len=self.process_data(test_clean,'test-clean',vocab)
        # data['test-other'],max_test_other_len=self.process_data(test_other,'test-other',vocab)
        # print(len(train),len(dev_clean),len(dev_other),len(test_clean),len(test_other)
        # max_train_len=self.get_max(0,data['train'],lambda x:floor(x[0].shape[1]/self.hp.hop_length)+1,-1)
        # max_len_notok={}
        if type(data['train'])==list:
            max_train_len=0
            max_target_train_len=0
            min_train_mels=float('inf')
            max_train_mels=0
            # train_max_len_notok=[]
            for i in range(len(data['train'])):
                curr_max_train_len,curr_max_target_train_len,data['train'][i]=self.get_max_lens(data,'train',i)
                max_train_len=max(max_train_len,curr_max_train_len)
                max_target_train_len=max(max_target_train_len,curr_max_target_train_len)
                # train_max_len_notok.append(curr_max_len_notok)
                # curr_min_train_mels,curr_max_train_mels=self.get_min_max_mels_values(data,'train',i)
                # min_train_mels=min(min_train_mels,curr_min_train_mels)
                # max_train_mels=max(max_train_mels,curr_max_train_mels)
            # train_max_len_notok=[[x-len(data['train'][i-1]) for x in train_max_len_notok[i]] if i>0 else train_max_len_notok[i] for i in range(len(data['train']))]
            # max_len_notok['train']=train_max_len_notok
        else:            
            max_train_len,max_target_train_len,data['train']=self.get_max_lens(data,'train')
            data['train']=[data['train']]
            # min_train_mels,max_train_mels=self.get_min_max_mels_values(data,'train')
            # print(max_train_len,floor(max_train_len/self.hp.hop_length)+1)
        max_dev_clean_len,max_target_dev_clean_len,data['dev-clean']=self.get_max_lens(data,'dev-clean')
        max_dev_other_len,max_target_dev_other_len,data['dev-other']=self.get_max_lens(data,'dev-other')
        max_test_clean_len,max_target_test_clean_len,data['test-clean']=self.get_max_lens(data,'test-clean')
        max_test_other_len,max_target_test_other_len,data['test-other']=self.get_max_lens(data,'test-other')
        
        # min_dev_clean_mels,max_dev_clean_mels=self.get_min_max_mels_values(data,'dev-clean')
        # min_dev_other_mels,max_dev_other_mels=self.get_min_max_mels_values(data,'dev-other')
        # min_test_clean_mels,max_test_clean_mels=self.get_min_max_mels_values(data,'test-clean')
        # min_test_other_mels,max_test_other_mels=self.get_min_max_mels_values(data,'test-other')
        # max_dev_clean_len=max([x[0].shape[1] for x in data['dev-clean']])
        # max_dev_other_len=max([x[0].shape[1] for x in data['dev-other']])
        # max_test_clean_len=max([floor(x[0].shape[1]/self.hp.hop_length)+1 for x in data['test-clean']])
        # max_test_other_len=max([floor(x[0].shape[1]/self.hp.hop_length)+1 for x in data['test-other']])
        self.max_len=max(max_train_len,max_dev_clean_len,max_dev_other_len,max_test_clean_len,max_test_other_len)
        self.max_len=floor(self.max_len/self.hp.hop_length)+1
        self.max_len+=(3-self.max_len%3)
        print('max len',self.max_len)
        # max_target_train_len=max([len(self.vocab.parse(x[2])) for x in data['train']])
        # print([len(x[2].strip().split()) for x in train])
        # print(self.max_target_train_len)
        # max_target_dev_clean_len=max([len(self.vocab.parse(x[2])) for x in data['dev-clean']])
        # max_target_dev_other_len=max([len(self.vocab.parse(x[2])) for x in data['dev-other']])
        # self.max_target_test_clean_len=max([len(x[2].strip().split()) for x in data['test-clean']])
        # self.max_target_test_other_len=max([len(x[2].strip().split()) for x in data['test-other']])
        if self.hp.max_target_len is None:
            self.max_target_len=max(max_target_train_len,max_target_dev_clean_len,max_target_dev_other_len,max_target_test_clean_len,max_target_test_other_len)
        else:
            self.max_target_len=self.hp.max_target_len
        # self.min_mels_value=min(min_train_mels,min_dev_clean_mels,min_dev_other_mels,min_test_clean_mels,min_test_other_mels)
        # self.max_mels_value=max(max_train_mels,max_dev_clean_mels,max_dev_other_mels,max_test_clean_mels,max_test_other_mels)
        # self.max_len=[floor(x)]
        # self.too_long={k:self.get_too_long_idx(data[k],100) for k in data}
        self.input_rows=self.hp.n_mels
        self.input_cols=self.max_len
        # print(type(data['train']))
        if type(data['train'])==list:
            start_train_idxes=[len(x) for x in data['train']]
            # print(start_train_idxes)
            start_train_idxes=[sum(start_train_idxes[:i+1]) for i in range(len(start_train_idxes))]
            self.start_idxes={'train':start_train_idxes}
            # self.start_idxes.update({k:0 for k in data if 'dev' in k or 'test' in k})
            if self.hp.max_target_len is None:
                train_idxes=[[y for y in range(len(x)) if self.vocab.is_tollerable(x[y][2])] for x in data['train']]
                # self.idxes={'train':list(range(sum([len(x) for x in data['train']])))}
            else:
                train_idxes=[self.long_enough(x,self.max_target_len) for x in data['train']]
                train_idxes=[[y for y in x if self.vocab.is_tollerable(data['train'][x][y][2])] for x in train_idxes]
            train_idxes=[[x+start_train_idxes[i-1] for x in train_idxes[i]] if i>0 else train_idxes[i] for i in range(len(start_train_idxes))]
            self.idxes={'train':reduce(lambda x,y: x+y,train_idxes)}
            # print(len(self.idxes['train']))
            # print(start_train_idxes)
            # print(self.start_idxes,self.idxes['train'][-1],len(self.idxes['train']))
            # print(self.max_target_len,self.hp.max_target_len)
        else:
            # print(data['train'])
            if self.hp.max_target_len is None:
                self.idxes={'train':list(range(len(data['train'])))}
            else:
                self.idxes={'train':self.long_enough(data['train'],self.max_target_len)}
            self.idxes['train']=[x for x in self.idxes['train'] if self.vocab.is_tollerable(data['train'][x][2])]
            # self.idxes['train']=[x for x in self.idxes['train'] if x not in self.too_long['train']]
        if self.hp.max_target_len is None:
            self.idxes.update({k:list(range(len(data[k]))) for k in data if 'dev' in k or 'test' in k})
        else:
            self.idxes.update({k:self.long_enough(data[k],self.max_target_len) for k in data if 'dev' in k or 'test' in k})
        # print(max_len_notok)
        # self.idxes={k:[i for i in self.idxes[k] if i not in max_len_notok[k]] for k in self.idxes}
        # print(self.idxes.keys())
        # self.idxes={k:[x for x in v if self.vocab.is_tollerable(data[k][x])] for k,v in self.idxes.items()}
        # self.idxes={k:self.idxes[:1000] for k in self.idxes}
        # print(len(data['train'][0]),len(self.idxes['train']))
        return data
    # def get_max(self,x,data,func,i):
    #     if i<len(data)-1:
    #         return self.get_max(data[i],data,func,i+1)
    #     else:
    #         return max(x,func(data[i])
    def long_enough(self,data,n):
        '''
            Audios with very long transcripts are filtered out
            
            Inputs:
                data: list; the dataset (ex. train dataset) to be filtered
                n: int; the maximum transcript lenght
            Outputs:
                idx: list; list of the indices of the audios that have transcripts that are not too long
        '''
        idx=[]
        for i,x in enumerate(data):
            p=self.vocab.parse(x[2])
            # print(p)
            # print('-----------')
            if len(p)>0 and len(p)<n:
                idx.append(i)
        return idx
    def get_max_lens(self,data,dataset_type='train',idx=0):
        '''
            Gets the maximum mel length and the maximum transcript length that are under a certain threshold declared in hyperparameters class.
            Finally it filters out data that have one of these values over the corresponding thesholds.
            
            Inputs:
                data: dict; the input data
                dataset_type: str; the dataset type as in train, test ecc. Default: 'train'
                idx: int; the dataset data[dataset_type] can be a LIBRISPEECH object or a list of LIBRISPEECH objects. In the latter case the idx must be specified. Default: 0
            Outputs:
                max_len: int; maximum mel length
                max_target_len: int; maximum transcript length
                data: list; filtered dataset
        '''
        max_len=[]
        max_target_len=[]
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        data=data[dataset_type]
        idxes=[]
        if type(data)==list:
            data=data[idx]
            t=tqdm(range(len(data)),unit='audio',dynamic_ncols=True,bar_format=bar_format,desc=dataset_type[0].upper()+dataset_type[1:]+' '+str(idx+1))
        else:
            t=tqdm(range(len(data)),unit='audio',dynamic_ncols=True,bar_format=bar_format,desc=dataset_type[0].upper()+dataset_type[1:])
        for i in t:
            x=data[i]
            # print(floor(x[0].shape[1]/self.hp.hop_length)+1,self.hp.max_len,floor(x[0].shape[1]/self.hp.hop_length)+1<self.hp.max_len)
            l=floor(x[0].shape[1]/self.hp.hop_length)+1
            l+=(3-l%3)
            if self.hp.max_len is None or l<self.hp.max_len:
                max_len.append(x[0].shape[1])
                max_target_len.append(len(self.vocab.parse(x[2])))
            else:
                idxes.append(i)
            if i==len(data)-1:
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        max_len=max(max_len)
        max_target_len=max(max_target_len)
        # print(idxes)
        data=[data[i] for i in range(len(data)) if i not in idxes]
        return max_len,max_target_len,data
    def get_start_idx(self,i,idx,dataset_type='train'):
        '''
            The dataset can be a list of LIBRISPEECH objects. If this the case then given an index of the flatten list, the index of the original one is retrieved. For example. idx 5 given from a list [0,1,2,3,4,5,6,7] can corrispond to i=2 in [[0,1],[1,2,3],[4,5,6,7]] and starting index 4.
            
            Inputs:
                i: int; the index in the original list. This function is recursive so on the first call must be 0
                idx: int; index in the flatten list
                dataset_type: str; the dataset type as in train, test ecc. Default: 'train'
            Outputs:
                i: int; the index in the original list.
                the index in the flatten list of the first element in the ith list the original list 
        '''
        if idx<self.start_idxes[dataset_type][i]:
            return i,self.start_idxes[dataset_type][i-1] if i>0 else 0
        return self.get_start_idx(i+1,idx,dataset_type)
    def get_mels(self,data,dataset_type='train',progress_bar=False):
        '''
            Computation of the log-mel spectograms of the audios
            
            Inputs: 
                data: dict or list or LIBRISPEECH; If a dict is given the right dataset_type has to be given.
                dataset_type: str; train, validation, ecc. Default: 'train'
                progress_bar: bool; whether a progress bar is to be printed. Default: False
            Outputs:
                mels: list[]; the log-mel spectograms of the audio from the input dataset
        '''
        mels=[]
        d=os.path.join(self.librispeech_dir,'mels')
        if not os.path.exists(d):
            os.mkdir(d)
        d=os.path.join(d,dataset_type)
        if not os.path.exists(d):
            os.mkdir(d)
        t=range(len(data))
        if progress_bar:
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            t=tqdm(t,unit='audio',dynamic_ncols=True,bar_format=bar_format,desc=dataset_type[0].upper()+dataset_type[1:]+' mels')
        for i in t:
            elem=list(data[i])
            filename=os.path.join(d,'mel_'+'_'.join([str(x) for x in elem[-3:]])+'.csv')
            if self.hp.read_mels and os.path.exists(filename):
                mel=self.read_mel(filename)
            else:
                if torch.is_tensor(elem[0]):
                    elem[0]=elem[0].squeeze().numpy()
                mel=librosa.feature.melspectrogram(y=elem[0],sr=elem[1],n_mels=self.hp.n_mels)
                mel=np.where(mel<1e-10,0,np.log(mel))
                # print(np.)
                # print(mel)
                # mel=mel.tolist()
                # mel=librosa.power_to_db(mel)
                # print(elem[0].shape,mel.shape,elem[1],elem[0].shape[1]/self.hp.hop_length)
                # print(type(mel),mel.shape)
                # if dataset_type in self.idxes:
                # else:
                #     filename=os.path.join(d,'mel_'+str(i)+'.csv')
                self.save_mel(filename,mel)
            mel=np.array(mel)
            # print(mel.shape)
            # print(type(mel),mel.shape)
            mels.append(mel)
            if progress_bar and i==len(data)-1:
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        return mels
    def read_mel(self,filename):
        '''
            Reads the log-mel spectogram from a file
            
            Inputs:
                filename: str; name of the file where the log-mel spectrogram is saved
            Outputs:
                mel: list; the loaded log-mel spectrogram
        '''
        with open(filename,'r') as f:
            mel=f.read().strip().split('\n')
        mel=[x.strip().split(',') for x in mel]
        mel=[[float(x.strip()) for x in m] for m in mel]
        # mel=np.array(mel)
        # err=mel.shape[0]!=self.hp.n_mels or floor(mel.shape[/self.hp.hop_length)+1
        return mel
    def save_mel(self,filename,mel):
        '''
            Saves the given log-mel spectogram 
            
            Inputs:
                filename: str; the name of the file where to store the spectogram
                mel: np.array; the log-mel spectogram
            Outputs:
                None
        '''
        mel=mel.tolist()
        mel='\n'.join([','.join([str(round(x,6)) for x in m]) for m in mel])
        with open(filename,'w+') as f:
            f.write(mel)
    # def get_min_max_mels_values(self,data,dataset_type,i=None):
    #     data=data[dataset_type]
    #     if i is not None:
    #         data=data[i]
    #     mels=self.get_mels(data,dataset_type,progress_bar=True)
    #     min_value=max(0,min([min(y) for x in mels for y in x])*1.0)
    #     max_value=max([max(y) for x in mels for y in x])*1.0
    #     return min_value,max_value
    def process_data(self,data,dataset_type):
        '''
            This function process the data. The log-mel spectograms of the audios are computed as well as the encoding of the labels.
            Inputs:
                data: dict; data before processing. The keys of the dictornary correspond to the set name as in train, validation, test ecc. The data in each dataset must follow the LIBRISPEECH structure.
            Outputs:
                data: dict; data after preprocessing
        '''
        mels=self.get_mels(data,dataset_type)
        max_len=max([len(x[0]) for x in mels])
        min_value=min([min(y) for x in mels for y in x])
        mels=[[[x-min_value for x in y] for y in z] for z in mels]
        max_value=max([max(y) for x in mels for y in x])
        mels=[[[x/max_value for x in y] for y in z] for z in mels]
        # print(self.min_mels_value,self.max_mels_value)
        # print(min_value,max_value)
        # print(mels)
        # print(type(mels))
        # mels=np.array(mels)
        # print(type(mels),np.nanmin(mels,axis=1,keepdims=True))
        # nanmin=[np.nanmin(m) for m in mels]
        # print(nanmin.shape)
        # nanmin=np.expand_dims(nanmin,axis=1)
        # print(nanmin)
        # mels=mels-nanmin
        # nanmax=np.nanmax(mels,axis=1)
        # nanmax=np.expand_dims(nanmax,axis=1)
        # mels=mels/np.nanmax(mels,axis=1,keepdims=True)
        # print(mels)
        # input_rows=len(data[0])
        # input_cols=max_len
        # mels=[mels[i:i+self.hp.batch_size] for i in range(0,len(mels),self.hp.batch_siz
        # target=[]
        # for x in data:
        #     target+=self.vocab.one_hot_sentence(x[2])
        # target=[self.vocab.one_hot_sentence(x[2]) for x in data]
        target=[self.vocab.parse(x[2]) for x in data]
        # target=[[[0]*len(target[0][0])]*len(target[0])]+target
        # d=[self.vocab.decode_one_hot(x) for x in target]
        # print(d)
        # target=[self.one_hot_encoding(x) for x in target]
        # target=[target[i:i+self.hp.batch_size] for i in range(0,len(target),self.hp.batch_size)]
        # print(target[0])
        # if SpecAugment:
        tau=[len(x[0]) for x in mels]# for y in x]
        target_lens=[len(x) for x in target]
        # lengths=self.vocab.get_word_lengths(target.flatten())
        batch={'input':{'mels':mels,'tau':tau},'target':{'transcripts':target,'lens':target_lens}}#,'word_lengths':lengths}# for i in range(mels)]
        # else:
        #     batch={'input':{'mels':mels},'target':target}# for i in range(mels)]
        # print(batch[0])
        return batch
    # def process_output(self,predicted,target):
    #     t_=' '.join(target)
    #     t_=t_.split(self.vocab.pad_token)
    #     t=[len(x.split()) for x in t_]
    #     tt=[]
    #     z=0
    #     for ti in t:
    #         if ti!=0:
    #             if z!=0:
    #                 tt.append([z,False])
    #                 z=0
    #             tt.append([ti,True])
    #         else:
    #             z+=1
    #     pp=[]
    #     s=0
    #     for x in tt:
    #         pi=predicted[s:s+x[0]]
    #         s+=x[0]
    #         if x[1]:
    #             pp.append(pi)
                
    #     tt=[x for x in t_ if x!='']
    #     pp=[' '.join(x) for x in pp]
    #     return pp,tt
    def shuffle(self,dataset_type='train'):
        '''
            Shuffles the dataset
            
            Inputs:
                dataset_type: str; type of dataset: train, mix or ecc. Defaults: 'train' 
            Outputs:
                None
        '''
        shuffle(self.idxes[dataset_type])
    def get_item(self,i,dataset_type='train'):
        '''
            Retrives the element of the dataset present at index i
            
            Inputs:
                i: int; index of the element to retreive
                dataset_type: str; the dataset as in train, validation, ecc. Default: 'train'
            Outputs:
                dataset element at index i; dict
        '''
        if dataset_type=='train':
            ii,s=self.get_start_idx(0,i,dataset_type)
            return self.data[dataset_type][ii][i-s]
        else:
            return self.data[dataset_type][i]
    def get_batch(self,i,dataset_type='train'):
        '''
            Retrives the ith batch
            
            Inputs:
                i: int; index of the batch to be retrived
                dataset_type: The dataset from which the dataset is be retreived (train, validation, ecc). Default: 'train'
            Outputs:
                batch: dict; The retrevied batch
            
        '''
        # if dataset_type=='train':
        idxes=self.idxes[dataset_type][i*self.hp.batch_size:i*self.hp.batch_size+self.hp.batch_size]
        batch=[self.get_item(idx,dataset_type) for idx in idxes]
        # else:
        #     start=i*self.hp.batch_size
        #     end=min(len(self.data[dataset_type]),start+self.hp.batch_size)
        #     # print(start,start+self.hp.batch_size,len(self.data[dataset_type]),end)
        #     batch=[self.data[dataset_type][i] for i in range(start,end)]
        # print(batch)
        batch=self.process_data(batch,dataset_type,SpecAugment)
        # if i==535:
        #     print()
        #     print(type(batch['input']['mels']),type(batch['input']['mels'][0]),type(batch['input']['mels'][-1]),type(batch['input']['mels'][0][0]),type(batch['input']['mels'][-1][-1]))
        #     print(len(batch['input']['mels']),len(batch['input']['mels'][0]),len(batch['input']['mels'][0][0]))
        #     print([len(x[0]) for x in batch['input']['mels']])
        #     print(self.max_len)
        batch['unpadded_len']=len(batch['input']['mels'])
        # print(self.max_len,[len(y) for x in batch['input']['mels'] for y in x])
        batch['input']['mels']=[[self.padding(y,self.max_len-len(y)) for y in x] for x in batch['input']['mels']]
        # if i==535:
        #     # print(np.array(batch['input']['mels']).shape)
        #     print(len(batch['input']['mels']),len(batch['input']['mels'][0]),len(batch['input']['mels'][0][0]))
        batch['input']['mels']=self.padding(batch['input']['mels'],self.hp.batch_size-len(batch['input']['mels']))
        # print(len(batch['input']['mels']))
        # if i==535:
        #     print(len(batch['input']['mels']),len(batch['input']['mels'][0]),len(batch['input']['mels'][0][0]))
        #     print(batch['input']['mels'][0])
        #     print(set(reduce(lambda x,y: x+y,[reduce(lambda x,y: x+y,[list(set([type(z) for z in x])) for x in y]) for y in batch['input']['mels']])))
        #     print(set(reduce(lambda x,y: x+y,[list(set([type(x) for x in y])) for y in batch['input']['mels']])))
        #     print(np.array(batch['input']['mels']).shape)
        # if SpecAugment:
        batch['input']['tau']=self.padding(batch['input']['tau'],self.hp.batch_size-len(batch['input']['tau']))
        # if i==535:
        #     print('hi')
        batch['input']['mels']=torch.FloatTensor(np.array(batch['input']['mels'])).to(self.hp.device).unsqueeze(1)
        # mask=batch['input']['mels'].lt(0)
        # batch['input']['mels']=torch.masked_fill(batch['input']['mels'],mask,value=0.0)
        # if i==535:
        #     print('--------------------')
        batch['input']['tau']=torch.LongTensor(np.array(batch['input']['tau'])).to(self.hp.device)
        batch['target']['transcripts']=[self.padding(x,self.max_target_len-len(x)) for x in batch['target']['transcripts']]
        batch['target']['transcripts']=self.padding(batch['target']['transcripts'],self.hp.batch_size-len(batch['target']['transcripts']))
        # print(batch['target'])
        # for x in batch['target']:
        #     print(len(x),self.max_target_train_len-len(x))
        #     print(x)
        #     print([type(y) for y in x])
        batch['target']['transcripts']=torch.LongTensor(np.array(batch['target']['transcripts'])).to(self.hp.device)#.unsqueeze(1)
        batch['target']['lens']=self.padding(batch['target']['lens'],self.hp.batch_size-len(batch['target']['lens']))
        batch['target']['lens']=torch.LongTensor(np.array(batch['target']['lens'])).to(self.hp.device)
        # batch['word_lengths']=self.vocab.get_word_lengths(batch['target'].flatten())
        # print(batch['target'].shape)
        # batch['target']=torch.flatten(batch['target'])#.unsqueeze(1)
        # batch['target']=batch['target'].view((batch['target'].shape[0]*batch['target'].shape[1],self.hp.ntokens))#.unsqueeze(1)
        return batch
    def padding(self,l,left):
        '''
            Pads the input list l with left elements of the same shape of the elements in l with a constant zero value
            
            Inputs:
                l: list; the input list to be padded
                left: int; the number of the pads to be added to the list
            Outputs:
                the input list with padding; list
        '''
        if left==0: return l
        return l+np.zeros([left]+list(np.array(l).shape)[1:]).tolist()
    #     if left==0:
    #         return l
    #     shape=np.array(l).shape[1:]
    #     pad=self.pad(0,shape)
    #     pad=[pad]*left
    #     # print(type(l))
    #     return l+pad
    # def pad(self,pad,shape):
    #     if len(shape)==0:
    #         return pad
    #     return self.pad([pad]*shape[-1],shape[:-1])
    def mix_datasets(self,U,targets):
        '''
            Mix the training dataset with a dataset where the labels are generated by a model
            
            Inputs:
                U: the dataset with no labels (in this implementation the ground truth exists already but is ignored to mimic the situation in which they don't)
                targets: list[str]; the generated labels
            Outputs:
                None
        '''
        # print(U[0])
        # print(len(U[0]))
        # print(targets[0])
        # print(len(U),len(targets))
        l=min(len(U),len(targets))
        # print(len(U))
        # print(U.data[0][0].shape)
        # print(targets)
        u=[[U.data[i][0],U.data[i][1],targets[i]] for i in range(l) if len(self.vocab.parse(targets[i]))<=self.max_target_len]
        # print(len(u))
        # for x in u: print(x)
        self.data['mix']=self.data['train'].copy()+u
        self.idxes['mix']=self.idxes['train']+list(range(len(u)))
        self.start_idxes['mix']=self.start_idxes['train']+[sum(self.start_idxes['train'])+len(u)]
        shuffle(self.data['mix'])
    # def compute_lm(self,ngram=1):
    #     '''
    #         Computes a language model from the data transcripts
    #     '''
    #     count={k:0 for k in self.vocab.vocab.get_itos()}
    #     if ngram>1:
    #         keys=product([k for k in count.keys() if '<' not in k],repeat=ngram)
    #         count_n={str(list(k)):0 for k in keys}
    #         count.update(count_n)
    #     #     for x in count_n.keys():
    #     #         print(x)
    #     # return
    #     if os.path.exists(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count.txt')):
    #         with open(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count.txt'),'r') as f:
    #             s=f.read().strip().split('\n')
    #             s=[x.split(';') for x in s]
    #             # print(s[:10])
    #             count.update({x[0]:int(x[1]) for x in s})
    #     n=0
    #     if os.path.exists(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count_n.txt')):
    #         with open(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count_n.txt'),'r') as f:
    #             n=int(f.read().strip())
    #     # train=self.data['train'] if type(self.data['train'])!=list else reduce(lambda x,y: x+y,self.data['train'])
    #     # bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
    #     # t=tqdm(range(len(train)),unit='utterance',dynamic_ncols=True,bar_format=bar_format,desc='LibriSpeech')
    #     # for i in t:
    #     #     x=train[i][2]
    #     #     x=self.vocab.partial_parse(x)
    #     #     n+=len(x)
    #     #     for p in x:
    #     #         if p in count:
    #     #             count[p]+=1
    #     #     if i==len(train)-1:
    #     #         t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
    #     count,n=self.get_corpus(count,n,ngram)
    #     with open(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count.txt'),'w+') as f:
    #         s='\n'.join([str(k)+';'+str(v) for k,v in count.items()])
    #         f.write(s)
    #     with open(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count_n.txt'),'w+') as f:
    #         # s='\n'.join([str(k)+','+str(v) for k,v in count.items()])
    #         f.write(str(n))
    #     l=len(self.vocab.vocab)
    #     # print(l)
    #     # count=self.compute_count(ngram)
    #     # print(count.keys())
    #     # print('hi')
    #     count[self.vocab.space_token]=0
    #     lm=[count[self.vocab.vocab.lookup_token(i)] for i in range(l)]
    #     if ngram>1:
    #         return lm,count
    #     return lm
    # def get_lm(self,ngram=1,transcripts=None):
    #     if ngram==1:
    #         if self.lm is None:
    #             lm=self.compute_lm(ngram)
    #             # lm[self.vocab.vocab[self.vocab.space_token]]=0
    #             lm=torch.FloatTensor(lm).to(self.hp.device)
    #             # n=sum(list(count.values()))
    #             n=torch.max(lm,-1)[0].item()
    #             lm=lm/n
    #             # print(torch.max(lm,-1)[0].item())
    #             # lm=torch.nn.LogSoftmax(-1)(lm)
    #             # lm=-torch.log(lm)
    #             # lm=[x/l for x in lm]
    #             lm=lm.view((1,1,lm.shape[-1]))
    #             lm=lm.expand(self.hp.batch_size,self.max_len,-1)
    #             # print(torch.max(lm[0][0],-1)[0].item())
    #             lm=lm*100
    #             lm=lm+1e-10
    #             lm=torch.log(lm)
    #             lm=self.hp.beta*lm
    #             # print(torch.max(lm[0][0],-1)[0].item())
    #             self.lm=lm
    #         return self.lm
    #     else:
    #         if self.lm is None:
    #             lm,self.count=self.compute_lm(ngram)
    #             self.lm=[lm]
    #             for x in self.vocab.vocab.get_itos():
    #                 c=[]
    #                 for y in self.vocab.vocab.get_itos():
    #                     k=str([x,y])
    #                     if k in self.count:
    #                         c.append(self.count[k])
    #                     else:
    #                         c.append(0)
    #                 self.lm.append(c)
    #             self.lm=torch.FloatTensor(self.lm).to(self.hp.device)
    #             n=torch.max(self.lm,-1)[0]
    #             n=n+1e-10
    #             n=n.unsqueeze(-1)
    #             n=n.expand((-1,self.lm.shape[-1]))
    #             # print(self.lm.shape,n.shape)
    #             self.lm=self.lm/n
    #             self.lm=self.lm*100
    #             self.lm=self.lm+1
    #             self.lm=torch.log(self.lm)
    #             # lm=lm.unsqueeze(0)
    #             # lm=lm.expand(self.hp.batch_size,-1,-1)
    #             # self.lm=-torch.log(self.lm)
    #             print(torch.max(self.lm.flatten()).item())
    #             self.lm=self.hp.beta*self.lm
    #         if transcripts is not None:
    #             lm=[]
    #             for transcript in transcripts:
    #                 clm=[self.lm[0]]
    #                 prec=None
    #                 for j,y in enumerate(transcript[:-1]):
    #                     if type(y)==str:
    #                         # print(y)
    #                         y=self.vocab.vocab[y]
    #                         # print(y)
    #                     else:
    #                         y=y.item()
    #                     if prec is None or y!=self.vocab.vocab[self.vocab.blank_token]:
    #                         prec=y
    #                     else: #if y==self.vocab.vocab[self.vocab.blank_token]:
    #                         clm[-1]=clm[-1]*0
                            
    #                     clm.append(self.lm[prec])
    #                     # c=[]
    #                     # for x in self.vocab.vocab.get_itos():
    #                     #     k=str([prec,x])
    #                     #     if k in self.count and 'blank' not in y and ((type(transcript[j+1])==str and 'blank' not in transcript[j+1]) or 'blank' not in self.vocab.vocab.lookup_token(transcript[j+1])):
    #                     #         c.append(self.count[k])
    #                     #     else:
    #                     #         c.append(0)
    #                     # clm.append(c)
    #                     # if 'blank' not in y:
    #                     #     prec=y
    #                 clm=torch.stack(clm)
                    
    #                 # clm=clm+[[0]*len(self.vocab.vocab)]*(self.max_len-len(clm))
    #                 lm.append(clm)
    #             lm=torch.stack(lm)
    #             # lm=torch.FloatTensor(lm).to(self.hp.device)
    #             # print(lm.shape)
    #             lm=torch.nn.functional.pad(lm,(0,self.max_len-lm.shape[1],0,len(self.vocab.vocab)-lm.shape[-1]))
    #             # mask=transcrips.
    #             # n=torch.max(lm,-1)[0]#.item()
    #             # print(lm.shape,n.shape)
    #             # if ngram>1:
    #             #     n=n.unsqueeze(-1)
    #             #     n=n.expand(-1,-1,self.hp.ntokens)
    #             # n=n+1e-10
    #             # lm=lm/n
    #             # # lm=lm.unsqueeze(0)
    #             # # lm=lm.expand(self.hp.batch_size,-1,-1)
    #             # lm=self.hp.beta*lm
    #             # print(lm[0])
    #             # print(lm)
    #             return lm
    # def get_corpus(self,count,n,ngram=1):
    #     book_files=[os.path.join(self.lm_corpus_dir,x,x+'.txt') for x in os.listdir(self.lm_corpus_dir)]
    #     # shuffle(book_files)
    #     # amount=50
    #     start=0
    #     if os.path.exists(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count_start.txt')):
    #         with open(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count_start.txt'),'r') as f:
    #             start=int(f.read().strip())
    #     with open(os.path.join(self.drive_librispeech_dir,str(ngram)+'_lm_count_start.txt'),'w+') as f:
    #         # s='\n'.join([str(k)+','+str(v) for k,v in count.items()])
    #         f.write(str(start+amount))
    #     # book_files=book_files[start:start+amount]
    #     # books=[]
    #     # pronuncie=[]
    #     # alpha=re.compile("^[A-Z' ]+$")
    #     # roman_num=re.compile('^[IVXLCDM]+$')
    #     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
    #     t=tqdm(range(len(book_files)),unit='book',dynamic_ncols=True,bar_format=bar_format,desc='LibriSpeech LM corpus')
    #     for bf in t:
    #         with open(book_files[bf],'r') as f:
    #             book=f.read().strip().split('\n')
    #             # b=[self.vocab.parse(x) for x in b]
    #             book=[self.vocab.partial_parse(x) for x in book if x.strip()!='' and len(self.vocab.partial_parse(x))>0]
    #             # b=[self.tokenizer(x) for x in b]
    #             # b=[[w.lower() for w in x] for x in b]
    #             # b=[self.space_adjust(x) for x in b]
    #             # print(b[0])
    #             # b=reduce(lambda x,y:x+y,b)
    #             for b in book:
    #                 n+=len(b)
    #                 if ngram==1:
    #                     for x in b:
    #                         if x in count:
    #                             count[x]+=1
    #                 else:
    #                     if b[0]!=self.vocab.space_token:
    #                         if '<' not in b[0] and b[0] in self.vocab.vocab:
    #                             # print(b[0])
    #                             count[b[0]]+=1
    #                     # else:
    #                     elif '<' not in b[1] and b[1] in self.vocab.vocab:
    #                             count[b[1]]+=1
    #                             # print(b[1])
    #                     for i in range(ngram-1,len(b)):
    #                         g=str(b[i-ngram+1:i+1])
    #                         if g in count:
    #                             count[g]+=1
                
    #             # b=self.clean_book(b,roman_num,alpha)
    #             # return
    #             # books.append(b)
    #         #     books+=b
    #         # # print(pronuncia_b_file)
    #         #     pronuncia_b_file=book_files[bf].replace('.txt','_pronuncia.txt')
    #         # # if not os.path.exists(pronuncia_b_file):
    #         #     pronuncia_b=self.vocab.get_pronuncie_data(b)
    #         #     pronuncia_b_='\n'.join(pronuncia_b)
    #         #     with open(pronuncia_b_file,'w+') as f:
    #         #             f.write(pronuncia_b_)
    #         # # else:
    #         # #     with open(pronuncia_b_file,'r') as f:
    #         # #         pronuncia_b=f.read().strip().split('\n')
    #         # pronuncie+=pronuncia_b
    #         if bf==len(book_files)-1:
    #             t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
    #     # print(len(books))
    #     # books_max_len=max([len(line) for line in books])
    #     # books=[line for line in books if len(line)>3*books_max_len//4]
    #     # print(len(books))
    #     # print(books)
    #     # corpus=[[pronuncie[i],books[i]] for i in range(len(books))]
    #     return count,n
    
        
        
                    
                
                

                
        