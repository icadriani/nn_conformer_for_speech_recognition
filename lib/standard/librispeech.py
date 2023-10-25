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
        self.hp=hp
        self.drive_librispeech_dir=os.path.join(hp.data_dir,'LibriSpeech')
        self.librispeech_dir='LibriSpeech'
        if not os.path.exists(self.librispeech_dir):
            os.mkdir(self.librispeech_dir)
        self.lm_dir=os.path.join(self.drive_librispeech_dir,'LibriSpeech_LM_data')
        self.lm_corpus_dir=os.path.join(self.lm_dir,'librispeech-lm-corpus','corpus')
        self.data=self.get_data()
        hp.set_input_dim(self.input_rows,self.input_cols)
        hp.set_vocab_len(len(self.vocab.vocab))
        hp.set_max_len(self.max_len)
        hp.set_target_max_len(self.max_target_len)
        hp.set_standard_out_size(len(self.vocab.vocab)*self.max_target_len)
        hp.set_space_index(self.vocab.vocab[self.vocab.space_token])
        hp.set_blank_index(self.vocab.vocab[self.vocab.blank_token])
    def get_data(self):
        '''
            Gathers the data and builds the vocabulary.
            
            Inputs:
                None
            Outputs:
                data: dict; the data where the keys are the sets names such as train, dev-clean ecc.
        '''
        data={}
        if type(self.hp.standard_train_type)==list:
            data['train']=[]
            for x in self.hp.standard_train_type:
                data['train'].append(LIBRISPEECH(self.librispeech_dir,x,download=not os.path.exists(os.path.join(self.librispeech_dir,x))))
                f=os.path.join(self.librispeech_dir,x+'.tar.gz')
                os.system('rm '+f)
        else:
            data['train']=LIBRISPEECH(self.librispeech_dir,self.hp.standard_train_type,download=not os.path.exists(os.path.join(self.librispeech_dir,self.hp.standard_train_type)))
            f=os.path.join(self.librispeech_dir,self.hp.standard_train_type+'.tar.gz')
            os.system('rm '+f)
        data['dev-clean']=LIBRISPEECH(self.librispeech_dir,'dev-clean',download=not os.path.exists(os.path.join(self.librispeech_dir,'dev-clean')))
        data['dev-other']=LIBRISPEECH(self.librispeech_dir,'dev-other',download=not os.path.exists(os.path.join(self.librispeech_dir,'dev-other')))
        data['test-clean']=LIBRISPEECH(self.librispeech_dir,'test-clean',download=not os.path.exists(os.path.join(self.librispeech_dir,'test-clean')))
        data['test-other']=LIBRISPEECH(self.librispeech_dir,'test-other',download=not os.path.exists(os.path.join(self.librispeech_dir,'test-other')))
        
        to_rm=[os.path.join(self.librispeech_dir,x) for x in os.listdir(self.librispeech_dir) if os.path.isfile(os.path.join(self.librispeech_dir,x))]
        to_rm+=[os.path.join(self.librispeech_dir,'LibriSpeech',x) for x in os.listdir(os.path.join(self.librispeech_dir,'LibriSpeech')) if os.path.isfile(os.path.join(self.librispeech_dir,'LibriSpeech',x))]
        for x in to_rm:
            os.system('rm '+x)
            
            
        if self.hp.wpm:
            self.vocab=WPM(self.hp.base_dir,data['train'],ntokens=self.hp.ntokens,unk_tol=self.hp.unk_tol)
        else:
            self.vocab=myVocab(self.hp.base_dir,data['train'],ntokens=self.hp.ntokens)
        if type(data['train'])==list:
            max_train_len=0
            max_target_train_len=0
            min_train_mels=float('inf')
            max_train_mels=0
            for i in range(len(data['train'])):
                curr_max_train_len,curr_max_target_train_len,data['train'][i]=self.get_max_lens(data,'train',i)
                max_train_len=max(max_train_len,curr_max_train_len)
                max_target_train_len=max(max_target_train_len,curr_max_target_train_len)
        else:            
            max_train_len,max_target_train_len,data['train']=self.get_max_lens(data,'train')
            data['train']=[data['train']]
        max_dev_clean_len,max_target_dev_clean_len,data['dev-clean']=self.get_max_lens(data,'dev-clean')
        max_dev_other_len,max_target_dev_other_len,data['dev-other']=self.get_max_lens(data,'dev-other')
        max_test_clean_len,max_target_test_clean_len,data['test-clean']=self.get_max_lens(data,'test-clean')
        max_test_other_len,max_target_test_other_len,data['test-other']=self.get_max_lens(data,'test-other')
        
        self.max_len=max(max_train_len,max_dev_clean_len,max_dev_other_len,max_test_clean_len,max_test_other_len)
        self.max_len=floor(self.max_len/self.hp.hop_length)+1
        self.max_len+=(3-self.max_len%3)
        print('max len',self.max_len)
        if self.hp.max_target_len is None:
            self.max_target_len=max(max_target_train_len,max_target_dev_clean_len,max_target_dev_other_len,max_target_test_clean_len,max_target_test_other_len)
        else:
            self.max_target_len=self.hp.max_target_len
        self.input_rows=self.hp.n_mels
        self.input_cols=self.max_len
        if type(data['train'])==list:
            start_train_idxes=[len(x) for x in data['train']]
            start_train_idxes=[sum(start_train_idxes[:i+1]) for i in range(len(start_train_idxes))]
            self.start_idxes={'train':start_train_idxes}
            if self.hp.max_target_len is None:
                train_idxes=[[y for y in range(len(x)) if self.vocab.is_tollerable(x[y][2])] for x in data['train']]
            else:
                train_idxes=[self.long_enough(x,self.max_target_len) for x in data['train']]
                train_idxes=[[y for y in x if self.vocab.is_tollerable(data['train'][x][y][2])] for x in train_idxes]
            train_idxes=[[x+start_train_idxes[i-1] for x in train_idxes[i]] if i>0 else train_idxes[i] for i in range(len(start_train_idxes))]
            self.idxes={'train':reduce(lambda x,y: x+y,train_idxes)}
        else:
            if self.hp.max_target_len is None:
                self.idxes={'train':list(range(len(data['train'])))}
            else:
                self.idxes={'train':self.long_enough(data['train'],self.max_target_len)}
            self.idxes['train']=[x for x in self.idxes['train'] if self.vocab.is_tollerable(data['train'][x][2])]
        if self.hp.max_target_len is None:
            self.idxes.update({k:list(range(len(data[k]))) for k in data if 'dev' in k or 'test' in k})
        else:
            self.idxes.update({k:self.long_enough(data[k],self.max_target_len) for k in data if 'dev' in k or 'test' in k})
        return data
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
                self.save_mel(filename,mel)
            mel=np.array(mel)
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
        target=[self.vocab.parse(x[2]) for x in data]
        tau=[len(x[0]) for x in mels]
        target_lens=[len(x) for x in target]
        batch={'input':{'mels':mels,'tau':tau},'target':{'transcripts':target,'lens':target_lens}}
        return batch
                
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
        idxes=self.idxes[dataset_type][i*self.hp.batch_size:i*self.hp.batch_size+self.hp.batch_size]
        batch=[self.get_item(idx,dataset_type) for idx in idxes]
        batch=self.process_data(batch,dataset_type,SpecAugment)
        batch['unpadded_len']=len(batch['input']['mels'])
        batch['input']['mels']=[[self.padding(y,self.max_len-len(y)) for y in x] for x in batch['input']['mels']]
        batch['input']['mels']=self.padding(batch['input']['mels'],self.hp.batch_size-len(batch['input']['mels']))
        batch['input']['tau']=self.padding(batch['input']['tau'],self.hp.batch_size-len(batch['input']['tau']))
        batch['input']['mels']=torch.FloatTensor(np.array(batch['input']['mels'])).to(self.hp.device).unsqueeze(1)
        batch['input']['tau']=torch.LongTensor(np.array(batch['input']['tau'])).to(self.hp.device)
        batch['target']['transcripts']=[self.padding(x,self.max_target_len-len(x)) for x in batch['target']['transcripts']]
        batch['target']['transcripts']=self.padding(batch['target']['transcripts'],self.hp.batch_size-len(batch['target']['transcripts']))
        batch['target']['transcripts']=torch.LongTensor(np.array(batch['target']['transcripts'])).to(self.hp.device)
        batch['target']['lens']=self.padding(batch['target']['lens'],self.hp.batch_size-len(batch['target']['lens']))
        batch['target']['lens']=torch.LongTensor(np.array(batch['target']['lens'])).to(self.hp.device)
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
    def mix_datasets(self,U,targets):
        '''
            Mix the training dataset with a dataset where the labels are generated by a model
            
            Inputs:
                U: the dataset with no labels (in this implementation the ground truth exists already but is ignored to mimic the situation in which they don't)
                targets: list[str]; the generated labels
            Outputs:
                None
        '''
        l=min(len(U),len(targets))
        u=[[U.data[i][0],U.data[i][1],targets[i]] for i in range(l) if len(self.vocab.parse(targets[i]))<=self.max_target_len]
        self.data['mix']=self.data['train'].copy()+u
        self.idxes['mix']=self.idxes['train']+list(range(len(u)))
        self.start_idxes['mix']=self.start_idxes['train']+[sum(self.start_idxes['train'])+len(u)]
        shuffle(self.data['mix'])
                            
                    
                
    
        
        
                    
                
                

                
        