import json
import librosa
import os
from tqdm import tqdm
from colorama import Fore
from random import shuffle
import numpy as np
import torch
from torch.utils.data import Dataset
from math import floor
import tensorflow_datasets as tfds
import tensorflow as tf
import copy

class SpokenDigits(Dataset):
    '''
        class SpokenDigits
        
        Handles the spokendigit dataset from tensorflow
        
        Inputs:
            hp: HParams; the class of hyperparameters
        Outputs:
            None
    '''
    def __init__(self,hp):
        super(SpokenDigits,self).__init__()
        self.hp=hp
        self.spokendigits_dir='spokendigits'
        if not os.path.exists(self.spokendigits_dir):
            os.mkdir(self.spokendigits_dir)
        self.data=tfds.load('spoken_digit')
        if 'train' not in self.data:
            self.data=tfds.load('spoken_digit')
        self.data=self.data['train']
        self.data=list(self.data)
        shuffle(self.data)
        self.data=self.process_data(self.data,progress_bar=True)
        self.idxes=list(range(len(self.data['mels'])))
        if hp.max_len is None:
            self.max_len=max([len(x) for x in self.data['mels']])
            self.input_cols=self.max_len
        else:
            self.filter_data(hp.max_len)
        self.input_rows=self.hp.n_mels
    def get_mels(self,data,progress_bar=False):
        '''
            Computation of the log-mel spectograms of the audios
            
            Inputs: 
                data: list; the input data
                progress_bar: bool; whether a progress bar is to be printed. Default: False
            Outputs:
                mels: list[]; the log-mel spectograms of the audio from the input dataset
        '''
        mels=[]
        d=os.path.join(self.spokendigits_dir,'mels')
        if not os.path.exists(d):
            os.mkdir(d)
        t=data
        if progress_bar:
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            t=tqdm(t,unit='audio',dynamic_ncols=True,bar_format=bar_format,desc='spokendigits mels')
        for i,elem in enumerate(t):
            filename=os.path.join(d,'mel_'+str(elem['audio/filename'])+'.csv')
            elem=elem['audio']
            if self.hp.read_mels and os.path.exists(filename):
                mel=self.read_mel(filename)
            else:
                if not isinstance(elem,np.ndarray):
                    elem=tf.squeeze(elem).numpy()
                    elem=elem.astype(float)
                mel=librosa.feature.melspectrogram(y=elem,sr=8,n_fft=512,n_mels=self.hp.n_mels)
                mel=np.where(mel<1e-10,0,np.log(mel))
                self.save_mel(filename,mel)
            mels.append(mel)
            if progress_bar and i==len(data)-1:
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        t.close()
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
    def process_data(self,data,progress_bar=False):
        '''
            This function process the data. The log-mel spectograms of the audios are computed.
            Inputs:
                data: list; data before processing. The list must follow the spokendigit tensorflow dataset
            Outputs:
                data: list; data after preprocessing
        '''
        mels=self.get_mels(data,progress_bar=progress_bar)
        min_value=min([min(y) for x in mels for y in x])
        mels=[[[x-min_value for x in y] for y in z] for z in mels]
        max_value=max([max(y) for x in mels for y in x])
        mels=[[[x/max_value for x in y] for y in z] for z in mels]
        tau=[len(x[0]) for x in mels]
        return {'mels':mels,'tau':tau}
    def shuffle(self):
        '''
            Shuffles the dataset 
            
            Inputs:
                None
            Outputs:
                None
        '''
        shuffle(self.idxes)
    def __len__(self):
        '''
            Returns the length of the dataset
            
            Inputs:
                None
            Outputs:
                length of the dataset; int
        '''
        return len(self.idxes)
    def filter_data(self,max_len):
        '''
            Filters out very long mels. This can be used in case of poor hardware.
            
            Inputs:
                max_len: int; maximum mel length
            Outputs:
                None
            
        '''
        self.max_len=max_len
        self.input_cols=self.max_len
        self.idxes=[x for x in self.idxes if len(self.data['mels'][x])<=max_len]
    def __getitem__(self,i):
        '''
            Returns the ith batch
            
            Inputs:
                i: int; batch index
            Outputs:
                batch: dict; the requested batch
        '''
        batch=self.idxes[i*self.hp.batch_size:i*self.hp.batch_size+self.hp.batch_size]
        batch={k:[self.data[k][j] for j in batch] for k in self.data}
        batch['mels']=[[self.padding(y,self.max_len-len(y)) for y in x] for x in batch['mels']]
        batch['mels']=self.padding(batch['mels'],self.hp.batch_size-len(batch['mels']))
        batch['mels']=torch.FloatTensor(np.array(batch['mels'])).to(self.hp.device).unsqueeze(1)
        batch['tau']=self.padding(batch['tau'],self.hp.batch_size-len(batch['tau']))
        batch['tau']=torch.FloatTensor(np.array(batch['tau'])).to(self.hp.device)
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
        
        
                    
                
                

                
        