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
        # self.data_dir=data_dir
        self.hp=hp
        # self.spokendigits_data_dir=os.path.join(hp.data_dir,'spokendigits')
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
        # self.data=[{k:self.data[k][i] for k in self.data} for i in range(len(self.data['mels']))]
        # self.small_dir=os.path.join(self.libriligh_data_dir,'small_32_secs')
        # self.data=self.get_data_from(self.small_dir)
        # self.data=spokendigitsLimited(self.spokendigits_dir,hp.spokendigits_subset,True)
        # self.min_mel_value,self.max_mel_value=self.get_min_max_mel_values(self.data)
        # print(len(self.data[0][0]))
        self.idxes=list(range(len(self.data['mels'])))
        if hp.max_len is None:
            self.max_len=max([len(x) for x in self.data['mels']])
            self.input_cols=self.max_len
        else:
            # self.max_len=hp.max_len
            self.filter_data(hp.max_len)
        # self.max_len+=(3-self.max_len%3)
        self.input_rows=self.hp.n_mels
        # self.make_dataset(self.small)
        # print(self.max_len)
    # def get_data_from(self,dir_name):
    #     data=[]
    #     speaker_ids=os.listdir(dir_name)
    #     # speaker_ids=speaker_ids[:10]
    #     dataset_name=dir_name.split('/')
    #     dataset_name=dataset_name[-2]+' '+dataset_name[-1]#[0].upper()+dataset_name[-1][1:]
    #     dataset_name=dataset_name[:dataset_name.find('_')]
    #     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
    #     t=tqdm(range(len(speaker_ids)),unit='speaker',dynamic_ncols=True,bar_format=bar_format,desc=dataset_name+' dataset')
    #     for speaker in t:
    #         speaker_dir=os.path.join(dir_name,speaker_ids[speaker])
    #         books=os.listdir(speaker_dir)
    #         # data[speaker]={}
    #         for book in books:
    #             book_dir=os.path.join(speaker_dir,book)
    #             chapters_files=os.listdir(book_dir)
    #             #print(self.small_dir)
    #             #json_file=[x for x in chapters_files if '.json' in x]
    #             #json_file=[os.path.join(book_dir,x) for x in json_file]
    #             # print(json_file)
    #             # os
    #             # return
    #             flac_files=[x for x in chapters_files if '.flac' in x]
    #             # json_file=os.path.join(book_dir,json_file)
    #             # with open(json_file,'r') as f:
    #             #     json_file=json.loads(f.read())
    #             flac_files.sort()
    #             # flacs=[]
    #             #srs=[]
    #             for x in flac_files:
    #                 x=os.path.join(book_dir,x)
    #                 flac,sr=librosa.load(x)
    #                 data.append([flac,sr])
    #                 #srs.append(sr)
    #             # mels=[librosa.feature.melspectrogram(x[0],sr=x[1],n_fft=self.hp.n_fft,hop_length=self.hp.hop_length,n_mels=self.hp.n_mels) for x in flacs]
    #             # data[speaker][book]=mels
    #         if speaker==len(speaker_ids)-1:
    #             t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
    #     # self.input_rows=self.hp.n_mels
    #     # self.input_cols=self.max_len
    #     # self.data=data
    #     return data
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
        # d='spokendigits'
        # if not os.path.exists(d):
        #     os.mkdir(d)
        d=os.path.join(self.spokendigits_dir,'mels')
        if not os.path.exists(d):
            os.mkdir(d)
        t=data
        if progress_bar:
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            t=tqdm(t,unit='audio',dynamic_ncols=True,bar_format=bar_format,desc='spokendigits mels')
        for i,elem in enumerate(t):
            # elem=list(data[i])
            # print(elem['audio'])
            filename=os.path.join(d,'mel_'+str(elem['audio/filename'])+'.csv')
            elem=elem['audio']
            if self.hp.read_mels and os.path.exists(filename):
                mel=self.read_mel(filename)
            else:
                # print(type(elem[0]))
                if not isinstance(elem,np.ndarray):
                    elem=tf.squeeze(elem).numpy()
                    elem=elem.astype(float)
                mel=librosa.feature.melspectrogram(y=elem,sr=8,n_fft=512,n_mels=self.hp.n_mels)
                mel=np.where(mel<1e-10,0,np.log(mel))
                # mel=mel.tolist()
                # mel=librosa.power_to_db(mel)
                # print(elem[0].shape,mel.shape,elem[1],elem[0].shape[1]/self.hp.hop_length)
                # print(type(mel),mel.shape)
                # if dataset_type in self.idxes:
                # else:
                #     filename=os.path.join(d,'mel_'+str(i)+'.csv')
                self.save_mel(filename,mel)
            # print(type(mel),mel.shape)
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
    # def get_min_max_mel_values(self,data):
    #     mels=self.get_mels(data,progress_bar=True)
    #     # max_len=max([len(x[0]) for x in mels])
    #     min_value=min([min(y) for x in mels for y in x])
    #     max_value=max([max(y) for x in mels for y in x])
    #     return min_value,max_value
    def process_data(self,data,progress_bar=False):
        '''
            This function process the data. The log-mel spectograms of the audios are computed.
            Inputs:
                data: list; data before processing. The list must follow the spokendigit tensorflow dataset
            Outputs:
                data: list; data after preprocessing
        '''
        mels=self.get_mels(data,progress_bar=progress_bar)
        # print(mels)
        # max_len=max([x.shape[-1] for x in mels])
        min_value=min([min(y) for x in mels for y in x])
        mels=[[[x-min_value for x in y] for y in z] for z in mels]
        max_value=max([max(y) for x in mels for y in x])
        mels=[[[x/max_value for x in y] for y in z] for z in mels]
        # max_len=max([len(x[0]) for x in mels])
        # min_value=min([min(y) for x in mels for y in x])
        # max_value=max([max(y) for x in mels for y in x])
        # mels=[[[((x-self.min_mel_value)/(self.max_mel_value-self.min_mel_value))*2-1 for x in y] for y in z] for z in mels]
        # input_rows=len(data[0])
        # input_cols=max_len
        # mels=[mels[i:i+self.hp.batch_size] for i in range(0,len(mels),self.hp.batch_siz
        # target=[]
        # for x in data:
        #     target+=self.vocab.one_hot_sentence(x[2])
        # target=[self.vocab.parse(x[2]) for x in data]
        # d=[self.vocab.decode_one_hot(x) for x in target]
        # print(d)
        # target=[self.one_hot_encoding(x) for x in target]
        # target=[target[i:i+self.hp.batch_size] for i in range(0,len(target),self.hp.batch_size)]
        # print(target[0])
        # tau=[len(x[0]) for x in mels]# for y in x]
        # if SpecAugment:
        #     batch={'input':{'mels':mels,'tau':tau},'target':target}# for i in range(mels)]
        # else:
        #     batch={'input':mels,'target':target}# for i in range(mels)]
        # print(batch[0])
        # print(mels)
        tau=[len(x[0]) for x in mels]
        return {'mels':mels,'tau':tau}
    # def make_dataset(self,data):
    #     self.data=[x.tolist() for s in data for b in data[s] for x in data[s][b]]
    #     self.max_len=max([len(y) for x in self.data for y in x])
    #     min_value=min([min(y) for x in self.data for y in x])
    #     max_value=max([max(y) for x in self.data for y in x])
    #     #print(min_value,max_value)
    #     self.data=[[[((x-min_value)/max_value)*2-1 for x in y] for y in z] for z in self.data]
    #     #min_value=min([min(y) for x in self.data for y in x])
    #     #max_value=max([max(y) for x in self.data for y in x])
    #     #print(min_value,max_value)
    #     #self.data=[[self.padding(y,max_len-len(y)) for y in x] for x in self.data]
    #     self.input_rows=len(self.data[0])
    #     self.input_cols=self.max_len
    #     #for k in self.data:
    #     self.data=[self.data[i:i+self.hp.batch_size] for i in range(0,len(self.data),self.hp.batch_size)]
    #     #self.divide()
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
    # def divide(self):
    #     self.train_size=round(0.7*len(self.data))
    #     self.val_size=round(0.2*len(self.data))
    #     self.test_size=len(self.data)-self.train_size-self.val_size
    #     train_set=self.data[:self.train_size]
    #     val_set=self.data[self.train_size:self.train_size+self.val_size]
    #     test_set=self.data[self.train_size+self.val_size:]
    #     self.data={'train':train_set,'val':val_set,'test':test_set}
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
        # print(self.data[0])
        # print(np.array(self.data[0][0]).shape)
        self.idxes=[x for x in self.idxes if len(self.data['mels'][x])<=max_len]
        # print(self.data)
    def __getitem__(self,i):#,return_tau=False):
        '''
            Returns the ith batch
            
            Inputs:
                i: int; batch index
            Outputs:
                batch: dict; the requested batch
        '''
        batch=self.idxes[i*self.hp.batch_size:i*self.hp.batch_size+self.hp.batch_size]
        batch={k:[self.data[k][j] for j in batch] for k in self.data}
        # batch=self.process_data(batch)
        # if return_tau:
        #     tau=[len(x[0]) for x in batch]
        #     tau=self.padding(tau,self.hp.batch_size-len(tau))
        batch['mels']=[[self.padding(y,self.max_len-len(y)) for y in x] for x in batch['mels']]
        batch['mels']=self.padding(batch['mels'],self.hp.batch_size-len(batch['mels']))
        batch['mels']=torch.FloatTensor(np.array(batch['mels'])).to(self.hp.device).unsqueeze(1)
        batch['tau']=self.padding(batch['tau'],self.hp.batch_size-len(batch['tau']))
        batch['tau']=torch.FloatTensor(np.array(batch['tau'])).to(self.hp.device)#.unsqueeze(1)
        # if return_tau:
        #     batch={'mels':batch,'tau':tau}
        return batch
    # def padding(self,l,left):
    #     if left==0:
    #         return l
    #     shape=np.array(l).shape[1:]
    #     pad=self.pad(0,shape)
    #     pad=[pad]*left
    #     return l+pad
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
    # def pad(self,pad,shape):
    #     if len(shape)==0:
    #         return pad
    #     return self.pad([pad]*shape[-1],shape[:-1])
        
        
                    
                
                

                
        