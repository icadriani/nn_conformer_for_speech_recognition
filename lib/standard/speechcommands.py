import json
import librosa
import os
from tqdm import tqdm
from colorama import Fore
from random import shuffle, randint, choice, seed
import numpy as np
import torch
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset
from lib.standard.myvocab import myVocab
from lib.standard.wordpiecemodel import WPM
from math import floor
from functools import reduce
from itertools import product
import copy
# from torchtext.transforms import SpmTokenizerTransform
np.seterr(divide = 'ignore') 
seed(42)

class SpeechCommands(Dataset):
    '''
        class SpeechCommands
        
        This classs handles the ononymous dataset from torch
        
        Inputs:
            hp: HParams; class of hyperparameters
    '''
    def __init__(self,hp):
        super(SpeechCommands,self).__init__()
        self.hp=hp
        self.speechcommands_dir='SpeechCommands'
        # self.max_value=0
        # self.min_value=inf
        if not os.path.exists(self.speechcommands_dir):
            os.makedirs(self.speechcommands_dir)
        self.data=self.get_data()
        self.data=self.process_data(self.data)
        self.input_rows=self.hp.n_mels
        self.input_cols=self.max_len
        hp.set_input_dim(self.input_rows,self.input_cols)
        hp.set_vocab_len(len(self.vocab.vocab))
        # print(speech.max_len,speech.max_og_len)
        hp.set_max_len(self.max_len)
        hp.set_target_max_len(self.max_target_len)
        hp.set_standard_out_size(len(self.vocab.vocab)*self.max_target_len)
        hp.set_space_index(self.vocab.vocab[self.vocab.space_token])
        hp.set_blank_index(self.vocab.vocab[self.vocab.blank_token])  
        hp.set_ntokens(max(hp.ntokens,len(self.vocab.vocab)))
        # hp.set_max_value(self.max_value)
    def get_data(self):
        '''
            This function gathers the data from the torch dataset. Tra training data is split for creating another dataset that can be used for pretraining and/or finetuning
        '''
        data={k:SPEECHCOMMANDS(self.speechcommands_dir,subset=k if k[0]!='t' else k+'ing',download=not os.path.exists(os.path.join(self.speechcommands_dir,k))) for k in ['train','validation','test']}
        # speakers=np.unique([x[3] for x in data['train']],return_counts=True)
        # print(len(speakers[0]),speakers)
        # if self.hp.do_pretraining or self.hp.nst or self.hp.just_nst:
        for k in data:
            data[k]=[list(x) for x in data[k]]
            for i in range(len(data[k])):
                data[k][i][0]=data[k][i][0].squeeze().numpy()
        for k in data:
            shuffle(data[k])
            # data[k]=data[k][:1000]
        data=self.split_pretrain(data)
        return data
        # data={k:self.add_augmentations(data[k],self.hp.n_augs[k]) for k in data}
        # for k in data:
        #     words,counts=np.unique([x[2] for x in data[k]],return_counts=True)
        #     # min_count=np.min(counts)
        #     # keep={w:[] for w in words}
        #     # for x in data[k]:
        #     #     if len(keep[x[2]])<min_count:
        #     #         keep[x[2]].append(x)
        #     # data[k]=reduce(lambda x,y: x+y,list(keep.values()))
        #     # shuffle(data[k])
        #     left=np.max(counts)-np.array(counts)
        #     # left={w:l for w,l in zip(words,left)}
        #     for w,l in zip(words,left):
        #         data[k]=self.add_augmentations(data[k],l,w)
        # return data
    def process_data(self,data):
        '''
            This function process the data. The log-mel spectograms of the audios are computed as well as the encoding of the labels.
            Inputs:
                data: dict; data before processing. The keys of the dictornary correspond to the set name as in train, validation, test ecc. The data in each dataset must follow the SPEECHCOMMANDS structure.
            Outputs:
                data: dict; data after preprocessing
        '''
        mels={k:self.get_mels(data,k,progress_bar=True) for k in data}
        # print(data)
        self.vocab=myVocab(self.hp.base_dir,data['train'],ntokens=self.hp.ntokens)
        labels={k:[self.vocab.parse(x[2]) for x in v] for k,v in data.items()}
        tau={k:[len(x) for x in v] for k,v in mels.items()}
        target_lens={k:[len(x) for x in v] for k,v in labels.items()}
        # print(len(mels['train']),len(tau['train']),len(labels['train']),len(target_lens['train']))
        data={k:{'input':{'mels':mels[k],'tau':tau[k]},'target':{'transcripts':labels[k],'lens':target_lens[k]}} for k in data}
        # data={k:self.add_augmentations(data[k],self.hp.n_augs[k]) for k in data}
        data={k:[{k1:{k2:data[k][k1][k2][i] for k2 in data[k][k1]} for k1 in data[k]} for i in range(len(data[k]['input']['mels']))] for k in data}
        # print(len(data['train']))
        self.max_len=max([max(tau[k]) for k in tau])
        # self.max_len+=(3-self.max_len%3)
        self.max_target_len=max([max(target_lens[k]) for k in target_lens])
        self.idxes={k:list(range(len(data[k]))) for k in data}
        return data
    def get_mels(self,data,dataset_type='train',progress_bar=False):
        '''
            Computation of the log-mel spectograms of the audios
            
            Inputs: 
                data: dict or list or SPEECHCOMMANDS; If a dict is given the right dataset_type has to be given.
                dataset_type: str; train, validation, ecc. Default: 'train'
                progress_bar: bool; whether a progress bar is to be printed. Default: False
            Outputs:
                mels: list[]; the log-mel spectograms of the audio from the input dataset
        '''
        if isinstance(data,dict):
            data=data[dataset_type]
        mels=[]
        d=os.path.join(self.speechcommands_dir,'mels')
        if not os.path.exists(d):
            os.mkdir(d)
        d=os.path.join(d,dataset_type)
        if not os.path.exists(d):
            os.mkdir(d)
        t=range(len(data))
        if progress_bar:
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            t=tqdm(t,unit='audio',dynamic_ncols=True,bar_format=bar_format,desc=dataset_type.capitalize())
        for i in t:
            elem=list(data[i])
            filename=os.path.join(d,'mel_'+'_'.join([str(x) for x in elem[-3:]])+'.csv')
            mel=None
            if self.hp.read_mels and os.path.exists(filename):
                mel=self.read_mel(filename)
            if mel is None or len(mel)!=self.hp.n_mels:
                self.hp.read_mels=False
                if torch.is_tensor(elem[0]):
                    elem[0]=elem[0].squeeze().numpy()
                mel=librosa.feature.melspectrogram(y=elem[0],sr=elem[1],n_mels=self.hp.n_mels)
                # mel = librosa.power_to_db(mel, ref=np.log)
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
            mel-=np.min(mel)
            curr_max=np.max(mel)
            # self.min_value=min(self.min_value,np.min(mel))
            # self.max_value=max(self.max_value,curr_max)
            mel/=curr_max
            # print(mel.shape)
            # print(type(mel),mel.shape)
            mels.append(mel)
            if progress_bar and i==len(data)-1:
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        t.close()
        # mels=[mel/self.max_value for mel in mels]
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
        mel='\n'.join([','.join([str(x) for x in m]) for m in mel])
        with open(filename,'w+') as f:
            f.write(mel)
    def shuffle(self,dataset_type='train'):
        '''
            Shuffles the dataset
            
            Inputs:
                dataset_type: str; type of dataset: train, mix or ecc. Defaults: 'train' 
            Outputs:
                None
        '''
        shuffle(self.data[dataset_type])
    def get_item(self,i,dataset_type='train'):
        '''
            Retrives the element of the dataset present at index i
            
            Inputs:
                i: int; index of the element to retreive
                dataset_type: str; the dataset as in train, validation, ecc. Default: 'train'
            Outputs:
                dataset element at index i; dict
        '''
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
        batch=self.data[dataset_type][i*self.hp.batch_size:i*self.hp.batch_size+self.hp.batch_size]
        # print(batch)
        batch={k1:{k2:[batch[i][k1][k2] for i in range(len(batch))] for k2 in batch[0][k1]} for k1 in batch[0]}
        # batch=[self.get_item(idx,dataset_type) for idx in idxes]
        # else:
        #     start=i*self.hp.batch_size
        #     end=min(len(self.data[dataset_type]),start+self.hp.batch_size)
        #     # print(start,start+self.hp.batch_size,len(self.data[dataset_type]),end)
        #     batch=[self.data[dataset_type][i] for i in range(start,end)]
        # print(batch)
        # batch=self.process_data(batch,dataset_type,SpecAugment)
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
        batch['input']['mels']=torch.FloatTensor(np.array(batch['input']['mels'])).to(self.hp.device)#.unsqueeze(1)
        # batch['input']['mels']=batch['input']['mels']*self.max_value
        # batch['input']['mels']=batch['input']['mels'].round().int()
        # mask=batch['input']['mels'].lt(0)
        # batch['input']['mels']=torch.masked_fill(batch['input']['mels'],mask,value=0.0)
        # if i==535:
        #     print('--------------------')
        batch['input']['tau']=torch.LongTensor(np.array(batch['input']['tau'])).to(self.hp.device)
        # batch['target']['transcripts']=[self.padding(x,self.max_target_len-len(x)) for x in batch['target']['transcripts']]
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
        if type(l)!=list: l=l.tolist()
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
        # print(U[0])
        # print(len(U[0]))
        # print(targets[0])
        # print(len(U),len(targets))
        # for t in targets:
        #     print(t)
        targets=[self.vocab.parse(target) if len(target)>0 else [0] for target in targets]
        l=min(len(U.data['pretrain']),len(targets))
        # print(len(U))
        # print(U.data[0][0].shape)
        # print(targets)
        # data={k:[{'input':{'mels':mels[k][i],'tau':tau[k][i]},'target':{'transcripts':labels[k][i],'lens':target_lens[k][i]}} for i in range(len(labels[k]))] for k in data}
        # u=[{'input':{'mels':U.data['mels'][i],'tau':U.data['tau'][i]},'target':{'transcripts':targets[i],'lens':len(targets[i])}} for i in range(l) if len(targets[i])<=self.max_target_len]
        u=[{'input':{'mels':U.data['pretrain'][i]['input']['mels'],'tau':U.data['pretrain'][i]['input']['tau']},'target':{'transcripts':targets[i],'lens':len(targets[i])}} for i in range(l) if len(targets[i])<=self.max_target_len]
        # print(len(u))
        # for x in u: print(x)
        # for t in targets:
        #     print(t)
        self.data['mix']=copy.deepcopy(self.data['train'])+u
        self.idxes['mix']=self.idxes['train']+list(range(len(u)))
        # self.start_idxes['mix']=self.start_idxes['train']+[sum(self.start_idxes['train'])+len(u)]
        shuffle(self.data['mix'])
    # def split_pretrain(self,)
    def add_augmentations(self,data,n_augs,word=None):
        '''
            Auguments the dataset by adding samples with audio that are augmented by gaussian noise
            
            Inputs:
                data: list; the dataset to be augmented (ex. self.dataset['train'])
                n_augs: int; the number of augmented samples to be added
                word: str; the samples the be drawn are drawn randomly from the dataset if word is None. Otherwise only samples that correspond to label word are consider.
            Outputs:
                data_w_augs: list; the input dataset concatanated with the augmented samples
                
        '''
        # idx=list(range(len(mels)))
        if n_augs==0:
            return data
        # mels=data['input']['mels']
        if word is None:
            to_aug_idx=[randint(0,len(data)-1) for _ in range(n_augs)]
        else:
            data_w_word=[i for i in range(len(data)) if data[i][2].strip().lower()==word.strip().lower()]
            to_aug_idx=[choice(data_w_word) for _ in range(n_augs)]
        # shape=[n_augs]+list(np.array(mels).shape)[1:]
        # print(shape)
        # print(len(mels),min(to_aug_idx),max(to_aug_idx))
        normal=[np.random.normal(loc=0,scale=0.25*(np.max(data[i][0])-np.min(data[i][0])),size=data[i][0].shape) for i in to_aug_idx]
        # print(word,np.max([np.max(x[0]) for x in data]),[np.max(x[0]) for x in data])
        audioaugs=[[data[j][0]+normal[i]]+data[j][1:] for i,j in enumerate(to_aug_idx)]
        # augs={k:{k1:[data[k][k1][i] for i in to_aug_idx] if k1!='mels' else melaugs for k1 in data[k]} for k in data}
        # data_w_augs={k:{k1:list(data[k][k1])+list(augs[k][k1]) for k1 in data[k]} for k in data}
        data_w_augs=data+audioaugs
        idx=list(range(len(data_w_augs)))
        shuffle(idx)
        # data_w_augs={k:{k1:[data_w_augs[k][k1][i] for i in idx] for k1 in data[k]} for k in data}
        data_w_augs=[data_w_augs[i] for i in idx]
        # print(data_w_augs[0])
        # for i in range(len(data_w_augs)):
        #     data_w_augs[i][0]-=torch.min(data_w_augs[i][0])
        #     data_w_augs[i][0]/=torch.max(data_w_augs[i][0])
        # data_w_augs['input']['mels']=[x-np.min(x) for x in data_w_augs['input']['mels']]
        # data_w_augs['input']['mels']=[x/np.max(x) for x in data_w_augs['input']['mels']]
        # print(len(mels),data_w_augs['input']['mels'])
        return data_w_augs
    def split_pretrain(self,data):
        '''
            Splits the train dataset into training and pretrain. The latter can be used for pretraing and/or finetuning
            
            Inputs:
                data: dict; the input data divided into the train, validation, ecc. datasets
            Outputs:
                data: dict; the input data divided into the train, validation, ecc. datasets where the train dataset its split into train and pretrain
        '''
        speakers=np.unique([x[3] for x in data['train']])
        pretrain_size=round(0.25*len(speakers))
        # shuffle(speakers)
        pretrain=speakers[:pretrain_size]
        # train=speakers[pretrain_size:]
        # print(pretrain_size)
        # print(data.keys(),data['train'].keys())
        # print({{type(data['train'][k][k1]) for k1 in data['train'][k]} for k in data['train']})
        # print({{len(data['train'][k][k1]) for k1 in data['train'][k]} for k in data['train']})
        train=copy.deepcopy(data['train'])
        data['pretrain']=[x for x in train if x[3] in pretrain]
        data['train']=[x for x in train if x[3] not in pretrain]
        return data