
import os
from torch.utils.data import Dataset
from string import punctuation
import re
from colorama import Fore
from tqdm import tqdm
from torchaudio.datasets import LIBRISPEECH
from lib.standard.myvocab import myVocab
from random import shuffle
import numpy as np
from lib.finetuning.lmvocab import LMVocab
import torch
from functools import reduce

class LibriSpeechLM(Dataset):
    '''
        class LibriSpeechLM
        
        This classs handles the LibriSpeech Language model
        
        Inputs:
            hp: HParams; class of hyperparameters
    '''
    def __init__(self,hp):
        self.hp=hp
        self.librispeech_dir=os.path.join(hp.data_dir,'LibriSpeech')
        self.lm_dir=os.path.join(self.librispeech_dir,'LibriSpeech_LM_data')
        self.lm_corpus_dir=os.path.join(self.lm_dir,'librispeech-lm-corpus','corpus')
        self.vocab=LMVocab(hp)
        corpus,pronuncia_corpus=self.get_corpus()
        transcripts=self.get_transcripts()
        pronuncia_transcripts={k:self.vocab.get_pronuncie_data(transcripts[k]) for k in transcripts}
        data=corpus+transcripts['train']
        pronuncia_data=pronuncia_corpus+pronuncia_transcripts['train']
        self.vocab.get_words_vocab(data)
        self.vocab.get_pronuncie_vocab(pronuncia_data)
        self.data={}
        self.data['train']=[[pronuncia_data[i],data[i]] for i in range(len(data))]
        self.data['dev-clean']=[[pronuncia_transcripts['dev-clean'][i],transcripts['dev-clean'][i]] for i in range(len(transcripts['dev-clean']))]
        self.data['dev-other']=[[pronuncia_transcripts['dev-other'][i],transcripts['dev-clean'][i]] for i in range(len(transcripts['dev-other']))]
        self.data={k:[v for v in vv if len(self.vocab.pronuncie_vocab.parse(v[0]))<=hp.lm_max_len and len(self.vocab.words_vocab.parse(v[1]))<=hp.lm_max_len] for k,vv in self.data.items()}
    def shuffle(self):
        '''
            Shuffles the dataset
            
            Inputs:
                None
            Outputs:
                None
        '''
        shuffle(self.data['train'])
    def get_batch(self,i,dataset_type='train'):
        '''
            Retrives the ith batch
            
            Inputs:
                i: int; index of the batch to be retrived
                dataset_type: The dataset from which the dataset is be retreived (train, validation, ecc). Default: 'train'
            Outputs:
                batch: dict; The retrevied batch
            
        '''
        batch=self.data[dataset_type][i*self.hp.batch_size:i*self.hp.batch_size+self.hp.batch_size].copy()
        batch={'input':[x[0] for x in batch],'target':[x[1] for x in batch]}
        batch['input']=[self.vocab.pronuncie_vocab.parse(x) for x in batch['input']]
        batch['target']=[self.vocab.words_vocab.parse(x) for x in batch['target']]
        batch['input']=[self.padding(x,self.hp.lm_max_len-len(x)) for x in batch['input']]
        batch['target']=[self.padding(x,self.hp.lm_max_len-len(x)) for x in batch['target']]
        batch['input']=self.padding(batch['input'],self.hp.batch_size-len(batch['input']))
        batch['target']=self.padding(batch['target'],self.hp.batch_size-len(batch['target']))
        batch['input']=[self.vocab.pronuncie_vocab.one_hot_encoding(x) for x in batch['input']]
        batch['target']=[self.vocab.words_vocab.one_hot_encoding(x) for x in batch['target']]
        batch['input']=torch.LongTensor(np.array(batch['input'])).to(self.hp.device)
        batch['input']=batch['input'].flatten(0,1)
        batch['target']=torch.LongTensor(np.array(batch['target'])).to(self.hp.device)
        batch['target']=batch['target'].flatten(0,1)
        return batch
    def get_transcripts(self):
        '''
            Retrives the audio transcripts from the LibriSpeech dataset
            
            Inputs:
                None
            Outputs:
                transcripts: list[str]; the audio transcripts from the LibriSpeech dataset
        '''
        transcripts={}
        transcripts['train']=[x[2] for x in LIBRISPEECH(self.librispeech_dir,self.hp.standard_train_type,download=not os.path.exists(os.path.join(self.librispeech_dir,self.hp.standard_train_type)))]
        transcripts['dev-clean']=[x[2] for x in LIBRISPEECH(self.librispeech_dir,'dev-clean',download=not os.path.exists(os.path.join(self.librispeech_dir,'dev-clean')))]
        transcripts['dev-other']=[x[2] for x in LIBRISPEECH(self.librispeech_dir,'dev-other',download=not os.path.exists(os.path.join(self.librispeech_dir,'dev-other')))]
        return transcripts
    def get_corpus(self):
        '''
            Retrieves the corpus of LibriSpeech for language modeling which is divided into transcripts of books and their pronunciations
            
            Inputs:
                None
            Outputs:
                books: list[str]; book corpus
                pronuncie: list[str]; pronunciation courpus
        '''
        book_files=[os.path.join(self.lm_corpus_dir,x,x+'.txt') for x in os.listdir(self.lm_corpus_dir)]
        book_files=book_files[:len(book_files)//2]
        books=[]
        pronuncie=[]
        alpha=re.compile("^[A-Z' ]+$")
        roman_num=re.compile('^[IVXLCDM]+$')
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        t=tqdm(range(len(book_files)),unit='book',dynamic_ncols=True,bar_format=bar_format,desc='LibriSpeech LM corpus')
        for bf in t:
            with open(book_files[bf],'r') as f:
                b=f.read().strip().split('\n')
                b=self.clean_book(b,roman_num,alpha)
                books+=b
                pronuncia_b_file=book_files[bf].replace('.txt','_pronuncia.txt')
                pronuncia_b=self.vocab.get_pronuncie_data(b)
                pronuncia_b_='\n'.join(pronuncia_b)
                with open(pronuncia_b_file,'w+') as f:
                        f.write(pronuncia_b_)
            pronuncie+=pronuncia_b
            if bf==len(book_files)-1:
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        return books,pronuncie
    def clean_book(self,book,roman_num,alpha):
        '''
            Cleans the given book excuting different tasks such as filtering out special characters and empty lines, it strips lines.
            
            Inputs:
                book: list[str]; the book to clean split into lines
                roman_num: compiled re; regular expression for roman numbers
                alpha: compiled re; regular expression for letters
            Outputs:
                book: list[str]; cleaned book
        '''
        book=[line.strip() for line in book if line.strip()!='']
        book=[line.upper() for line in book]
        book=[line for line in book if '[' not in line]
        book=[''.join([c if c not in punctuation or c=="'" else ' ' for c in line]) for line in book]
        book=[' '.join([word for word in line.split() if len(word)>0]) for line in book]
        book=[line for line in book if len([True for word in line.split() if roman_num.match(word.strip())])==0]
        book=[line for line in book if alpha.match(line)]
        book=[' '.join(line.split()[:self.hp.lm_max_len]) for line in book]
        return book
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
        
        