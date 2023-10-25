
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
        self.vocab=LMVocab(hp)#,None,pronuncia_corpus)#,self.data)
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
        # print(len(self.data['dev-clean']))
        # train_max_in_len=max([len(x[0].split()) for x in self.data['train']])
        # train_max_out_len=max([len(x[1].split()) for x in self.data['train']])
        # # dev_clean_max_in_len=max([len(x[0].split()) for x in self.data['dev-clean']])
        # # dev_clean_max_out_len=max([len(x[1].split()) for x in self.data['dev-clean']])
        # # dev_other_max_in_len=max([len(x[0].split()) for x in self.data['dev-other']])
        # # dev_other_max_out_len=max([len(x[1].split()) for x in self.data['dev-other']])
        # max_in_len=train_max_in_len#max(train_max_in_len,dev_clean_max_in_len,dev_other_max_in_len)
        # max_out_len=train_max_out_len#max(train_max_out_len,dev_clean_max_out_len,dev_other_maxoutn_len)
        # self.max_len=max(max_in_len,max_out_len)
        # self.max_len=hp.lm_max_len
        self.data={k:[v for v in vv if len(self.vocab.pronuncie_vocab.parse(v[0]))<=hp.lm_max_len and len(self.vocab.words_vocab.parse(v[1]))<=hp.lm_max_len] for k,vv in self.data.items()}
        # print(len(self.data['dev-other']))
        # print(self.data['dev-clean'])
        # print(self.data['dev-other'])
        # print(self.data['train'][0][0].split())
    # def get_lexicon(self):
    #     pass
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
        # print([len(x) for x in batch['target']])
        # print(self.max_len)
        batch['input']=[self.padding(x,self.hp.lm_max_len-len(x)) for x in batch['input']]
        batch['target']=[self.padding(x,self.hp.lm_max_len-len(x)) for x in batch['target']]
        # print([len(x) for x in batch['target']])
        # print(self.max_len)
        # print(batch['input'][-1])
        # batch['input']=reduce(lambda x,y: ' '.join(x)+' '+' '.join(y),batch['input']).split()
        # batch['target']=reduce(lambda x,y: ' '.join(x)+' '+' '.join(y),batch['target']).split()
        # print([len(x) for x in batch['target']])#,len(batch['target'][0][0]))
        # print(batch['target'][0])
        # print(len(batch['target']),len(batch['target'][0]),len(batch['target'][0][0]))
        batch['input']=self.padding(batch['input'],self.hp.batch_size-len(batch['input']))
        batch['target']=self.padding(batch['target'],self.hp.batch_size-len(batch['target']))
        # print(batch['input'][-1])
        # print(batch['target'][0])
        # print([len(x) for x in batch['target']])#,len(batch['target'][0][0]))
        # print(len(batch['target']),len(batch['target'][0]),len(batch['target'][0][0]))
        # print(batch['target'])
        batch['input']=[self.vocab.pronuncie_vocab.one_hot_encoding(x) for x in batch['input']]
        batch['target']=[self.vocab.words_vocab.one_hot_encoding(x) for x in batch['target']]
        # print(batch['input'][-1])
        # print(len(batch['input']))
        # print(batch['input'][-1])
        # print([len(x) for x in batch['target']])#,len(batch['target'][0][0]))
        batch['input']=torch.LongTensor(np.array(batch['input'])).to(self.hp.device)
        # print(batch['input'].shape)
        batch['input']=batch['input'].flatten(0,1)
        # print(len(batch['target']))#,len(batch['target'][0]),len(batch['target'][0][0]))
        # print([len(x) for x in batch['target']])#,len(batch['target'][0][0]))
        batch['target']=torch.LongTensor(np.array(batch['target'])).to(self.hp.device)
        batch['target']=batch['target'].flatten(0,1)
        # batch['input']=batch['target']
        return batch
    def get_transcripts(self):
        '''
            Retrives the audio transcripts from the LibriSpeech dataset
            
            Inputs:
                None
            Outputs:
                transcripts: list[str]; the audio transcripts from the LibriSpeech dataset
        '''
        # train=LIBRISPEECH(self.librispeech_dir,self.hp.standard_train_type,download=not os.path.exists(os.path.join(self.librispeech_dir,self.hp.standard_train_type)))
        # print('hi')
        transcripts={}
        transcripts['train']=[x[2] for x in LIBRISPEECH(self.librispeech_dir,self.hp.standard_train_type,download=not os.path.exists(os.path.join(self.librispeech_dir,self.hp.standard_train_type)))]
        # print('hi')
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
        # shuffle(book_files)
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
                # return
                # books.append(b)
                books+=b
            # print(pronuncia_b_file)
                pronuncia_b_file=book_files[bf].replace('.txt','_pronuncia.txt')
            # if not os.path.exists(pronuncia_b_file):
                pronuncia_b=self.vocab.get_pronuncie_data(b)
                pronuncia_b_='\n'.join(pronuncia_b)
                with open(pronuncia_b_file,'w+') as f:
                        f.write(pronuncia_b_)
            # else:
            #     with open(pronuncia_b_file,'r') as f:
            #         pronuncia_b=f.read().strip().split('\n')
            pronuncie+=pronuncia_b
            if bf==len(book_files)-1:
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        # print(len(books))
        # books_max_len=max([len(line) for line in books])
        # books=[line for line in books if len(line)>3*books_max_len//4]
        # print(len(books))
        # print(books)
        # corpus=[[pronuncie[i],books[i]] for i in range(len(books))]
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
        # for line in book: print(line)
        book=[line.upper() for line in book]
        book=[line for line in book if '[' not in line]
        book=[''.join([c if c not in punctuation or c=="'" else ' ' for c in line]) for line in book]
        book=[' '.join([word for word in line.split() if len(word)>0]) for line in book]
        book=[line for line in book if len([True for word in line.split() if roman_num.match(word.strip())])==0]
        # print([x for x in book if 'BLEPHARIGLOTTIS' in x])
        # print([x.split() for x in book if 'CHAPTER' in x])
        # for line in book: print(line)
        # print([x for x in book if 'BULLETINSCROWDSTRUMPETSVOICESSOLDIERS' in x])
        # book=[' '.join([x for x in line.strip().split() if len(x)>0]) for line in book if line.strip()!='']
        book=[line for line in book if alpha.match(line)]
        book=[' '.join(line.split()[:self.hp.lm_max_len]) for line in book]
        # book_max_len=max([len(line) for line in book])
        # book=[line for line in book if len(line)>book_max_len//2]
        return book
    # def padding(self,l,left):
    #     # if pad_token is None:
    #     #     pad_token=self.vocab.pad_token
    #     if left==0:
    #         return l
    #     shape=np.array(l).shape[1:]
    #     # print(shape)
    #     # print(l)
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
        if type(l)!=list: l=l.tolist()
        if left==0: return l
        return l+np.zeros([left]+list(np.array(l).shape)[1:]).tolist() 
        
        