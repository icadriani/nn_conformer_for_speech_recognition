import numpy as np
from torchtext.vocab import vocab
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stopset=stopwords.words('english')
# from string import punctuation
import re
import os
from os.path import join,exists,split
from os import mkdir,listdir
from tqdm import tqdm
from colorama import Fore
from collections import Counter, OrderedDict
import torch
from lib.standard.myvocab import myVocab
from lib.standard.wordpiecemodel import WPM

import sys

class LMVocab():
    def __init__(self,hp,data=None,pronuncia_data=None):
        """
            This class handles the language model vocabulary
            
            Inputs:
                hp: HParams; the class of hyperparameters
                data: data from which the vocabulary is built. Default: None
                pronuncia_data: pronunciation data from which the pronunciation vocabulary is built. Default: None
            Outputs:
                None
        """
        # self.full_predicates_list=full_predicates_list
        # self.ntokens=ntokens
        self.pad_token='<pad>'
        # self.unk_token='<unk>'
        # self.bos_token='<bos>'
        # self.eos_token='<eos>'
        # self.pad_index=0
        self.hp=hp
        self.librispeech_dir=os.path.join(hp.data_dir,'LibriSpeech')
        self.lm_dir=os.path.join(self.librispeech_dir,'LibriSpeech_LM_data')
        self.lexicon_file=os.path.join(self.lm_dir,'librispeech-lexicon.txt')
        # print(vocabs_path,'vocabs' not in vocabs_path,not exists(join(vocabs_path,'vocabs')))
        if 'vocabs' not in hp.base_dir and not exists(join(hp.base_dir,'vocabs')):
            mkdir(join(hp.base_dir,'vocabs'))
        self.words_vocab_path=join(hp.base_dir,'vocabs','words_lexicon_vocab.txt')
        self.pronuncie_vocab_path=join(hp.base_dir,'vocabs','pronuncie_lexicon_vocab.txt')
        # self.vocabs_path=vocabs_path
        self.get_lexicon()
        # self.get_vocab(data)#,self.vocabs_path
        
        if data is not None:
            self.get_words_vocab(data)
            # self.words_vocab=myVocab(hp.base_dir,ntokens=None,vocab_path=self.words_vocab_path,tokens=list(self.word2lexi.keys()))
        # self.pronuncie_data=self.get_pronuncie_data(data)
        if pronuncia_data is not None:
            self.get_pronuncie_vocab(pronuncia_data)
    def get_words_vocab(self):
        '''
            Retrieves the words vocabulary
            
            Inputs: 
                None
            Outputs:
                None
        '''
        self.words_vocab=WPM(self.hp.base_dir,ntokens=self.hp.lm_ntokens,vocab_path=self.words_vocab_path,tokens=list(self.word2lexi.keys()))
        # self.words_vocab=WPM(self.hp.base_dir,data=data,ntokens=self.hp.lm_ntokens,vocab_path=self.words_vocab_path,tokens=list(self.word2lexi.keys()))
    def get_pronuncie_vocab(self,pronuncia_data):
        '''
            Builds a vocabulary given the pronunciations data
            
            Inputs: 
                pronuncia_data: list; pronunciation data
            Outputs:
                None
        '''
        # print(pronuncia_data[:100])
        self.pronuncie_vocab=myVocab(self.hp.base_dir,data=pronuncia_data,ntokens=self.hp.lm_ntokens,vocab_path=self.pronuncie_vocab_path)
        # self.pronuncie_vocab=myVocab(self.hp.base_dir,ntokens=None,vocab_path=self.pronuncie_vocab_path)
    def get_pronuncie_data(self,data):
        '''
            Retrieves the pronunciations data
            
            Inputs:
                data: list[str]; list of sentences
            Outputs:
                pronuncie_data: list[str]; list of pronunciations
        '''
        # pronuncie_data='\n'.join(data)
        pronuncie_data=[]
        # max_len=max([len(x) for x in self.word2lexi])
        # bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        # t=tqdm(range(max_len),unit='sentence',dynamic_ncols=True,bar_format=bar_format,desc='Retriving the pronounciation of the data')
        for sentence in data:
            pronuncie_data.append([])
            sentence=sentence.strip().split()
            for word in sentence:
                if '<' in word:
                    pronuncie_data[-1].append(word)
                # elif word in self.word2lexi:
                #     pronuncie_data[-1].append(self.word2lexi[word])
                else:
                    word=self.separate_word(word,len(word))
                    word=[self.word2lexi[x] for x in word if x in self.word2lexi]
                    pronuncie_data[-1]+=word
                    # print(pronuncie_data[-1])
                    # print(word)
            pronuncie_data[-1]=' '.join(pronuncie_data[-1])
            # print(pronuncie_data[-1])
            # print()
        # for i in t:
        #     sentence=data[i]
        # for n in range(max_len,-1,-1):
        #     tokens=[x for x in self.word2lexi if len(x)==n]
        #     for x in tokens:
        #         # print(pronuncie_data)
        #         if '<' not in x:
        #             pronuncie_data=pronuncie_data.replace(x,' '+self.word2lexi[x]+' ')
        #         else:
        #             pronuncie_data=pronuncie_data.replace(x,' '+x+' ')
                # print(sentence)
                # sentence=' '.join()
            # pronuncie_data.append(sentence)
            # if n==0:#len(data)-1:
            #     t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
            # t.update(1)
        # pronuncie_data=pronuncie_data.split('\n')
        # pronuncie_data=[' '.join([x for x in line.split() if len(x)>0]) for line in pronuncie_data]
        return pronuncie_data
    def separate_word(self,word):
        # print(word,word in self.word2lexi)
        '''
            Separates a given word into tokens where each token has a corresponding lexicon
            
            Inputs:
                word: str; the given word
            Outputs:
                None
        '''
        if word in self.word2lexi:
            return [word]
        pieces=[[word,False,len(word)]]
        while False in [x[1] for x in pieces if x[2]>0]:
            for j,p in enumerate(pieces):
                if not p[1] and p[2]>0:
                    ngrams=self.get_ngrams([p[0]],p[2])
                    present=[]
                    for g in ngrams:
                        if g in self.word2lexi:
                            present.append(g)
                    # print()
                    # print('hi')
                    # print(p)
                    # print(ngrams)
                    # print(present)
                    # print()
                    if p[2]>1:
                        i=0
                        while i<len(present)-1:
                            if present[i][1:]==present[i][:-1] and present[i]+present[i+1][-1] in word:
                                if self.word2lexi[present[i]]<self.word2lexi[present[i+1]]:
                                    if i<len(present)-2:
                                        present=present[:i+1]+present[i+2:]
                                    else:
                                        present=present[:i+1]
                                    i+=1
                                else:
                                    present=present[:i]+present[i+1:]
                            else:
                                i+=1
                    piece=p[0]
                    for x in present:
                        piece=piece.replace(x,'_'+x+'_')
                    piece=piece.split('_')
                    for i in range(len(piece)):
                        if piece[i] in present:
                            piece[i]=[piece[i],True,p[2]]
                        else:
                            piece[i]=[piece[i],False,min(len(piece[i]),p[2]-1)]
                    # print(pieces[j])
                    piece=[pi for pi in piece if pi[0]!='']
                    pieces[j]=piece
                    # if j<len(pieces)-1:
                    #     pieces=pieces[:j]+piece+pieces[j+1:]
                    # else:
                    #     pieces=pieces[:j]+piece+pieces[j+1:]
            pieces_=[]
            # print(pieces)
            for p in pieces:
                if type(p[0])==list:
                    pieces_+=p
                else:
                    pieces_.append(p)
            pieces=pieces_
            # print(pieces)
        pieces=[p[0] for p in pieces]
        return pieces
    def get_ngrams(self,words,n):
        '''
            Gets the n-grams of the words (groups of n letters)
            
            Inputs:
                words: list[str]; list of words
                n: int; number of letters in the gram (group)
            Outputs:
                ngrams: list[str]; the list of ngrams (groups of n letters)
        '''
        ngrams=[]
        for w in words:
            # if len(w)<n:
            wgrams=[w[i:i+n] for i in range(len(w)-n+1)]
            # if len(wgrams)>0:
            #     wgrams[0]='_'+wgrams[0]
            #     wgrams[-1]=wgrams[-1]+'_'
            # print(wgrams)
            ngrams+=wgrams
        return ngrams
    def get_lexicon(self):
        '''
            Retrieves the lexicon data
            
            Inputs:
                None
            Outputs:
                None
        '''
        with open(self.lexicon_file,'r') as f:
            self.word2lexi=f.read().strip().split('\n')
        self.word2lexi=[x.replace('  ','\t').split('\t') if x!='HH HH' else x.split() for x in self.word2lexi]
        self.word2lexi={kv[0].strip():kv[1].strip() for kv in self.word2lexi}
        # self.lexi2word={v:k for k,v in self.word2lexi.items()}
        # sample={k:v for k,v in self.word2lexi.items() if len(v.split())==1}
        # for x in sample: print(x)
        # a=[x.split() for x in self.word2lexi.values()]
        # b=[]
        # for x in a:
        #     b+=x
        # b=list(set(b))
        # for x in b: print(x)
        # print(len(b))
        