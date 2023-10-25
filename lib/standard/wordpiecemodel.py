import numpy as np
from torchtext.vocab import vocab
import re
from os.path import join,exists,split
from os import mkdir,listdir
from tqdm import tqdm
from colorama import Fore
from collections import Counter, OrderedDict
import torch
import sys
from functools import reduce
from torchtext.transforms import SentencePieceTokenizer
from transformers import RobertaTokenizer, RobertaModel
from jiwer import *
import os

class WPM():
    def __init__(self,base_path,data=None,ntokens=None,vocab_path=None,tokens=None,unk_tol=0.5):
        """
            class WPM
        
            This class computes the word-piece-model from given data and stores them
            in folder. If data is not given it will read the wpm from a 
            given folder.

            Inputs:
                base_path:str, path to the project folder.
                data:List[str], list of sentences. The vocabulary will be produce from these. Default: None
                ntokens: int; maximum number of tokens/words to be considered, the rest will be unknown; Default None
                vocab_path:str, path to the vocabulary. Default: None
                tokens: list[str]; list of tokens to be considered. This can be used to filter some tokens out. Default None
                unk_tol: float; the percentage (normalized so 1 is 100%) of the tollerable unknown words in a sentence. If the sentence has too many unknown then this is discarded
            Outputs:
                None
        """
        self.ntokens=ntokens
        self.tokens=tokens
        self.space_token=None 
        self.pad_token='<pad>'
        self.unk_token='<unk>'
        self.bow_token='<bow>'
        self.eow_token='<eow>'
        self.blank_token='<blank>'
        self.pad_index=0
        self.unk_tol=unk_tol
        self.transform=Compose([ToLowerCase(),RemoveMultipleSpaces(),Strip(),RemoveEmptyStrings(),RemovePunctuation(),ToUpperCase()])
        self.tokenizer=SentencePieceTokenizer(r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model")
        self.alpha=re.compile("^[a-zA-Z_']+$")
        lm_dir=os.path.join(base_path,'data','LibriSpeech','LibriSpeech_LM_data')
        self.lexicon_file=os.path.join(lm_dir,'librispeech-lexicon.txt')
        self.get_lexicon()
        if 'vocabs' not in base_path and not exists(join(base_path,'vocabs')):
            mkdir(join(base_path,'vocabs'))
        if vocab_path is None:
            self.vocab_path=join(base_path,'vocabs','wmp_vocab.txt')
        else:
            self.vocab_path=vocab_path
        self.get_vocab(data)
    def get_vocab(self,data):
        """
            Builds the vocabulary from the data and saves them or it read the vocabularies from the class folder.

            Inputs:
                data:List; list of sentences. Vocabularies will be computed from these
            Outputs:
                None
        """
        if data is None:
            self.read_vocab()
        else:
            self.build_vocab(data)
            self.save_vocab()
        self.vocab.set_default_index(self.vocab[self.unk_token])
    def build_vocab(self,data):
        """
            Builds the vocabularies from the data. 

            Inputs:
                data:List[str], list of sentences. 
            Outputs:
                None
        """
        vocab_data,words=self.get_data(data)
        if self.tokens is not None:
            vocab_data=[x for x in vocab_data if x in self.tokens]
        min_freq=1
        """
            Build a torchtext vocabulary from the given data.

            Inputs:
                data:list, list of words from which the vocabularies are built.
                special:list, list of special tokens such as padding.
                min_freq:int, minimum frequency of tokens in order to be considered when developing the vocabulary.
                              Default is 1. Is preferable, when building a vocabulary for input tokens for the model.
                              to have a number greater than 1 to make the model more robust for generalizing, so the
                              model can work also with tokens not encontered during training.
            Outputs:
                torchtext vocabulary computed from the given data.
        """
        counter = self.counter(vocab_data,words)
        sorted_by_freq_tuples = sorted(counter, key=lambda x: x[1], reverse=True)
        if self.ntokens is not None:
            sorted_by_freq_tuples=sorted_by_freq_tuples[:self.ntokens]
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        self.vocab = vocab(ordered_dict)
        if self.unk_token not in self.vocab:
            self.vocab.insert_token(self.unk_token, 0)
        if self.blank_token not in self.vocab:
            self.vocab.insert_token(self.blank_token, 0)
        if self.pad_token not in self.vocab:
            self.vocab.insert_token(self.pad_token, 0)
    def is_tollerable(self,sentence):
        '''
            Checks if the given sentence doesn't have too many unknown words
            
            Inputs:
                sentence: str; the transcript of the audio
            Outputs:
                True if the given sentence doesn't have too many unknown words, False otherwise; bool
        '''
        sentence=self.parse(sentence)
        sentence=self.decode([sentence])[0].split()
        n_unk=sentence.count(self.vocab[self.unk_token] if type(sentence[0])!=str else self.unk_token)
        return n_unk<int(self.unk_tol*len(sentence))
    def get_data(self,dataset):
        """
            Retrieves a list of tokens that compose the senteces in the dataset

            Inputs:
                dataset:List[str], list of sentences. 
            Outputs:
                data: List[str]; list of tokens
        """
        words=[]
        if type(dataset)==list:
            dataset=reduce(lambda x,y: x+y,dataset)
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        bar=tqdm(range(len(dataset)),unit='transcript',dynamic_ncols=True,bar_format=bar_format,desc="Retrieving words to build the transcript's vocabulary")
        for i in bar:
            s=dataset[i]
            if type(s) in [list,tuple]:
                s=s[2]
            s=s.strip()
            s=s.lower()
            w=self.partial_parse(s)
            words+=w
            if i==len(dataset)-1:
                bar.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        data=list(set(words))
        return data,words
    def counter(self,data,words):
        '''
            Counts the number of each word or token
            
            Inputs:
                data: list; list of tokens
            Outputs:
                counter: list; list of tokens with their corresponding number of occurrencies
        '''
        words=' '.join(words)
        counter=[[x,words.count(x)] for x in data]
        return counter
    def adjust_words(self,parsed):
        '''
            Differently from myVocab where each token corresponds to a word, here a token may represent a piece of a word or a whole word. Thus, the words have to be reconstructed from the list of tokens
            
            Inputs:
                parsed: list[int]; list of encoded tokens
            Outputs:
                s1: list[list[int]]; each element corresponds to a word (a list of tokens that form it)
        '''
        s=[]
        w=[]
        for i,x in enumerate(parsed):
            if self.space_token in self.vocab.vocab.lookup_token(x):
                if len(w)>0:
                    s.append(w)
                w=[x]
            else:
                w.append(x)
        if len(w)>0:
            s.append(w)
        s1=[]
        for i,x in enumerate(s):
            if self.vocab[self.unk_token] in x:
                s1.append(self.vocab[self.unk_token])
            else:
                s1+=x
        return s1
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
            wgrams=[w[i:i+n] for i in range(len(w)-n+1)]
            ngrams+=wgrams
        ngrams=[x for x in ngrams]
        return ngrams

    def save_vocab(self):
        """
            Saves the vocabulary

            Inputs:
                None                                         
            Outputs:
                None
        """
        text='\n'.join(list(self.vocab.lookup_tokens(list(range(len(self.vocab))))))
        with open(self.vocab_path,'w+',encoding='utf-8') as f:
            f.write(text)
    def read_vocab(self):
        """
            Loads a vocabulary given a path to its file.

            Inputs:
                vocabs_path:str, path to the vocabs folder.
            Outputs:
                name:str, name of the vocabulary which corresponds to the name of the file.
                a torchtext vocabulary of given the information in the file. Here minimum frequency
                is considered to be 1 (default for the Vocabs function) beacuse it's assumed that the 
                file contains only tokens with a minimum frequency above a certain threshold.

        """
        with open(self.vocab_path,'r',encoding='utf-8') as f:
            v=f.read().split('\n')
        v=v[:self.ntokens]
        if self.space_token is None:
            self.get_space_token(v)
        v={v[i]:len(v)-i for i in range(len(v))}
        self.vocab=vocab(v)
    def get_space_token(self,toks):
        '''
            Retreives the space token from tokens. Depending on how the tokens are generated the space could be identifies as ' '
 or '_'. This function gets the used one.   
 
            Inputs:
                toks: list[str]; list of tokens
            Outputs:
                None
        '''
        toks=''.join(toks)
        for t in toks:
            if not self.alpha.match(t):
                self.space_token=t
                break
    def one_hot_encoding(self,sentence):
        '''
            Computes the one encoding of the sentence: each token is substituted by a list of zeros, long as the vocab size minus 1 and 1 in the position that corresponds to the token's encoding (if the vocab contains 3 tokens and current word's encoding is 1 then the one-hot-encoding is [0,1,0])
        Inputs:
            sentence: list[int]; encoded sentence
        Output:
            encoded: list[list[int]]; the sentence of one hot encoded tokens
        
        '''
        encoded=[]
        vocab_len=len(self.vocab.vocab)
        for w in sentence:
            enc=[0]*vocab_len
            enc[w]=1
            encoded.append(enc)
        return encoded

    def encode_line(self,sentence,mask=None,tok=None):
        """
            Encodes a sentence.

            Inputs:
                sentence:List[str], list of tokens
            Outputs:
                encoded:List[int], an encoded sentence
        """
        encoded=[self.vocab[x] for x in sentence]
        if mask is not None and tok is not None:
            encoded=[encoded[i] if mask[i]==1 else tok for i in range(len(encoded))]
        return encoded
    def __len__(self):
        '''
            Overrides the len built-in function to make it work with the current class as well
            
            Inputs:
                None
            Output:
                the vocab length
        '''
        return len(self.vocab)
    def one_hot_sentence(self,sentence):
        '''
            Performes stardard encoding and one hot encoding on a given sentence
            
            Inputs:
                sentence: str; the sentence
            Outputs:
                sentence: list[list[int]]; one hot encoded sentence
        '''
        sentence=self.parse(sentence)
        sentence=self.one_hot_encoding(sentence)
        return sentence
        
    def space_adjust(self,sentence):
        '''
            Incorporating the space tokens in the tokens where needed and deleting consecutive space tokens
            
            Inputs:
                sentence: list[str]; list of tokens
            Outputs:
                sentence: list[str]; list of tokens after processing
        '''
        if self.space_token not in sentence[0]:
            sentence[0]=self.space_token+sentence[0]
        for i in range(1,len(sentence)):
            if sentence[i-1]==self.space_token:
                sentence[i]=self.space_token+sentence[i]
        sentence=[w for w in sentence if w!=self.space_token]
        return sentence
    def partial_parse(self,sentence):
        '''
            Tokenize the sentence
            
            Inputs:
                sentence: str; the transcript to tokenize
            Outputs:
                sentence: list[str]; list of tokens
        '''
        sentence=sentence.replace('*','')
        sentence=self.transform(sentence)
        if len(sentence)==0:
            return sentence
        sentence=self.tokenizer(sentence)
        if len(sentence)==0:
            return sentence
        sentence=[w.lower() for w in sentence]
        if self.space_token is None:
            self.get_space_token(sentence)
        sentence=self.space_adjust(sentence)
        return sentence
    def parse(self,sentence):
        '''
            Encodes the given sentence
            
            Inputs:
                sentence: str; the sentence to encode
            Outputs:
                sentence: list[int]; encoded sentence
                
        '''
        sentence=self.partial_parse(sentence)
        sentence=[self.vocab[w] for w in sentence]
        sentence=self.adjust_words(sentence)
        return sentence
        
    def decode(self,batch):
        '''
            Decodes the batch of sentences
            
            Inputs:
                batch: list[list[int]]; encoded predicted sentences
            Outputs:
                sentences: list[str]; decoded predicted sentence
        '''
        sentences=[]
        pads=[]
        if type(batch)!=list:
            batch=batch.cpu()
            batch=batch.numpy().tolist()
        for j,sentence in enumerate(batch):
                    sentence=self.vocab.lookup_tokens(sentence)
                    sentence=[x for x in sentence if x==self.blank_token or x==self.unk_token or '<' not in x]
                    if self.blank_token in ''.join(sentence):
                        sentence=[sentence[i] if i==0 or sentence[i]!=sentence[i-1] else '' for i in range(len(sentence))]
                        sentence=[x for x in sentence if x!=self.blank_token]
                    sentence=''.join(sentence)
                    sentence=sentence.replace(self.unk_token,self.space_token+self.unk_token)
                    sentence=sentence.replace(self.space_token,' ')
                    sentence=sentence.strip()
                    sentence=sentence.split()
                    sentence=[x for x in sentence if x!='']
                    sentence=' '.join(sentence)
                    sentences.append(sentence)
        return sentences
            
