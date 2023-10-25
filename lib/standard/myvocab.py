import numpy as np
from torchtext.vocab import vocab
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stopset=stopwords.words('english')
# from string import punctuation
import re
from os.path import join,exists,split
from os import mkdir,listdir
from tqdm import tqdm
from colorama import Fore
from collections import Counter, OrderedDict
import torch
from functools import reduce
import sys

class myVocab():
    def __init__(self,base_path,data=None,ntokens=None,vocab_path=None,tokens=None):
        """
            class myVocab
        
            This class computes the vocabularies from given data and stores them
            in folder. If data is not given it will read the vocabularies from a 
            given folder.

            Inputs:
                base_path:str, path to the project folder.
                data:List[str], list of sentences. The vocabulary will be produce from these. Default: None
                ntokens: int; maximum number of tokens/words to be considered, the rest will be unknown; Default None
                vocab_path:str, path to the vocabulary. Default: None
                tokens: list[str]; list of tokens to be considered. This can be used to filter some tokens out. Default None
            Outputs:
                None
        """
        # self.full_predicates_list=full_predicates_list
        self.tokens=tokens
        self.ntokens=ntokens
        self.pad_token='<pad>'
        self.unk_token='<unk>'
        self.bos_token='<bos>'
        self.eos_token='<eos>'
        self.blank_token='<blank>'
        self.space_token=' '
        self.pad_index=0
        # print(vocabs_path,'vocabs' not in vocabs_path,not exists(join(vocabs_path,'vocabs')))
        if 'vocabs' not in base_path and not exists(join(base_path,'vocabs')):
            mkdir(join(base_path,'vocabs'))
        if vocab_path is None:
            self.vocab_path=join(base_path,'vocabs','myvocab.txt')
        else:
            self.vocab_path=vocab_path
        # self.vocabs_path=vocabs_path
        self.get_vocab(data)#,self.vocabs_path)
    def get_vocab(self,data):
        """
            Builds the vocabulary from the data and saves them or it read the vocabularies from the class folder.

            Inputs:
                data:List; list of sentences. Vocabularies will be computed from these
            Outputs:
                None
        """
        if data is None:
            self.read_vocab()#self.vocab_path)
        else:
            self.build_vocab(data)
            self.save_vocab()#self.vocab_path)
        self.vocab.set_default_index(self.vocab[self.unk_token])
    def build_vocab(self,data):
        """
            Builds the vocabularies from the data. 

            Inputs:
                data:List[str], list of sentences. 
            Outputs:
                None
        """
        vocab_data=self.get_data(data)
        if self.tokens is not None:
            vocab_data=[x for x in vocab_data if x in self.tokens]
        # vocab_data=self.clean_words(vocab_data)
        # specials=[self.pad_token,self.unk_token,self.bos_token,self.eos_token]
        min_freq=1
        #specials.update({k:[self.pad_token,self.bos_token,self.eos_token] for k in ['roles','one_hot_roles']})
        #min_freqs.update({k:1 for k in ['roles','one_hot_roles']})
        #self.vocab=self.build_vocab(vocab_data,specials,min_freqs)
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
        counter = Counter(vocab_data)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        if self.ntokens is not None:
            sorted_by_freq_tuples=sorted_by_freq_tuples[:self.ntokens]
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        self.vocab = vocab(ordered_dict)
        # print(len(self.vocab))
        # if self.space_token not in self.vocab:
        #     self.vocab.insert_token(self.space_token, 0)
        if self.unk_token not in self.vocab:
            self.vocab.insert_token(self.unk_token, 0)
        if self.pad_token not in self.vocab:
            self.vocab.insert_token(self.pad_token, 0)
        if self.blank_token not in self.vocab:
            self.vocab.insert_token(self.blank_token, 0)            
        # self.vocab.set_default_index(0)
    def get_data(self,dataset):
        """
            Retrieves a list of words that compose the senteces in the dataset

            Inputs:
                dataset:List[str], list of sentences. 
            Outputs:
                data: List[str]; list of words
        """
        # bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        # t=tqdm(range(len(dataset)),unit='audio',dynamic_ncols=True,bar_format=bar_format,desc="Retrieving words to build the transcript's vocabulary")
        # dataset=reduce(lambda x,y: x+y,dataset)
        data=[]
        for i in range(len(dataset)):
            s=list(dataset[i])#[2]
            # print(s)
            if type(s)==list:
                # print(s)
                s=s[2]
            w=s.strip().split()
            # print(w)
            data+=w
            # if i==len(dataset)-1:
            #     t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        return data
    # def clean_words(self,words,tokens=None):
    #     """
    #         Given a list of words returns a list of these stopset words since these don't give new information on the possible labels.

    #         Inputs:
    #             words:List[str], list of words or lemmas.
    #         Outputs:
    #             words:List[str], list of words/lemmas except for punctuation, stopset and alphanumeric words. 
    #     """
    #     # words=[x for x in words if x not in punctuation]
    #     words=[x for x in words if x.lower() not in stopset]
    #     # alpha=re.compile('^[a-zA-Z_]+$')
    #     # words=[x for x in words if alpha.match(x)]
    #     return words
    def save_vocab(self):
        """
            Saves the vocabulary

            Inputs:
                None                                         
            Outputs:
                None
        """
        #vocab_filename=join(vocabs_path,vocab_type+'.txt')
        #vocab=self.vocabs[vocab_type].stoi
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
        #vocab_name=split(vocab_path)[-1]
        #vocab_name=vocab_name[:vocab_name.rfind('.')]
        with open(self.vocab_path,'r',encoding='utf-8') as f:
            v=f.read().split('\n')
        v=v[:self.ntokens]
        v={v[i]:len(v)-i for i in range(len(v))}
        #specials=[x for x in v if x[0]=='<' and x[-1]=='>']
        self.vocab=vocab(v)#,specials=specials)
    def one_hot_encoding(self,sentence):
        '''
            Computes the one encoding of the sentence: each word is substituted by a list of zeros, long as the vocab size minus 1 and 1 in the position that corresponds to the word's encoding (if the vocab contains 3 words and current word's encoding is 1 then the one-hot-encoding is [0,1,0])
        Inputs:
            sentence: list[int]; encoded sentence
        Output:
            encoded: list[list[int]]; the sentence of one hot encoded words
        
        '''
        encoded=[]#[0]*len(self.vocab.vocab)]*len(sentence)
        vocab_len=len(self.vocab.vocab)
        # print(np.array(encoded).shape)
        for w in sentence:
            # print(i,w)
            enc=[0]*vocab_len
            enc[w]=1
            encoded.append(enc)
        return encoded
    # def encode(self,x):
    #     """
    #         Encodes a word or a token given a vocabulary.

    #         Inputs:
    #             x:str, word or token to encode.
    #             vocab:Vocab, a torchtext vocab to encode the token.
    #         Outputs:
    #             int, the token's encoding.
    #     """
    #     if x in self.vocab.vocab:
    #         return self.vocab[x]
    #     else:
    #         return self.vocab[self.unk_token]
    def encode_line(self,sentence):
        """
            Encodes a sentence.

            Inputs:
                sentence:List[str], list of words
            Outputs:
                encoded:List[int], an encoded sentence
        """
        # print(sentence)
        encoded=[self.vocab[x] for x in sentence]
        # if mask is not None and tok is not None:
        #     encoded=[encoded[i] if mask[i]==1 else tok for i in range(len(encoded))]
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
        # print(sentence)
        sentence=self.parse(sentence)
        # print(sentence)
        # sentence=self.encode_line(sentence)
        # print(sentence)
        sentence=self.one_hot_encoding(sentence)
        return sentence
    def parse(self,sentence):
        '''
            Encodes the given sentence
            
            Inputs:
                sentence: str; the sentence to encode
            Outputs:
                sentence: list[int]; encoded sentence
                
        '''
        sentence=sentence.strip().split()
        sentence=[self.vocab[x] for x in sentence]
        return sentence
    def decode(self,batch):#,sentence_len=None):
        '''
            Decodes the batch of sentences
            
            Inputs:
                batch: list[list[int]]; encoded predicted sentences
            Outputs:
                sentences: list[str]; decoded predicted sentence
        '''
        # sentence=torch.argmax(sentence,-1)
        # print(sentence)
        # print(sentence.shape)
        sentences=[]
        pads=[]
        # print(batch.shape)
        batch=batch.cpu()
        batch=batch.numpy().tolist()
        for j,sentence in enumerate(batch):
            # if sentence is not None:
            #     sentence=[x for x in sentence if x!=0]
            # else:
            #     sentence=sentence[:sentence_len]
            sentence=self.vocab.lookup_tokens(sentence)
            # print(self.vocab[self.space_token])
            # print(sentence)
            # if target is None:
                # print(sentence)
            pad_idxes=[x==self.pad_token or x==self.blank_token for x in sentence]
            sentence=[x for p,x in zip(pad_idxes,sentence) if not p]
            sentence=' '.join(sentence)#.split()
                # sentence=sentence.replace('__',' ')
                # sentence=sentence.replace('_','')
                # sentence=sentence.stripde()
                # sentence=sentence.split()
                # sentence=[x for x in sentence if x!='']
                # sentence=' '.join(sentence)
                # print(sentence)
                # return sentence,pad_idxes
            pads.append(pad_idxes)
            # else:
            #     # m=max(len(sentence),len(target))
            #     # print(sentence)
            #     sentence=[sentence[i] for i in range(len(sentence)) if i<len(target[j]) and not target[j][i]]
            #     sentence=' '.join(sentence)
            #     # sentence=sentence.replace('__',' ')
                # sentence=sentence.replace('_','')
                # sentence=sentence.strip()
                # sentence=sentence.split()
                # sentence=[x for x in sentence if x!='']
                # sentence=' '.join(sentence)
                # print(sentence)
                # print(target)
                # return sentence
            # if sentence!='':
            sentences.append(sentence)
        # if target is None:
        #     return sentences,pads
        return sentences
    # def get_mask(self,sentence,device):
    #     # sentence=sentence.cpu()
    #     # sentence=sentence.numpy().tolist()
    #     mask=sentence.eq(self.vocab[self.space_token])
    #     # mask=[w!=self.vocab[self.space_token] for w in]
    #     return mask
    #     # # print(self.pad_token in sentence)
    #     # # sentence=sentence.replace('<',' <')
    #     # # sentence=sentence.replace('>','> ')
    #     # # sentence=sentence.replace('  ',' ')
    #     # sentence=sentence.split(self.pad_token)
    #     # sentence=[x for x in sentence if len(x)>0 and x!=['']]
    #     # sentence=' '.join(sentence)
    #     # sentence=[s.split() for s in sentence]
    #     # sentence=[x for x in sentence if len(x)>0]
    #     # sentence=sentence.split()
    #     # print(sentence)
        