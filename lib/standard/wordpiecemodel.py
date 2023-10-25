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
import sys
from functools import reduce
from torchtext.transforms import SentencePieceTokenizer
from transformers import RobertaTokenizer, RobertaModel
# from torchnlp.word_to_vector import FastText
from jiwer import *# ExpandCommonEnglishContractions, RemoveMultipleSpaces, RemovePunctuation, Compose, ToLowerCase, 
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
        # self.full_predicates_list=full_predicates_list
        self.ntokens=ntokens
        self.tokens=tokens
        self.space_token=None #' '
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
        # hi(vocabs_path,'vocabs' not in vocabs_path,not exists(join(vocabs_path,'vocabs')))
        lm_dir=os.path.join(base_path,'data','LibriSpeech','LibriSpeech_LM_data')
        self.lexicon_file=os.path.join(lm_dir,'librispeech-lexicon.txt')
        self.get_lexicon()
        if 'vocabs' not in base_path and not exists(join(base_path,'vocabs')):
            mkdir(join(base_path,'vocabs'))
        if vocab_path is None:
            self.vocab_path=join(base_path,'vocabs','wmp_vocab.txt')
        else:
            self.vocab_path=vocab_path
        # self.vocabs_path=vocabs_path
        self.get_vocab(data)#,self.vocabs_path)
        # self.weights=(torch.FloatTensor(list(range(len(self.vocab))))+1)/len(self.vocab)
        # self.weights[self.vocab[self.space_token]]=self.weights[self.vocab[self.space_token]]*0.8
        # self.weights=torch.ones(len(self.vocab))
        # self.weights[:self.vocab[self.space_token]+1]=0.1
        # name="roberta-base"
        # tokenizer = RobertaTokenizer.from_pretrained(name)
        # tokens=self.vocab.lookup_tokens(list(range(len(self.vocab))))
        # tokens=[tokenizer(tokens[i],return_tensors="pt") for i in range(len(tokens))]
        # model = RobertaModel.from_pretrained(name)#, return_all_hiddens=True)
        # if device is not None:
        #     model.to(device)
        # model.eval()
        # # Extract all layer's features (layer 0 is the embedding layer)
        # print(tokens)
        # self.embeddings=[model(tokens[i]['input_ids'],output_hidden_states=True).hidden_states[0].squeeze()[1:-1] for i in range(len(tokens))]
        # print(self.embeddings[0].shape)
        # self.embeddings=[torch.sum(x,dim=0) for x in self.embeddings]
        # print(self.embeddings[0].shape)
        # self.embeddings=torch.stack(self.embeddings,dim=0)
        # print(self.embeddings.shape)
        # print(self.embeddings[0].shape)
        # self.fasttext=FastText()
        # print([self.vocab.lookup_token(i) in self.fasttext for i in range(len(self.vocab))])
        # print([self.vocab.lookup_token(i) for i in range(len(self.vocab))])
        # self.embeddings=[self.fasttext[self.vocab.lookup_token(i)] for i in range(len(self.vocab))]
        # self.embeddings=torch.stack(self.embeddings)
        # print(stopwords)
        # for w in stopset:
        #     idx=self.vocab[w.upper()]
        #     # if idx!=self.vocab[self.unk_token]:
        #     self.weights[idx]=0.1
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
        vocab_data,words=self.get_data(data)
        if self.tokens is not None:
            vocab_data=[x for x in vocab_data if x in self.tokens]
        # vocab_data=self.clean_words(vocab_data)
        # vocab_data=[x for x in vocab_data if x.replace(self.space_token,'').lower() not in stopset]
        # specials=[self.pad_token,self.unk_token,self.bos_token,self.eos_token]
        min_freq=1
        #specials.update({k:[self.pad_token,self.bos_token,self.eos_token] for k in ['roles','one_hot_roles']})
        #min_freqs.update({k:1 for k in ['roles','one_hot_roles']})
        # """
        #     Vocabs is a dictionary of vocabularies where the keys corrispond to the respective key in a sentence of the data.
        # """
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
        counter = self.counter(vocab_data,words)
        sorted_by_freq_tuples = sorted(counter, key=lambda x: x[1], reverse=True)
        if self.ntokens is not None:
            sorted_by_freq_tuples=sorted_by_freq_tuples[:self.ntokens]
        # hi(sorted_by_freq_tuples)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        self.vocab = vocab(ordered_dict)
        # hi(len(self.vocab))
        # if self.space_token not in self.vocab:
        #     self.vocab.insert_token(self.space_token, 0)
        if self.unk_token not in self.vocab:
            self.vocab.insert_token(self.unk_token, 0)
        # if self.eow_token not in self.vocab:
        #     self.vocab.insert_token(self.eow_token, 0)
        # if self.bow_token not in self.vocab:
        #     self.vocab.insert_token(self.bow_token, 0)
        if self.blank_token not in self.vocab:
            self.vocab.insert_token(self.blank_token, 0)
        if self.pad_token not in self.vocab:
            self.vocab.insert_token(self.pad_token, 0)
        # self.vocab.set_default_index(self.vocab[self.unk_token])
        # self.organize_by_len()
    def is_tollerable(self,sentence):
        '''
            Checks if the given sentence doesn't have too many unknown words
            
            Inputs:
                sentence: str; the transcript of the audio
            Outputs:
                True if the given sentence doesn't have too many unknown words, False otherwise; bool
        '''
        # print(sentence)
        sentence=self.parse(sentence)
        # print(sentence)
        sentence=self.decode([sentence])[0].split()
        n_unk=sentence.count(self.vocab[self.unk_token] if type(sentence[0])!=str else self.unk_token)
        # print(sentence)
        # print(n_unk,self.unk_tol*len(sentence))
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
        # self.longest_w=0
        for i in bar:
            s=dataset[i]
            if type(s) in [list,tuple]:
                s=s[2]
            s=s.strip()
            s=s.lower()
            # s=s.replace(' ','_ _')
            # s='_'+s+'_'
            # w=s.split()
            # w=[self.word2lexi[x].split() for x in w if x in self.word2lexi]
            # w=reduce(lambda x,y: x+y,w)
            # self.space_token='_'
            # apo=False
            # if "'" in s:
            #     apo=True
                # print(s)
            # s=ExpandCommonEnglishContractions()(s).upper()
            # # if apo and "'" in s:
            # #     print(s)
            # #     print()
            # w=self.tokenizer(s)
            # w=[x.lower() for x in w]
            # if self.space_token is None:
            #     self.get_space_token(w)
            #     # alpha=re.compile("^[a-zA-Z']+$")
            #     # tmp=''.join(w)
            #     # for t in tmp:
            #     #     if not alpha.match(t):
            #     #         self.space_token=t
            #     #         break
            # #     hi(self.space_token)
            # # if i<10:
            # #     hi()
            # #     hi(w)
            # #     hi(self.space_token)
            # #     hi([self.space_token+w[j] if j>0 and w[j-1]==self.space_token and w[j][0]!=self.space_token else w[j] for j in range(len(w))])
            # # w=[self.space_token+w[j] if j>0 and w[j-1]==self.space_token and w[j][0]!=self.space_token else w[j] for j in range(len(w))]
            # # w=[x for x in w if x!=self.space_token]
            # w=self.space_adjust(w)
            w=self.partial_parse(s)
            # self.longest_w=max(self.longest_w,max(self.get_word_lengths(w)))
            # w=self.clean_words(w)
            # if i<10:
            #     hi(w)
            #     hi()
            # w=reduce(lambda x,y: x+y,w)
            # w=['_'+x+'_' for x in w]
            words+=w
            if i==len(dataset)-1:
                bar.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        # hi(words)
        # self.longest_w=max([len(x) for x in words])
        # # hi(longest_w)
        # data=[]#words.copy()
        # for i in range(1,longest_w):
        #     data+=self.get_ngrams(words,i)
        # data=list(set(data))
        # data=[x for x in data if x!='_' and x!='__']
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
    # def organize_by_len(self):
    #     tokens=self.vocab.get_itos()
    #     tokens=[x for x in tokens if '<' not in x]
    #     tokens.sort(key=lambda x: len(x),reverse=True)
    #     self.len_ordered_toks=tokens
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
        # parsed=self.vocab.lookup_tokens(parsed)
        for i,x in enumerate(parsed):
            # if x[0]==self.space_token:
            # if x==self.vocab[self.space_token]:
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
                # s[i]=[self.vocab[self.space_token],self.vocab[self.unk_token]]
                # s[i]=[self.vocab[self.unk_token]]
                s1.append(self.vocab[self.unk_token])
            else:
                s1+=x
        # s=reduce(lambda x,y: x+y,s)
        # s=[self.vocab[w] for w in s]
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
            # if len(w)<n:
            wgrams=[w[i:i+n] for i in range(len(w)-n+1)]
            # if len(wgrams)>0:
            #     wgrams[0]='_'+wgrams[0]
            #     wgrams[-1]=wgrams[-1]+'_'
            # hi(wgrams)
            ngrams+=wgrams
        ngrams=[x for x in ngrams]
        return ngrams
    # def clean_words(self,words):
    #     """
    #         Given a list of words (or lemmas) returns a list of these without punctuation, stopset words and aphanumeric words
    #         since these don't give new information on the possible labels.

    #         Inputs:
    #             words:List[str], list of words or lemmas.
    #         Outputs:
    #             words:List[str], list of words/lemmas except for punctuation, stopset and alphanumeric words. 
    #     """
    #     # words=[x for x in words if x.replace('_','') not in punctuation]
    #     words=[words[i] for i in range(len(words)) if not (i<len(words)-1 and words[i][0]==self.space_token and words[i+1][0]==self.space_token and words[i].replace(self.space_token,'').lower() in stopset)]
    # #     alpha=re.compile('^[a-zA-Z_]+$')
    # #     words=[x for x in words if alpha.match(x)]
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
        if self.space_token is None:
            self.get_space_token(v)
        v={v[i]:len(v)-i for i in range(len(v))}
        #specials=[x for x in v if x[0]=='<' and x[-1]=='>']
        self.vocab=vocab(v)#,specials=specials)
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
        # alpha=re.compile("^[a-zA-Z'<>]+$")
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
        encoded=[]#[0]*len(self.vocab.vocab)]*len(sentence)
        vocab_len=len(self.vocab.vocab)
        # hi(np.array(encoded).shape)
        for w in sentence:
            # hi(i,w)
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
    def encode_line(self,sentence,mask=None,tok=None):
        """
            Encodes a sentence.

            Inputs:
                sentence:List[str], list of tokens
            Outputs:
                encoded:List[int], an encoded sentence
        """
        # hi(sentence)
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
        # hi()
        # hi(len(self))
        # hi(sentence)
        sentence=self.parse(sentence)
        # hi(sentence)
        # sentence=self.encode_line(sentence)
        # hi(sentence)
        sentence=self.one_hot_encoding(sentence)
        # sentence=[[0]*len(sentence[0])]+sentence
        # hi(sentence)
        # hi()
        return sentence
    # def separate_word(self,word,n):
        
    #     # hi(word,word in self.vocab)
    #     if word in self.vocab:
    #         return [word]
    #     pieces=[[word,False,len(word)]]
    #     while False in [x[1] for x in pieces if x[2]>0]:
    #         for j,p in enumerate(pieces):
    #             if not p[1] and p[2]>0:
    #                 ngrams=self.get_ngrams([p[0]],p[2])
    #                 present=[]
    #                 for g in ngrams:
    #                     if g in self.vocab:
    #                         present.append(g)
    #                 # hi()
    #                 # hi('hi')
    #                 # hi(p)
    #                 # hi(ngrams)
    #                 # hi(present)
    #                 # hi()
    #                 if p[2]>1:
    #                     i=0
    #                     while i<len(present)-1:
    #                         if present[i][1:]==present[i][:-1] and present[i]+present[i+1][-1] in word:
    #                             if self.vocab[present[i]]<self.vocab[present[i+1]]:
    #                                 if i<len(present)-2:
    #                                     present=present[:i+1]+present[i+2:]
    #                                 else:
    #                                     present=present[:i+1]
    #                                 i+=1
    #                             else:
    #                                 present=present[:i]+present[i+1:]
    #                         else:
    #                             i+=1
    #                 piece=p[0]
    #                 for x in present:
    #                     piece=piece.replace(x,'_'+x+'_')
    #                 piece=piece.split('_')
    #                 for i in range(len(piece)):
    #                     if piece[i] in present:
    #                         piece[i]=[piece[i],True,p[2]]
    #                     else:
    #                         piece[i]=[piece[i],False,min(len(piece[i]),p[2]-1)]
    #                 # hi(pieces[j])
    #                 piece=[pi for pi in piece if pi[0]!='']
    #                 pieces[j]=piece
    #                 # if j<len(pieces)-1:
    #                 #     pieces=pieces[:j]+piece+pieces[j+1:]
    #                 # else:
    #                 #     pieces=pieces[:j]+piece+pieces[j+1:]
    #         pieces_=[]
    #         # hi(pieces)
    #         for p in pieces:
    #             if type(p[0])==list:
    #                 pieces_+=p
    #             else:
    #                 pieces_.append(p)
    #         pieces=pieces_
    #         # hi(pieces)
    #     pieces=[p[0] for p in pieces]
    #     return pieces
    # def space_adjust1(self,sentence):
    #     # sentence=[self.space_token+sentence[j] if j>0 and sentence[j-1]==self.space_token and sentence[j][0]!=self.space_token else sentence[j] for j in range(len(sentence))]
    #     # sentence=[x for x in sentence if x!=self.space_token]
    #     s=[]
    #     for w in sentence:
    #         if self.space_token in w and w!=self.space_token:
    #             s+=[self.space_token,w[1:]]
    #         else:
    #             s.append(w)
    #     return s#entence
    def space_adjust(self,sentence):
        '''
            Incorporating the space tokens in the tokens where needed and deleting consecutive space tokens
            
            Inputs:
                sentence: list[str]; list of tokens
            Outputs:
                sentence: list[str]; list of tokens after processing
        '''
        # s=[]
        # sentence=[x if x!=self.space_token else self.space_token+x for x in sentence]
        # prec=sentence[0] if self.space_token not in sentence[0] else self.space_token+sentence[0]
        # sentence=[] if prec==self.space_token else [prec] 
        # prec=''
        # if '_' in sentence:
        #     print(sentence)
        if self.space_token not in sentence[0]:
            sentence[0]=self.space_token+sentence[0]
        for i in range(1,len(sentence)):
            if sentence[i-1]==self.space_token:
                sentence[i]=self.space_token+sentence[i]
            # if w==self.space_token:
            #     prec=
        sentence=[w for w in sentence if w!=self.space_token]
        return sentence
    # def get_word_lengths(self,sentence):
    #     # sentence=sentence.numpy().tolist()
    #     space_idx=[i for i in range(len(sentence)) if sentence[i]==self.vocab[self.space_token] or sentence[i]==self.space_token or sentence[i]==self.vocab[self.pad_token] or sentence[i]==self.pad_token]
    #     # print(len(sentence),len(space_idx))
    #     lengths=[space_idx[0]]+[space_idx[i]-space_idx[i-1] for i in range(1,len(space_idx))]+[len(sentence)-space_idx[-1]]
    #     lengths=[x for x in lengths if x>0]
    #     return lengths
    # def group_words(self,sentence):
    #     # print(self.vocab.lookup_tokens(sentence.cpu().numpy().tolist()))
    #     # print(space_idx)
    #     # print(lengths)
    #     # print()
    #     s=[]
    #     w=[]
    #     for x in sentence:
    #         if x==self.vocab[self.space_token]:
    #             if len(w)>0:
    #                 s.append(w)
    #             w=[x]
    #         else:
    #             w.append(x)
    #     if len(w)>0:
    #         s.append(w)
    #     # s=[w+[0]*(self.longest_w-len(w)) for w in s]
    #     # if target:
    #     #     s=torch.LongTensor(s).to(device)
    #     # else:
    #     #     s=torch.FloatTensor(s).to(device)
    #     return s
    def partial_parse(self,sentence):
        '''
            Tokenize the sentence
            
            Inputs:
                sentence: str; the transcript to tokenize
            Outputs:
                sentence: list[str]; list of tokens
        '''
        # sentence=ExpandCommonEnglishContractions()(sentence.lower()).upper()
        # sentence=RemoveMultipleSpaces()(sentence)
        # sentence=RemovePunctuation()(sentence)
        # print(sentence)
        sentence=sentence.replace('*','')
        sentence=self.transform(sentence)
        if len(sentence)==0:
            return sentence
        # print(sentence)
        # if '_' in sentence:
        #     print(sentence)
        sentence=self.tokenizer(sentence)
        if len(sentence)==0:
            return sentence
        # print(sentence)
        sentence=[w.lower() for w in sentence]
        # print(sentence)
        if self.space_token is None:
            self.get_space_token(sentence)
        sentence=self.space_adjust(sentence)
        # print(sentence)
        return sentence
    def parse(self,sentence):
        '''
            Encodes the given sentence
            
            Inputs:
                sentence: str; the sentence to encode
            Outputs:
                sentence: list[int]; encoded sentence
                
        '''
        # hi()
        # hi(sentence)
        # sentence is one and is a string
        # sentence=sentence.split()
        # sentence=[[self.space_token]+self.word2lexi[x].split() if x in self.word2lexi else [self.space_token,self.unk_token] for x in sentence]
        # sentence=reduce(lambda x,y: x+y,sentence)
        # print(sentence)
        # hi()
        # hi(stopset)
        # hi(sentence)
        # sentence=[self.unk_token if i<len(sentence)-1 and sentence[i][0]==self.space_token and sentence[i+1][0]==self.space_token and sentence[i].replace(self.space_token,'').lower() in stopset else sentence[i] for i in range(len(sentence))]
        # hi(sentence)
        # hi()
        # print(sentence)
        sentence=self.partial_parse(sentence)
        sentence=[self.vocab[w] for w in sentence]
        sentence=self.adjust_words(sentence)
        # if check and not self.is_tollerable(sentence):
        #     sentence=[]
        # print(self.vocab.lookup_tokens(sentence))
        # print()
        return sentence
        # sentence=sentence.strip()
        # sentence='_'+sentence.replace(' ','__')+'_'
        # # sentence=sentence.split()
        # # sentence=['_'+x+'_' for x in sentence]
        # # sentence=self.bow_token+sentence+self.eow_token
        # # sentence=sentence.replace(' ',self.eow_token+'_'+self.bow_token)
        # # s=[]
        # # for x in self.vocab.get_itos():
        # #     sentence=sentence.replace(x,'_'+str(self.vocab[x])+'_')
        # # for w in sentence:
        # #     s+=self.separate_word(w,len(w))
        # max_len=max([len(x) for x in self.vocab.get_itos()])
        # for n in range(max_len,-1,-1):
        #     tokens=[x for x in self.vocab.get_itos() if len(x)==n]
        #     for x in tokens:
        #         # sentence=sentence.replace(x,'_'+str(self.vocab['_'+x+'_'])+'_')
        #         # s='_' if x[0]=='_' else ''
        #         # e='_' if x[-1]=='_' else ''
        #         sentence=sentence.replace(x,'_'+str(self.vocab[x])+'_')
        
        # # for x in tokens:
        # #     if len(x)>1:
        # # tokens=[x for x in tokens if len(x)==1]
        # # for x in tokens:
        # #     # if len(x)>1:
        # #     sentence=sentence.replace(x,'_'+str(self.vocab[x])+'_')
        # sentence=sentence.split('_')
        # sentence=[x.strip() for x in sentence if x.strip()!='']
        # sentence=[self.vocab[x] if self.alpha.match(x) else x for x in sentence]
        # sentence=[int(x) for x in sentence]
        # # hi(sentence)
        # # hi()
        # return sentence
    # def mask(self,sentence):
    #     # sentence=sentence.replace(self.vocab[self.bow_token],self.voab[self.pad_token])
    #     # sentence=sentence.replace(self.vocab[self.eow_token],self.voab[self.pad_token])
    #     # sentence=
    #     return sentence
    def decode(self,batch):#,sentence_len=None):
        '''
            Decodes the batch of sentences
            
            Inputs:
                batch: list[list[int]]; encoded predicted sentences
            Outputs:
                sentences: list[str]; decoded predicted sentence
        '''
        # sentence=torch.argmax(sentence,-1)
        # hi(sentence)
        # hi(sentence.shape)
        sentences=[]
        pads=[]
        # hi(batch.shape)
        if type(batch)!=list:
            batch=batch.cpu()
            batch=batch.numpy().tolist()
        for j,sentence in enumerate(batch):
            # if sentence is not None:
            #     sentence=[x for x in sentence if x!=0]
            # else:
            #     sentence=sentence[:sentence_len]
                    # print(self.vocab.lookup_tokens(sentence))
                    # sentence=self.adjust_words(sentence)
                    sentence=self.vocab.lookup_tokens(sentence)
                # hi(self.vocab[self.space_token])
                # hi(sentence)
                # if target is None:
                    # hi(sentence)
                    # if j==0:
                    #     hi()
                    #     hi(''.join(sentence))
                    # pad_idxes=[x==self.pad_token or (x==self.space_token and sentence[i+1]==self.unk_token) or x==self.unk_token for i,x in enumerate(sentence)]
                    # sentence=[x for p,x in zip(pad_idxes,sentence) if not p]
                    sentence=[x for x in sentence if x==self.blank_token or x==self.unk_token or '<' not in x]
                    # print(sentence)
                    if self.blank_token in ''.join(sentence):
                        sentence=[sentence[i] if i==0 or sentence[i]!=sentence[i-1] else '' for i in range(len(sentence))]
                        sentence=[x for x in sentence if x!=self.blank_token]
                    # print(sentence)
                    sentence=''.join(sentence)#.split()
                    # sentence=sentence.split(self.space_token)
                    # sentence=[x.strip() for x in sentence if x.strip()!='']
                    # sentence=[self.lexi2word[x] if x in self.lexi2word else ''.join([self.lexi2word[y] if y in self.lexi2word else y for y in x.split()]) for x in sentence]
                    # sentence=[x if self.alpha.match(x) else '' for x in sentence]
                    # sentence=' '.join(sentence)#.split()
                    # sentence=sentence.replace(self.space_token+self.space_token,' ')
                    # hi(self.space_token)
                    sentence=sentence.replace(self.unk_token,self.space_token+self.unk_token)
                    sentence=sentence.replace(self.space_token,' ')
                    sentence=sentence.strip()
                    sentence=sentence.split()
                    sentence=[x for x in sentence if x!='']
                    sentence=' '.join(sentence)
                    # hi(sentence)
                    # return sentence,pad_idxes
                    # if sentence!='':
                    sentences.append(sentence)
                        # pads.append(pad_idxes)
                # else:#if j<len(target):
                #     # m=max(len(sentence),len(target))
                #     # hi(sentence)
                #     sentence=[sentence[i] for i in range(len(sentence)) if i<len(target[j]) and not target[j][i]]
                #     sentence=''.join(sentence)
                #     # sentence=sentence.replace(self.space_token+self.space_token,' ')
                #     sentence=sentence.replace(self.space_token,' ')
                #     sentence=sentence.strip()
                #     # sentence=sentence.split()
                #     # sentence=[x for x in sentence if x!='']
                #     # sentence=' '.join(sentence)
                #     # hi(sentence)
                #     # hi(target)
                #     # return sentence
                #     # if sentence!='':
                #     sentences.append(sentence)
        # if target is None:
        #     return sentences,pads
        return sentences
        # # sentence=sentence.replace('<',' <')
        # # sentence=sentence.replace('>','> ')
        # # sentence=sentence.replace('  ',' ')
        # sentence=sentence.split(self.pad_token)
        # sentence=[x for x in sentence if len(x)>0 and x!=['']]
        # sentence=[s.split() for s in sentence]
        # sentence=[x for x in sentence if len(x)>0]
        # hi(sentence)
        # return sentence
    # def get_mask(self,sentence):
    #     # sentence=sentence.cpu()
    #     # sentence=sentence.numpy().tolist()
    #     mask=sentence.eq(self.vocab[self.space_token])
    #     # mask=[w!=self.vocab[self.space_token] for w in]
    #     return mask
    # def get_weight(self,sentence):
    #     n=len(self.vocab)
    #     sentence=(sentence+1)/n
    #     # w=[(x+1)/n for x in sentence]
    #     return w
    # def get_lexicon(self):
    #     '''
    #         Retrieves the lexicon of the 
            
    #         Inputs:
    #             None
    #         Outputs:
    #             None
    #     '''
    #     with open(self.lexicon_file,'r') as f:
    #         self.word2lexi=f.read().strip().split('\n')
    #     self.word2lexi=[x.replace('  ','\t').split('\t') if x!='HH HH' else x.split() for x in self.word2lexi]
    #     self.word2lexi={kv[0].strip():kv[1].strip() for kv in self.word2lexi}
    #     self.lexi2word={v:k for k,v in self.word2lexi.items()}
