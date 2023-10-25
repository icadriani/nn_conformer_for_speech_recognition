import torch
from torch.optim import Adam
from transformers import Adafactor
from torch.nn import CTCLoss
from statistics import mean
from lib.evals import Evals
from tqdm import tqdm
from colorama import Fore
from math import ceil, isnan
import os
from jiwer import wer
import torch.nn as nn
from math import isnan

class Runner():
    '''
        class Runner
        
        This class includes the functions that run the model such training and testing/
        
        Inputs:
            model: ASRNN; The model to train/test
            hp: HParametes; The class of hyparameters
            lr: float; learning rate. Default None
            lm: the language model. Default None
    '''
    def __init__(self,model,hp,lr=None,lm=False):
        self.model=model
        self.hp=hp
        self.lm=lm
        if lr is None:
            lr=hp.lr
        self.model.to(hp.device)
        self.loss=CTCLoss(blank=hp.blank_idx,zero_infinity=True)
        self.optimizer=Adafactor(model.parameters(),lr=hp.lr,beta1=hp.beta1,scale_parameter=hp.scale_parameter,relative_step=hp.relative_step)
        self.evals=Evals(hp.plots_dir,'standard' if not lm else 'lm')
    def set_model(self,model):
        '''
            Sets the model to be runned
            
            Inputs:
                model: ASRNN; the model this class runs
            Outputs:
                None
        '''
        self.__init__(model,self.hp)
    def save_model(self,finetuning=False):
        '''
            Saves the model
            
            Inputs:
                finetuning: bool; whether the training is runned during the finetuning phase
            Outputs:
                None
        '''
        if finetuning:
            torch.save(self.model.state_dict(), self.hp.finetuning_model_path)
        else:
            torch.save(self.model.state_dict(), self.hp.standard_model_path if not self.lm else self.hp.lm_model_path)
    def load_model(self,model_path):
        '''
            Loads pretrained weights of the conferms into the current models conformer.
            
            Inputs:
                model_path: str; path to the pretrained model weights
            Outputs:
                None
        '''
        pretrained_dict=torch.load(model_path,map_location='cpu')
        self.model.cpu()
        model_dict=self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict={k:v for k,v in pretrained_dict.items() if 'conformer' in k}
        pretrained_dict.update({k:v for k,v in model_dict.items() if k not in pretrained_dict})
        self.model.load_state_dict(pretrained_dict)
        self.model.to(self.hp.device)
    def fuse_models(self,lm_path):
        '''
            Fuses the language model to the current model
            
            Inputs:
                lm_path: str; path to the language model
            Outputs:
                None
        '''
        lm_dict=torch.load(lm_path)
        model_dict=self.model.state_dict()
        lm_mhas=[x for x in lm_dict.keys() if 'mhas' in x]
        model_mhas=[x for x in model_dict.keys() if '.mhsa.' in x]
        fuse_dict={}
        for i in range(self.hp.lm_in_N):
            curr_lm_mhas=[x for x in lm_mhas if '.'+str(i)+'.' in x and 'input' in x and 'proj' in x]
            curr_model_mhas=[x for x in model_mhas if '.'+str(i)+'.' in x and 'proj' in x]
            curr_dict={curr_model_mhas[j]:model_dict[curr_model_mhas[j]]+lm_dict[curr_lm_mhas[j]] for j in range(len(curr_model_mhas))}
            curr_lm_mhas=[x for x in lm_mhas if '.'+str(i)+'.' in x and 'output' in x and 'proj' in x and 'mask' not in x]
            curr_model_mhas=[x for x in model_mhas if '.'+str(self.hp.n_conformers-i-1)+'.' in x and 'proj' in x]
            curr_dict.update({curr_model_mhas[j]:model_dict[curr_model_mhas[j]]+lm_dict[curr_lm_mhas[j]] for j in range(len(curr_model_mhas))})
            fuse_dict.update(curr_dict)
        fuse_dict.update({k:v for k,v in model_dict.items() if k not in fuse_dict})
        self.model.load_state_dict(fuse_dict)
    def train(self,train_set,epochs,SpecAugment=False,use_mix=False,finetuning=False):
        '''
            Trains the class model
            
            Inputs:
                train_set: SpeechCommands; the dataset
                epochs: int; number of epochs
                SpecAugment: bool; whether the specaugmentaion is to be performed on the dataset; Default: False
                use_mix: bool; whether to train on the mixed the dataset. This is used during the finetuning phase (last phase). Default: False
                finetuning: bool; whether the model is trained during the finetuning phase. Default False
            Outputs:
                None
        '''
        losses=[]
        metrics=[]
        val_losses=[]
        val_metrics=[]
        metric_type='wer' if not self.lm else 'ppw'
        if use_mix:
            dataset_type='mix'
        else:
            dataset_type='train'
        train_size=ceil(len(train_set.idxes[dataset_type])/self.hp.batch_size)
        for epoch in range(epochs):
            self.model.train()
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            train_set.shuffle(dataset_type)
            eloss=[]
            emetric=[]
            with tqdm(total=train_size,unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
                t.set_description('Epoch '+str(epoch+1)+'/'+str(epochs))
                for i in range(train_size):
                    batch=train_set.get_batch(i,dataset_type)
                    inbatch=batch['input']['mels']
                    input_lens=batch['input']['tau']
                    target=batch['target']['transcripts']
                    target_lens=batch['target']['lens']
                    unpadded_len=batch['unpadded_len']
                    self.optimizer.zero_grad()
                    logits,output_lengths=self.model(inbatch,input_lens,SpecAugment if not self.lm else target,finetuning=finetuning)
                    output_lengths=nn.functional.pad(output_lengths,(0,self.hp.batch_size-output_lengths.shape[0]))
                    loss=self.loss(logits.transpose(0,1),target,output_lengths,target_lens)
                    curr_loss=loss.item()
                    loss.backward()
                    self.optimizer.step()
                        
                    eloss.append(curr_loss)
                    if not self.lm:
                        predicted=torch.argmax(logits,dim=-1)
                        target=train_set.vocab.decode(target)
                        predicted=train_set.vocab.decode(predicted)
                        predicted=[predicted[i] for i in range(min(len(target),len(predicted))) if target[i]!='']
                        target=[x for x in target if x!='']
                        metric=wer(target,predicted)*100
                    else:
                        metric=torch.exp(loss).item()
                    emetric.append(metric)
                    t.postfix='loss: '+str(round(curr_loss,4))+', '+metric_type+': '+str(round(metric,2))
                    t.update(1)
                eloss=[100 if isnan(x) else x for x in eloss]
                eloss=mean(eloss)
                emetric=mean(emetric)
                losses.append(eloss)
                metrics.append(emetric)
                t.postfix='loss: '+str(round(eloss,4))+', '+metric_type+': '+str(round(emetric,2))
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
            val_loss,val_metric=self.test(train_set,'validation',finetuning=finetuning)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)
            print()
        self.evals.plot(losses,val_losses)
        if self.lm:
            self.evals.plot(metrics,val_metrics,'Perplexity per word','PPW')
        else:
            self.evals.plot(metrics,val_metrics,'Word Error Rate','WER (%)')
        self.save_model(finetuning)
    def test(self,test_set,dataset_type='test',heatmap=False,finetuning=False):
        '''
            It test the model on the given dataset, validation or test.
            
            Inputs:
                test_set: SpeechCommands; the dataset
                dataset_type: type of dataset, validation or test
                heatmap: bool; whether a heatmap of the results must be plot; Default: False
                finetuning: bool; whether the model is trained during the finetuning phase. Default: False
            Outputs:
                eloss: float; the loss of the test
                emetric: float; the word error rate of the test
        '''
        self.model.eval()
        testset_size=ceil(len(test_set.idxes[dataset_type])/self.hp.batch_size)
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        metric_type='wer' if not self.lm else 'ppw'
        y_true=[]
        y_pred=[]
        with tqdm(total=testset_size,unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
            t.set_description(dataset_type[0].upper()+dataset_type[1:])
            eloss=[]
            emetric=[]
            with torch.no_grad():
                for i in range(testset_size):
                    batch=test_set.get_batch(i,dataset_type)
                    inbatch=batch['input']['mels']
                    input_lens=batch['input']['tau']
                    target=batch['target']['transcripts']
                    target_lens=batch['target']['lens']
                    unpadded_len=batch['unpadded_len']
                    logits,output_lens=self.model(inbatch,input_lens,lm=test_set.get_lm if self.hp.lm else None,finetuning=finetuning)
                    output_lens=nn.functional.pad(output_lens,(0,self.hp.batch_size-output_lens.shape[0]))
                    loss=self.loss(logits.transpose(0,1),target,output_lens,target_lens)
                    curr_loss=loss.item()
                    eloss.append(curr_loss)
                    if not self.lm:
                        predicted=torch.argmax(logits,dim=-1)
                        target=test_set.vocab.decode(target)
                        predicted=test_set.vocab.decode(predicted)
                        predicted=[predicted[i] for i in range(min(len(target),len(predicted))) if target[i]!='']
                        target=[x for x in target if x!='']
                        metric=wer(target,predicted)*100
                        if heatmap:
                            y_true+=target
                            y_pred+=predicted
                        try:
                            with open(os.path.join(self.hp.plots_dir,'pred_tgt.txt'),'w+') as f:
                                f.write(dataset_type[0].upper()+dataset_type[1:]+':\n\n'+predicted[0]+'\n\n'+target[0]+'\n\n'+str(round(metric,2)))
                        except:
                            pass
                    else:
                        metric=torch.exp(loss).item()
                    emetric.append(metric)
                    t.postfix=' loss: '+str(round(curr_loss,4))+', '+metric_type+': '+str(round(metric,2))
                    t.update(1)
                eloss=[100 if isnan(x) else x for x in eloss]
                eloss=mean(eloss)
                emetric=mean(emetric)
                t.postfix=' loss: '+str(round(eloss,4))+', '+metric_type+': '+str(round(emetric,2))
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
                if heatmap:
                    self.evals.heatmap(dataset_type,y_true,y_pred,finetuning=True)
                    self.evals.heatmap(dataset_type,y_true,y_pred,finetuning=True,norm='true')
                return eloss,emetric
    def generate_labels(self,dataset):
        '''
            Generates labels for the given dataset
            
            Inputs:
                dataset: SpeechCommands; the given dataset
            Outputs:
                targets: List; the generated labels
        '''
        self.model.eval()
        dataset_size=ceil(len(dataset.idxes['pretrain'])/self.hp.batch_size)
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        with tqdm(total=dataset_size,unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
            t.set_description('Generating labels')
            targets=[]
            with torch.no_grad():
                for i in range(dataset_size):
                    batch=dataset.get_batch(i,'pretrain')['input']
                    mels=batch['mels']
                    input_lengths=batch['tau']
                    self.optimizer.zero_grad()
                    logits,_=self.model(mels,input_lengths)
                    predicted=self.model.predict(logits)
                    predicted=dataset.vocab.decode(predicted)
                    targets+=predicted
                    t.update(1)
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        print()
        return targets

