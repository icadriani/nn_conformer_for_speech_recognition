import torch
import torch.nn as nn
from torch.optim import Adam
from statistics import mean
from lib.evals import Evals
from tqdm import tqdm
from colorama import Fore
from math import ceil
import os
from lib.pretraining.loss import PretrainingLoss

class PreTrainRunner():
    '''
        class PreTrainRunner
        
        This class handles the given model
        
        Inputs:
            model: NN; the pretraining model
            hp: HParams; the class of hyperparameters
        Outputs:
            None
    '''
    def __init__(self,model,hp):
        self.model=model
        self.hp=hp
        # self.model_path=os.path.join(hp.model_dir,'pretrained_weights.pth')
        self.loss=PretrainingLoss(hp)
        self.optimizer=Adam(model.parameters(),lr=hp.pretraining_lr)#.minimize(self.loss)
        self.model.to(hp.device)
        self.evals=Evals(hp.plots_dir,'pretraining')
    def save_model(self):
        '''
            Saves the model weights
            
            Inputs:
                None
            Outputs:
                None
        '''
        torch.save(self.model.state_dict(), self.hp.pretrained_model_path)
    def load_model(self,model_path):
        '''
            Loads the model weights
            
            Inputs:
                model_path: str; the path of the stored weights
            Outputs:
                None
        '''
        self.model.load_state_dict(torch.load(model_path))#,map_location=self.hp.device)
    def train(self,train_set):
        '''
            Trains the pretraining model
            
            Inputs:
                train_set: SPEECHCOMMADS; the dataset used for the pretraining
            Outputs:
                None
        '''
        losses=[]
        #val_losses=[]
        trainset_size=ceil(len(train_set.data['pretrain'])/self.hp.batch_size)
        for epoch in range(self.hp.pretraining_epochs):
            self.model.train()
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            train_set.shuffle()
            eloss=[]
            with tqdm(total=trainset_size,unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
                t.set_description('Epoch '+str(epoch+1))
                for i in range(trainset_size):
                    inbatch=train_set.get_batch(i,'pretrain')['input']
                    mels=inbatch['mels']
                    input_lengths=inbatch['tau']
                    # print(mels)
                    #inbatch=batch['input']
                    #target=batch['target']
                    #unpadded_len=batch['unpadded_len']
                    self.optimizer.zero_grad()
                    logits,target=self.model(mels,input_lengths)
                    # print(logits)
                    loss=self.loss(logits,target)
                    loss.backward()
                    self.optimizer.step()
                    curr_loss=loss.item()
                    eloss.append(curr_loss)
                    t.postfix='loss: '+str(round(curr_loss,4))
                    t.update(1)
                # break
                eloss=mean(eloss)
                losses.append(eloss)
                t.postfix='loss: '+str(round(eloss,4))#+', moe: '+str(round(emoe,4))+', mrde: '+str(round(emrde,4))#+' rmse: '+str(round(ermse,4))
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
            #val_loss=self.test(train_set,'val')
            #val_losses.append(val_loss)
            #print()
        self.evals.plot(losses)
        self.save_model()
        
    #def test(self,test_set,dataset_type='test'):
    #    self.model.eval()
    #    testset_size=test_set.test_size if dataset_type=='test' else test_set.val_size
    #    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
    #    with tqdm(total=testset_size,unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
    #        if dataset_type=='test':
    #            t.set_description('Test')
    #            else:
    #            t.set_description('Validation')
    #        eloss=[]
    #        with torch.no_grad():
    #            for i in range(testset_size):
    #                inbatch=test_set.get_batch(i,dataset_type)
    #                #inbatch=batch['input']
    #                #target=batch['target']
    #                #unpadded_len=batch['unpadded_len']
    #                self.optimizer.zero_grad()
    #                logits,target=self.model(inbatch)
    #                loss=self.loss(logits,target)
    #                curr_loss=loss.item()
    #                eloss.append(curr_loss)
    #                t.postfix=' loss: '+str(round(curr_loss,4))#+', moe: '+str(round(moe,4))+', mrde: '+str(round(mrde,4))#+' rmse: '+str(round(rmse,4))
    #                t.update(1)
    #            eloss=mean(eloss)
    #            t.postfix=' loss: '+str(round(eloss,4))#+', moe: '+str(round(emoe,4))+', mrde: '+str(round(emrde,4))#+' rmse: '+str(round(ermse,4))
    #            t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
    #            return eloss

