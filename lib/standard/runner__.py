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
from lib.standard.wploss import WPLoss
# from torchmetrics import WordErrorRate
# from math import inf
import torch.nn as nn
from math import isnan

class Runner():
    def __init__(self,model,hp,lm=False):
        self.model=model
        self.hp=hp
        # self.standard_model_path=os.path.join(hp.model_dir,'standard_weights.pth')
        self.lm=lm
        # weights=torch.ones(hp.ntokens).to(hp.device)
        # weights[hp.space_idx]=0.1
        # if weights is not None:
        #     self.loss=CrossEntropyLoss(weight=weights.to(hp.device),ignore_index=0)
        # else:
        self.model.to(hp.device)
        print(hp.blank_idx)
        self.loss=CTCLoss(blank=hp.blank_idx,zero_infinity=True)#,reduction='none')
        # self.loss=CrossEntropyLoss(ignore_index=0)#,reduction='none')
        # self.loss=WPLoss(hp.device,ignore_index=0)#,reduction='none')
        # self.loss=CrossEntropyLoss(reduction='none')
        # self.optimizer=Adam(model.parameters(),lr=hp.lr)#.minimize(self.loss)
        self.optimizer=Adafactor(model.parameters(),lr=hp.lr,beta1=hp.beta1,scale_parameter=hp.scale_parameter,relative_step=hp.relative_step)#.minimize(self.loss)
        self.evals=Evals(hp.plots_dir,'standard' if not lm else 'lm')
        # self.wer=WordErrorRate()
        # os.system('rm '+os.path.join(hp.plots_dir,'pred_tgt.txt'))
    def set_model(self,model):
        self.__init__(model,self.hp)
    def save_model(self):
        torch.save(self.model.state_dict(), self.hp.standard_model_path if not self.lm else self.hp.lm_model_path)
    def load_model(self,model_path):
        # tnl=["fc.weight", "fc.bias", "linear_quantization.weight", "linear_quantization.bias"]
        pretrained_dict=torch.load(model_path,map_location='cpu')
        self.model.cpu()
        # self.model.load_state_dict()#,map_location=self.hp.device)
        model_dict=self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict) 
        pretrained_dict={k:v for k,v in pretrained_dict.items() if 'conformer' in k}
        pretrained_dict.update({k:v for k,v in model_dict.items() if k not in pretrained_dict})
        print(pretrained_dict.keys())
        self.model.load_state_dict(pretrained_dict)
        self.model.to(self.hp.device)
    # def fuse_models(self):
    #     pass
    def fuse_models(self,lm_path):
        # tnl=["fc.weight", "fc.bias", "linear_quantization.weight", "linear_quantization.bias"]
        lm_dict=torch.load(lm_path)
        # self.model.load_state_dict()#,map_location=self.hp.device)
        model_dict=self.model.state_dict()
        # if 'lm' in model_path:
        # print('----------')
        # print([x for x in lm_dict.keys() if 'mhas' in x and '0' in x])
        # print()
        # print([x for x in model_dict.keys() if 'mhsa' in x and '0' in x])
        # print('----------')
        lm_mhas=[x for x in lm_dict.keys() if 'mhas' in x]
        model_mhas=[x for x in model_dict.keys() if '.mhsa.' in x]
        fuse_dict={}
        for i in range(self.hp.lm_in_N):
            # print(i)
            curr_lm_mhas=[x for x in lm_mhas if '.'+str(i)+'.' in x and 'input' in x and 'proj' in x]
            curr_model_mhas=[x for x in model_mhas if '.'+str(i)+'.' in x and 'proj' in x]
            # for j in range(len(curr_model_mhas)):
            #     print(model_dict[curr_model_mhas[j]].shape,lm_dict[curr_lm_mhas[j]].shape)
            curr_dict={curr_model_mhas[j]:model_dict[curr_model_mhas[j]]+lm_dict[curr_lm_mhas[j]] for j in range(len(curr_model_mhas))}
            curr_lm_mhas=[x for x in lm_mhas if '.'+str(i)+'.' in x and 'output' in x and 'proj' in x and 'mask' not in x]
            curr_model_mhas=[x for x in model_mhas if '.'+str(self.hp.n_conformers-i-1)+'.' in x and 'proj' in x]
            # print(curr_lm_mhas)
            # print(curr_model_mhas)
            curr_dict.update({curr_model_mhas[j]:model_dict[curr_model_mhas[j]]+lm_dict[curr_lm_mhas[j]] for j in range(len(curr_model_mhas))})
            fuse_dict.update(curr_dict)
            # break
        # lm_dict = {k: v for k, v in lm_dict.items() if k in model_dict}
        # # model_dict.update(pretrained_dict) 
        fuse_dict.update({k:v for k,v in model_dict.items() if k not in fuse_dict})
        self.model.load_state_dict(fuse_dict)
    # def loss_fn(self,logits,target,input_lens,output_lens):
    #     # loss=self.loss(logits,target,input_lens,output_lens)
    #     loss=[(i+1)*self.loss(logits,target[target==i],input_lens,torch.where(target[target==i]<output_lens,len(target,output_lens)) for i in range(self.hp.ntokens)]
    #     loss=sum(loss)
    #     return loss
    def train(self,train_set,epochs,SpecAugment=False,use_mix=False):#,lm=False):
        # if not nst:
        #     epochs=self.hp.epochs
        # else:
        #     epochs=self.hp.nst_epochs
        losses=[]
        metrics=[]
        val_clean_losses=[]
        val_clean_metrics=[]
        val_other_losses=[]
        val_other_metrics=[]
        metric_type='wer' if not self.lm else 'ppw'
        if use_mix:
            dataset_type='mix'
        else:
            dataset_type='train'
        train_size=ceil(len(train_set.idxes[dataset_type])/self.hp.batch_size)
        # OutAugment=0.9
        for epoch in range(epochs):
            # with open(os.path.join(self.hp.plots_dir,'pred_tgt.txt'),'a+') as f:
            #     f.write('Train '+str(epoch+1))
            self.model.train()
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            # if not use_mix:
            # if epoch>0:
            train_set.shuffle(dataset_type)
            eloss=[]
            emetric=[]
            # previous_prediction=None
            with tqdm(total=train_size,unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
                t.set_description('Epoch '+str(epoch+1))
                for i in range(train_size):
                    batch=train_set.get_batch(i,dataset_type,SpecAugment)
                    inbatch=batch['input']['mels']
                    input_lens=batch['input']['tau']
                    target=batch['target']['transcripts']
                    target_lens=batch['target']['lens']
                    unpadded_len=batch['unpadded_len']
                    # if previous_prediction is None:
                    #     previous_prediction=torch.zeros(target.shape,dtype=torch.long).to(self.hp.device)
                    # no_space_target=target.clone().masked_fill_(mask=train_set.vocab.get_mask(target,self.hp.device),value=0)
                    # no_space_target=torch.cat([torch.zeros(no_space_target.shape[1:],dtype=torch.long).unsqueeze(0).to(self.hp.device),no_space_target],dim=0)[:-1]
                    # word_lengths=batch['word_lengths']
                    # tau=batch['tau']
                    self.optimizer.zero_grad()
                    # print(no_space_target)
                    # print(target)
                    # print('--------------')
                    logits,output_lengths=self.model(inbatch,input_lens,SpecAugment if not self.lm else target)#,lm=train_set.get_lm if self.hp.lm else None)
                    # print(logits)
                    # print(logits.shape)
                    # if self.lm:
                    # print(logits.flatten(0,1).shape,target.flatten(0,1).shape)
                    # loss=self.loss(logits.flatten(0,1),target.flatten(0,1))#.masked_fill(train_set.vocab.get_mask(target),value=0))
                    # print(loss)
                    # loss=torch.mean(loss,dim=-1)
                    # print(loss)
                    # target=torch.argmax(target,-1)#.flatten()
                    # print(logits.shape,target.shape)
                    # w=train_set.vocab.get_weight(logits.flatten(0,1))
                    # print(logits.flatten(0,1).shape,target.flatten().shape)
                    # train_set.vocab.group_words(target.flatten())
                    # word_lengths=train_set.vocab.get_word_lengths(target.flatten())
                    # print(logits.shape,logits.transpose(0,1).shape,target.shape,output_lengths.shape,target_lens.shape,input_lens.shape,)
                    # print(torch.max(output_lengths),torch.max(target_lens),torch.max(input_lens))
                    loss=self.loss(logits[:unpadded_len].transpose(0,1),target[:unpadded_len],output_lengths[:unpadded_len],target_lens[:unpadded_len])#.masked_fill(train_set.vocab.get_mask(target),value=0))
                    # print(loss)
                    # if torch.any(torch.isnan(loss).bool()) or torch.any(torch.isnan(torch.mean((loss/target_lens)[:unpadded_len])).bool()):
                    #     print(loss[:unpadded_len])
                    #     print(torch.isnan(torch.mean((loss/target_lens)[:unpadded_len])))
                    # loss=torch.nanmean(loss[:unpadded_len]/target_lens[:unpadded_len])
                    # print(loss)
                    # print(logits.shape,target.shape)
                    # print(logits.transpose(0,1).shape,target.shape)
                    curr_loss=loss.item()
                    # if isnan(curr_loss):
                    #     print(loss)
                    #     print(logits)
                    # if 
                    loss.backward()
                    self.optimizer.step()
                    # if i==train_size-1:
                        
                    eloss.append(curr_loss)
                    if not self.lm:
                        # print()
                        # previous_prediction=target
                        # print(logits.shape)
                        # predicted=self.model.predict(logits)
                        predicted=torch.argmax(logits,dim=-1)
                        # predicted=predicted.view((predicted.shape[1],predicted.shape[0]))
                        target=train_set.vocab.decode(target)
                        predicted=train_set.vocab.decode(predicted)#,len(target))
                        predicted=[predicted[i] for i in range(min(len(target),len(predicted))) if target[i]!='']
                        target=[x for x in target if x!='']
                        # predicted,target=train_set.process_output(predicted,target)
                        print()
                        print(target)
                        print(predicted)
                        metric=wer(target,predicted)*100#.item()
                        if i%50:
                            try:
                                with open(os.path.join(self.hp.plots_dir,'pred_tgt.txt'),'w+') as f:
                                    f.write('Train '+str(epoch+1)+'\n\n'+predicted[0]+'\n\n'+target[0]+'\n\n'+str(round(metric,2)))
                            except:
                                pass
                    else:
                        metric=torch.exp(loss).item()
                    emetric.append(metric)
                    t.postfix='loss: '+str(round(curr_loss,4))+', '+metric_type+': '+str(round(metric,2))#+', '+metric_type+'(torch): '+str(round(self.wer([predicted],[target]).item(),2))
                    t.update(1)
                # for l in eloss:
                #     print(l)
                eloss=[100 if isnan(x) else x for x in eloss]
                eloss=mean(eloss)
                emetric=mean(emetric)
                losses.append(eloss)
                metrics.append(emetric)
                t.postfix='loss: '+str(round(eloss,4))+', '+metric_type+': '+str(round(emetric,2))
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
            # with open(os.path.join(self.hp.plots_dir,'pred_tgt.txt'),'a+') as f:
            #     f.write('\nDev-clean '+str(epoch+1)+'\n')
            val_clean_loss,val_clean_metric=self.test(train_set,'dev-clean')
            val_clean_losses.append(val_clean_loss)
            val_clean_metrics.append(val_clean_metric)
            # with open(os.path.join(self.hp.plots_dir,'pred_tgt.txt'),'a+') as f:
            #     f.write('\nDev-other '+str(epoch+1)+'\n')
            val_other_loss,val_other_metric=self.test(train_set,'dev-other')
            val_other_losses.append(val_other_loss)
            val_other_metrics.append(val_other_metric)
            # OutAugment-=0.1
            try:
                self.hp.writer.add_scalars('Loss',{'train':eloss,'dev-clean':val_clean_loss,'dev-other':val_other_loss},epoch)
                if not self.hp.lm:
                    self.hp.writer.add_scalars('WER',{'train':emetric,'dev-clean':val_clean_metric,'dev-other':val_other_metric},epoch)
                else:
                    self.hp.writer.add_scalars('PPW',{'train':emetric,'dev-clean':val_clean_metric,'dev-other':val_other_metric},epoch)
            except:
                pass
            print()
        self.evals.plot(losses,val_clean_losses,val_other_losses)
        if self.lm:
            self.evals.plot(metrics,val_clean_metrics,val_other_metrics,'Perplexity per word','PPW')
        else:
            self.evals.plot(metrics,val_clean_metrics,val_other_metrics,'Word Error Rate','WER (%)')
        self.save_model()
        # self.writer.close()
    def test(self,test_set,dataset_type='test-clean',SpecAugment=False):#,lm=False):
        self.model.eval()
        testset_size=ceil(len(test_set.idxes[dataset_type])/self.hp.batch_size)
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        metric_type='wer' if not self.lm else 'ppw'
        # previous_prediction=None
        with tqdm(total=testset_size,unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
            # if dataset_type=='test':
            #     t.set_description('Test')
            # else:
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
                    # if previous_prediction is None:
                    #     previous_prediction=torch.zeros(target.shape,dtype=torch.long).to(self.hp.device)
                    # no_space_target=previous_prediction.clone().masked_fill_(mask=test_set.vocab.get_mask(previous_prediction,self.hp.device),value=0)
                    # word_lengths=batch['word_lengths']
                    # tau=batch['tau']
                    self.optimizer.zero_grad()
                    logits,output_lens=self.model(inbatch,input_lens,SpecAugment if not self.lm else target,lm=test_set.get_lm if self.hp.lm else None)
                    # if self.lm:
                    # target=torch.argmax(target,-1)#.flatten()
                    loss=5*self.loss(logits[:unpadded_len].transpose(0,1),target[:unpadded_len],output_lens[:unpadded_len],target_lens[:unpadded_len])
                    # loss=torch.nanmean((loss/(target_lens+1e-10))[:unpadded_len])
                    curr_loss=loss.item()
                    eloss.append(curr_loss)
                    if not self.lm:
                        # predicted=self.model.predict(logits)
                        # previous_prediction=predicted
                        predicted=torch.argmax(logits,dim=-1)
                        # predicted=predicted.view((predicted.shape[1],predicted.shape[0]))
                        target=test_set.vocab.decode(target)
                        predicted=test_set.vocab.decode(predicted)#,len(target))
                        predicted=[predicted[i] for i in range(min(len(target),len(predicted))) if target[i]!='']
                        target=[x for x in target if x!='']
                        metric=wer(target,predicted)*100#.item()
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
                return eloss,emetric
    def generate_labels(self,dataset,vocab):
        self.model.eval()
        dataset_size=ceil(len(dataset)/self.hp.batch_size)
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        with tqdm(total=dataset_size,unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
            # if dataset_type=='test':
            #     t.set_description('Test')
            # else:
            t.set_description('Generating labels')
            targets=[]
            with torch.no_grad():
                for i in range(dataset_size):
                    batch=dataset[i]
                    # inbatch=batch['input']
                    # target=batch['target']
                    # unpadded_len=batch['unpadded_len']
                    self.optimizer.zero_grad()
                    logits=self.model(batch)
                    # loss=self.loss(logits,target)
                    # curr_loss=loss.item()
                    # eloss.append(curr_loss)
                    predicted=self.model.predict(logits)
                    # target=test_set.vocab.decode(target)
                    predicted=vocab.decode(predicted)#,len(target))
                    # metric=self.wer(predicted,target).item()
                    # emetric.append(metric)
                    # targets.append(predicted)
                    targets+=predicted
                    # t.postfix=' loss: '+str(round(curr_loss,4))+', '+metric_type+': '+str(round(metric,2))
                    t.update(1)
                # eloss=mean(eloss)
                # emetric=mean(emetric)
                # t.postfix=' loss: '+str(round(eloss,4))+', '+metric_type+': '+str(round(emetric,2))
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        print()
        return targets

