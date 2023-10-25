import torch
from lib.standard.runner import Runner

class FineTune():
    '''
        class FineTune
        
        Applies the finetuning phase where the model is trained on the supervised dataset as well as the unsupervised dataset.
        
        Inputs:
            hp: HParams; class of hyperparameters
        Outputs:
            None
    '''
    def __init__(self,hp):
        self.hp=hp
    def fine_tuning(self,model,S,U):
        '''
            Application of the last phase, finetuning also known as noisy student training
            
            Inputs:
                model: ASRNN; the model to be used
                S: the dataset for supervised learning (has labels)
                U: the dataset for unsupervised learning (no labels)
            Outputs:
                runner: Runner; the runner class the handles the training and testing of the model
        '''
        runner=Runner(model,self.hp,lr=self.hp.ft_lr)
        runner.train(S,self.hp.ft_train_epochs,SpecAugment=True,finetuning=True)
        for epoch in range(self.hp.ft_epochs):
            if self.hp.nst:
                labels=runner.generate_labels(U)
                S.mix_datasets(U,labels)
            runner.train(S,self.hp.ft_train_epochs,SpecAugment=True,use_mix=self.hp.nst,finetuning=True)
        return runner