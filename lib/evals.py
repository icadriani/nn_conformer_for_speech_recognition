import os
import torch
import matplotlib.pyplot as plt
from statistics import mean
import matplotlib.colors as colors
import numpy as np
from sklearn.metrics import confusion_matrix


class Evals():
    '''
       class Evals:
       
       This class contains functions to plot results obtained from training and testing the model.
       
       Inputs:
            plots_dir: str; The folder in which the plots must be saved
            training_type: str; This information is embedded in the plots filename. If the model is pretrained this can be used to distinguish the plots from the normal training phase. It can also be used to divide plots of the performance of the model given through the use of different hyperparameters.
    '''
    def __init__(self,plots_dir,training_type):
        self.plots_dir=plots_dir
        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)
        self.training_type=training_type
    def plot(self,metric,val_metric=[],title='Loss',ylabel='loss'):
        '''
            Plots the training progress such as loss or word error rate (wer).
            
            Inputs:
                metric: list[float]; This list contains the metric or loss value of each epoch.
                val_metric: list[float]; This list contains the validation metric or loss value of each epoch; Default: [] (empty list).
                title: str; The plot title; Default: 'Loss'
                ylabel: str; the y axes label; Default: 'loss'
            Outputs:
                None                
        '''
        if len(metric)>1:
            plt.figure()
            epochs=[int(x) for x in range(1,len(metric)+1)]
            plt.plot(epochs,metric,'b-')
            if len(val_metric)>0:
                plt.plot(epochs,val_metric,'r-')
            # if len(val_other)>0:
            #     plt.plot(epochs,val_other,'g-')
            plt.xlabel('epochs')
            plt.ylabel(ylabel)
            plt.title(title)
            # plt.ylim(bottom=0)
            # plt.box()
            # plt.grid(False)
            filename=self.training_type+'_'+title.lower().replace(' ','_').replace('/','_')+'.pdf'
            # if len(val_other)>0:
            #     plt.legend(['train','dev-clean','dev_other'])
            if len(val_metric)>0:
                plt.legend(['train','validation'])
            # else:
                # filename='test_metric.pdf'
            plt.savefig(os.path.join(self.plots_dir,filename))
            # plt.close()
    def heatmap(self,model_name,y_true,y_pred,finetuning=False,norm=None):
        """  
            Computes the confusion matrix and visualizes it as a heatmap.
            
            Inputs:
                model_name:str; name of the model.   
                y_true:List[str]; list of the targets. 
                y_pred: List[str]; list of predictions.
                finetuning: bool; whether the model is trained during the finetuning phase. Default: False
                norm:str; confusion matrix normalization type. "true" to normalize along the true labels. Note that if this is not None the
                        heatmap will show the percentages by multiplying the normalize value by 100.
            Outputs:
                None
        """
        cm=confusion_matrix(y_true,y_pred,normalize=norm)
        if norm is not None:
            cm*=100
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(10)
        cmap=plt.cm.hot
        im = ax.imshow(cm,cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax,cmap=cmap,fraction=0.0475,pad=0.005)
        cbar.ax.set_ylabel('', rotation=-90, va="bottom")
        labels=np.unique(list(y_true)+list(y_pred))
        # print(labels)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        # rotation=45 if len(labels)>2 else 0
        ax.set_xticklabels(labels,rotation=90)
        ax.set_yticklabels(labels)
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         text=ax.text(j, i, round(cm[i, j],2), ha="center", va="center", color="black")
        ax.set_title('Heatmap')
        plt.tight_layout()
        normalized='nomalized' if norm is not None else 'cases'
        plt.savefig(os.path.join(self.plots_dir,('finetuning' if finetuning else '')+model_name+'_'+normalized+'_heatmap.png'))
        # plt.close()

        
   