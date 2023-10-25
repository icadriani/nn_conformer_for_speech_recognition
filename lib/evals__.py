import os
import torch
import matplotlib.pyplot as plt
from statistics import mean

class Evals():
    def __init__(self,plots_dir,training_type):
        self.plots_dir=plots_dir
        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)
        self.training_type=training_type
    def plot(self,loss,val_loss=[],val_other=[],title='Loss',ylabel='loss'):
        plt.figure()
        epochs=[int(x) for x in range(1,len(loss)+1)]
        plt.plot(epochs,loss,'b-')
        if len(val_loss)>0:
            plt.plot(epochs,val_loss,'r-')
        if len(val_other)>0:
            plt.plot(epochs,val_other,'g-')
        plt.xlabel('epochs')
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.box()
        # plt.grid(False)
        filename=self.training_type+'_'+title.lower().replace(' ','_').replace('/','_')+'.pdf'
        if len(val_other)>0:
            plt.legend(['train','dev-clean','dev_other'])
        elif len(val_loss)>0:
            plt.legend(['train','dev'])
        # else:
            # filename='test_loss.pdf'
        plt.savefig(os.path.join(self.plots_dir,filename))
        plt.close()
   