import torch
import torch.nn as nn
from functools import reduce

class WPLoss(nn.Module):
    def __init__(self,device,ignore_index=-100):
        super(WPLoss,self).__init__()
        self.ignore_index=ignore_index
        self.device=device
        self.softmax=nn.Softmax(-1)
        self.celoss=nn.CrossEntropyLoss(ignore_index=ignore_index)#,reduction='none')
    def forward(self,logits,target,lengths):
        # lengths=[[x]*x for x in lengths]
        # lengths=reduce(lambda x,y: x+y,lengths)
        # lengths=torch.LongTensor(lengths).to(self.device)
        # print(target)
        # print()
        sm=self.softmax(logits)
        sm=torch.argmax(sm,dim=-1)
        # sm=torch.log(sm)
        # lengths=lengths.unsqueeze(1)
        # print(sm,torch.min(sm),torch.max(sm))
        # sm=torch.gather(sm,-1,lengths)
        # w=lengths/(1-sm)
        # mask=target.ge(self.ignore_index)
        # # logits=torch.gather(logits,0,target)
        # # logits=torch.masked_select(logits,mask)
        # # target=torch.masked_select(target,mask)
        # logits=logits*mask.expand(logits.shape)
        mask=target.gt(self.ignore_index)
        l=self.celoss(logits,target)
        # l=torch.masked_select(l,mask=mask)
        return l
        # l=torch.masked_select(l,mask=target.gt(self.ignore_index))
        idx=[0]
        for x in lengths:
            idx.append(x+idx[-1])
        # print(target)
        # print(idx)
        sm=1-torch.eq(sm,target)*1.0
        # print(target)
        # print(sm)
        # print(lengths)
        # print([len(sm[idx[i]+1:idx[i+1]]) for i in range(len(idx)-1)])
        w=[torch.mean(sm[idx[i]+1:idx[i+1]]) for i in range(len(idx)-1)]
        # print(torch.FloatTensor(w).to(self.device))
        w=[[50/w[i] if w[i].item()>0 else w[i]+1]*lengths[i] for i in range(len(idx)-1)]
        w=reduce(lambda x,y: x+y,w)
        w=torch.FloatTensor(w).to(self.device)
        # print(w)
        w=torch.masked_select(w,mask=mask)
        l=torch.masked_select(l,mask=mask)
        # print(l)
        # print(torch.sum(l[:lengths[0]]),lengths[0]/torch.sum(l[:lengths[0]]))
        # print(w)
        # print()
        l=torch.mean(w*l)
        # print(w*l)
        # print('---------------')
        # l=[torch.mean(l[idx[i]:idx[i+1]]) for i in range(len(idx)-1)]
        # l=torch.stack(l,dim=0)
        # l=torch.mean(l)
        return l
        