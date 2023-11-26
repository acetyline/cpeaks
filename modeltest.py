import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast as autocast
import copy
import torch.nn.functional as F
from typing import Callable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys
sys.path.append('./cPeaks/deepCNN')
import DeepCNN
backbonemodel=torch.load('./cPeaks/deepCNN/epoch5_20230822.pth').to(device)

class loaddata(Dataset):
    def __init__(self,ls):
        self.ls=ls
        
    def __len__(self):
        return len(self.ls)
    def __getitem__(self,index):
        return self.ls[index]

batch_size=32

import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class DenseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        batch_norm: bool = True,
        dropout: float = 0.2,
        activation_fn: Callable = nn.GELU(),
    ):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features, bias=use_bias)
        self.batch_norm = (
            nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        )
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation_fn(x)
        return x
    
class decn1d(nn.Module):
    def __init__(self,num_queries=5):
        super(decn1d,self).__init__()
        self.stem=backbonemodel.stem
        self.tower=backbonemodel.tower
        self.pre_bottle=backbonemodel.pre_bottleneck
        self.bottle = DenseLayer(
            in_features=5120,
            out_features=160,
            use_bias=True,
            batch_norm=True,
            dropout=0.2,
            activation_fn=nn.Identity(),
        )
        hidden_dim=32
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.box_embed=MLP(hidden_dim,hidden_dim,2,3)

    def forward(self,x):
        x=x.permute(0,2,1)
        x = self.stem(x)
        x = self.tower(x)
        x = self.pre_bottle(x)
        x = x.view(x.shape[0], -1)
        hs = self.bottle(x)
        hs=hs.view(32,5,32)
        hs=hs.unsqueeze(0)
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.box_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

lr=0.0001
detrmodel=decn1d()
optimizer = optim.Adam(detrmodel.parameters(), lr=lr)
detrmodel=detrmodel.to(device)

def midwtoxy(x):
    mid,w= x.unbind(-1)
    b = [(mid - 0.5 * w),(mid + 0.5 * w)]
    return torch.stack(b, dim=-1)

def xytomidw(a):
    x,y=a.unbind(-1)
    b=[((x+y)*0.5),(y-x)]
    return torch.stack(b,dim=-1)

def boxsize(x):
    return x[:,1]-x[:,0]

def box_iou(boxes1, boxes2):
    area1 = boxsize(boxes1)
    area2 = boxsize(boxes2)
    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    union = area1[:, None] + area2 - wh
    iou = wh / union
    return iou, union

def getgiou(box1,box2):
    iou, union = box_iou(box1, box2)
    lt = torch.min(box1[:, None, 0], box2[:, 0])
    rb = torch.max(box1[:, None, 1], box2[:, 1])
    wh = (rb - lt).clamp(min=0)  
    return iou - (wh - union) / wh
       
def gettarget(x):
    result=[]
    #"labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
    #          objects in the target) containing the class labels
    #"boxes": Tensor of dim [num_target_boxes, 2] containing the target box coordinates
    for i in x:
        dic={}
        
        if i[0]<0:
            dic['boxes']=torch.tensor([[0.0,1.0]]).to(device)
            dic['labels']=[1]
        else:
            dic['boxes']=torch.tensor([[i[0],i[1]]]).to(device)
            dic['labels']=[0]
        for j in range(1,5):
            if i[j*2]<0:
                break 
            dic['boxes']=torch.cat((dic['boxes'],torch.tensor([[i[j*2],i[j*2+1]]]).to(device)),axis=0)
            dic['labels'].append(0)
        dic['labels']=torch.tensor(dic['labels']).to(device)
        result.append(dic)
    return result

class Accuracy:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
    
    def update(self, output, target):
        pred = output.round()
        tp = (pred == 1) & (target == 1)
        tn = (pred != 1) & (target != 1)
        fp = (pred == 1) & (target != 1)
        fn = (pred != 1) & (target == 1)
        self.tp += tp.sum().item()
        self.tn += tn.sum().item()
        self.fp += fp.sum().item()
        self.fn += fn.sum().item()
    
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def confusion_matrix(self):
        return ([self.tn, self.fp, self.fn, self.tp],[(self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn),self.tp/(self.tp+self.fn),self.tp/(self.tp+self.fp)])



acc=Accuracy()
import sys
sys.path.append('./cPeaks/deepCNN')
import DeepCNN
model1=torch.load('/data1/zyb/model1/epochs4.pt')
model2=torch.load('/data1/zyb/model2/epochs4.pt').to(device)
model3=torch.load('./cPeaks/deepCNN/epoch5_20230419.pth').to(device)
for chrnum in range(1,23):
    print('chr'+str(chrnum)+':')
    try:
        del testset
        del testloader
    except:
        pass
    testset=torch.load('/data1/zyb/3/'+'chr'+str(chrnum)+'-test.pt')
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
    acc_with_reject=Accuracy()
    acc_no_reject=Accuracy()
    with torch.no_grad():
        model1=model1.to(device)
        model1.eval()
        model2.eval()
        for data,_,evaluatetarget,target in testloader:
            data=data.to(device)
            target=target.to(device)
            evaluatetarget=evaluatetarget.to(device)
            if data.shape[0]!=32:
                continue
            result=model1(data)
            result['pred_logits']=torch.softmax(result['pred_logits'],dim=-1)
            tobox=torch.zeros([batch_size,2000]).to(device)
            inputs=data.permute(0,2,1)
            outputs1 = model2(inputs).float().detach()
            outputs2=torch.round(outputs1)
            reject1result=model3(inputs)
            for i in range(batch_size):
                if reject1result[i][0]>=0.5:
                    acc_with_reject.update(tobox[i][501:1501],evaluatetarget[i][501:1501])
                    continue
                else:
                    for j in range(5):
                        if result["pred_logits"][i][j][0]<0.6:
                            pass
                        else:
                            l=result['pred_boxes'][i][j][0]-result['pred_boxes'][i][j][1]*0.5
                            r=result['pred_boxes'][i][j][0]+result['pred_boxes'][i][j][1]*0.5
                            l=max(int(2000*l),0)
                            r=min(int(2000*r)-1,1999)
                            while True:
                                if outputs1[i][l]>=0.5 or l>=r-1:
                                    break
                                else:
                                    l=l+1
                            while True:
                                if l==0 or outputs1[i][l-1]<=0.5  :
                                    break
                                else:
                                    l=l-1
                            while True:
                                if outputs1[i][r]>=0.5 or r<=l:
                                    break
                                else:
                                    r=r-1
                            while True:
                                if r>=1998 or outputs1[i][r+1]<=0.5  :
                                    break
                                else:
                                    r=r+1
                            tobox[i][l:r+1]=1.0
                acc_with_reject.update(tobox[i][501:1501],evaluatetarget[i][501:1501])
                acc_no_reject.update(tobox[i][501:1501],evaluatetarget[i][501:1501])
    print(acc_with_reject.confusion_matrix())
    print(acc_no_reject.confusion_matrix())
    with open('result.txt','a') as f:
        f.write(str(chrnum)+':\n')
        f.write(str(acc_with_reject.confusion_matrix())+'\n')
        f.write(str(acc_no_reject.confusion_matrix())+'\n')