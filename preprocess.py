import torch
from torch.utils.data import Dataset
import pandas as pd
import random
readpath='cpeaks/data/'
savepath='/data1/zyb/2/'
random.seed(2023)

batch_size=32

class getdata1(Dataset):#for trainset
    '''
    input: string,label,step
    string: string;DNA sequence,contains a,c,g,t,n
    label: string;is peak or not,contains 0,1;len(label)=len(string)
    step: int; distance between two adjacent starting sites
    
    output: res,res3,res2
    res: tensor,shape=2000;one-hot encoding of string
    res3: tensor,shape=20;peak position,even position is the starting point and odd position is the end point of peak, value between(0,1);fill in the rest space with -1.0
    res2: tensor,shape=2000;label
    '''
    def __init__(self,string,label,step=500):
        self.string=string
        self.label=label
        assert len(string)==len(label)
        self.step=step
        self.num=(len(self.string)+step-1-2000)//step
        
    def __len__(self):
        return self.num
    def __getitem__(self,index):
        assert index<=self.num
        tmp=''
        tmp2=''
        res=[]
        res2=[]
        if index*self.step+2000>len(self.string):
            tmp=self.string[-2000:]
            tmp2=self.label[-2000:]
        else:
            tmp=self.string[index*self.step:index*self.step+2000]
            tmp2=self.label[index*self.step:index*self.step+2000]
        for i in tmp:
            if i=='N' or i=='n':
                res.append([0.25,0.25,0.25,0.25])
            elif i=='A' or i=='a':
                res.append([1.0,0.0,0.0,0.0])
            elif i=='C' or i=='c':
                res.append([0.0,1.0,0.0,0.0])
            elif i=='G' or i=='g':
                res.append([0.0,0.0,1.0,0.0])
            else: 
                res.append([0.0,0.0,0.0,1.0])
        for i in tmp2:
            if int(i)==1:
                res2.append(1)
            else:
                res2.append(0)
        
        res3=torch.zeros(20)-1
        pos=0
        flag=1
        start=0
        for i in range(1,2000):
            if res2[i]==0:
                start=1
                continue
            if res2[i]>res2[i-1] and start==1:
                res3[pos]=float(i)/2000.0
                pos=pos+1
                flag=0
            if res2[i]<res2[i-1] and start==1:
                res3[pos]=float(i-1)/2000.0
                pos=pos+1
                flag=0
        if res2[1999]==1:
            res3[pos-1]=-1.0
            
        if flag==1 or pos==0:
            return None
        return torch.tensor(res),res3,torch.tensor(res2)

class getdata2(Dataset):#for testset or find new cpeaks
    '''
    input: string,label,step
    string: string;DNA sequence,contains a,c,g,t,n
    label: string;is peak or not,contains 0,1;len(label)=len(string)
    step: int; distance between two adjacent starting sites
    
    output: res,start,res2,flag
    res: tensor;one-hot encoding of string
    start: int;start position of string
    res2: tensor;label
    flag: tensor;whether the string contains peak;y:1,n:0
    '''
    def __init__(self,string,label,step=500):
        self.string=string
        self.label=label
        assert len(string)==len(label)
        self.step=step
        self.num=(len(self.string)+step-1-2000)//step
        
    def __len__(self):
        return self.num
    def __getitem__(self,index):
        assert index<=self.num
        tmp=''
        tmp2=''
        res=[]
        res2=[]
        if index*self.step+2000>len(self.string):
            tmp=self.string[-2000:]
            tmp2=self.label[-2000:]
            start=len(self.string-2000)
        else:
            tmp=self.string[index*self.step:index*self.step+2000]
            tmp2=self.label[index*self.step:index*self.step+2000]
            start=index*self.step
        for i in tmp:
            if i=='N' or i=='n':
                res.append([0.25,0.25,0.25,0.25])
            elif i=='A' or i=='a':
                res.append([1.0,0.0,0.0,0.0])
            elif i=='C' or i=='c':
                res.append([0.0,1.0,0.0,0.0])
            elif i=='G' or i=='g':
                res.append([0.0,0.0,1.0,0.0])
            else: 
                res.append([0.0,0.0,0.0,1.0])
        for i in tmp2:
            if int(i)==1:
                res2.append(1)
            else:
                res2.append(0)
        
        res3=torch.zeros(20)-1
        pos=0
        flag=0
        if res2[0]==1:
            res3[pos]=0.0
            pos=pos+1
            flag=1
        for i in range(1,2000):
            if res2[i]>res2[i-1] :
                res3[pos]=float(i)/2000.0
                pos=pos+1
                flag=1
            if res2[i]<res2[i-1] :
                res3[pos]=float(i-1)/2000.0
                pos=pos+1
                flag=1
        if res2[1999]==1:
            res3[pos]=1.0

        return torch.tensor(res),torch.tensor(start),torch.tensor(res2),torch.tensor(flag)
    
class loaddata(Dataset):
    #help save and load data
    def __init__(self,ls):
        self.ls=ls
    def __len__(self):
        return len(self.ls)
    def __getitem__(self,index):
        return self.ls[index]

if __name__=='__main__':
    ls=[i for i in range(1,23)]
    for i in ls:
        randnum=random.uniform(0,0.8)#randomly choose a 20% region to be testset
        filename='chr'+str(i)+'.fa' 
        labelname='chr'+str(i)+'.is'
        with open(readpath+filename) as f:
            f.readline()
            string=f.readline().replace('\n','')
        with open(readpath+labelname) as f:
            label=f.readline().replace('\n','')
        begin=int(len(label)*randnum)
        end=int(len(label)*(randnum+0.2))
        chrset1=getdata1(string[:begin],label[:begin])
        chrset2=getdata2(string[begin:end],label[begin:end],step=1000)
        chrset3=getdata1(string[end:],label[end:])
        trainset=[]
        for j in range(chrset1.__len__()):
            if chrset1[j]:
                trainset.append(chrset1[j])
        for j in range(chrset3.__len__()):
            if chrset3[j]:
                trainset.append(chrset3[j])
        testset=[]
        for j in range(chrset2.__len__()):
            if chrset2[j]:
                testset.append(chrset2[j])
        train=loaddata(trainset)
        test=loaddata(testset)
        torch.save(train,savepath+'chr'+str(i)+'-train.pt')
        torch.save(test,savepath+'chr'+str(i)+'-test.pt')
        
        del chrset1
        del chrset2
        del chrset3
        del trainset
        del testset
        del train
        del test
    