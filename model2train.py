import os
import torch
import random
import numpy as np
import sys
sys.path.append('./cPeaks/deepCNN')
from DeepCNN import DeepCNN,setup_seed
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
setup_seed(2023)
lr = 0.001
batch_size = 32
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = DeepCNN(n_cells=2000,dropout=0.1)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

from torch.utils.data import Dataset,DataLoader
class loaddata(Dataset):
    def __init__(self,ls):
        self.ls=ls
        
    def __len__(self):
        return len(self.ls)
    def __getitem__(self,index):
        return self.ls[index]
    
num_epochs = 5
trainpath='/data1/zyb/2/'
for epochs in range(num_epochs):
    for chrnum in range(1,23):
        try:
            del trainset
            del trainloader
        except:
            pass
        trainset=torch.load(trainpath+'chr'+str(chrnum)+'-train.pt')
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
        for data,_,target in trainloader:
            data=data.to(device)
            target=target.to(device)
            data=data.permute(0,2,1)
            optimizer.zero_grad()
            outputs = model(data).float()
            loss = criterion(outputs, target.float())
            loss.backward()
            optimizer.step()
    torch.save(model,'/data1/zyb/model2/epochs'+str(epochs)+'epoch.pt')