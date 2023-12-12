import pandas as pd
import numpy as np
import os
from seaborn import heatmap
import matplotlib.pyplot as plt

path='/nfs/mqc/Consensus_peak/ATAC/ENCODE_2022_Bulk/processed_data/bed_unzip/'
beds=[i for i in os.listdir(path) if i.endswith('.bed')]
for i in beds:
    df=pd.read_csv(path+i,sep='\t',header=None)
    df=df.drop_duplicates(subset=[1,2],keep='first')
    try:
        alldata=pd.concat([alldata,df],axis=0,ignore_index=True)
    except:
        alldata=df

        
def query(chrname,left1,right1,smallstep=20,uplimitrate=0.1,minpeakrate=0.1):
    '''
    input:
    chrname: string;chromosome name
    left1: int;left boundary
    right1: int;right boundary
    smallstep: int;small step length,assert((right1-left1)%smallstep==0)
    uplimitrate:float;the uplimit of peak length in 2nd step
    minpeakrate:float;in 3rd step,if the peak length is less than minpeakrate*(right1-left1),merge it with its neighbor peak
    
    output:
    newnamelist: list;new peak name list,like ['938020-938980','938980-939940']
    mat1: np.array;mat1[i,j] means p(i|j)
    mat2: np.array;co-expression matrix,mat2[i,j] means rate open from i to j
    
    other value:
    
    split1,sumsplit1: the start position of peak
    split2,sumsplit2: the end position of peak
    arealen: the length of peak
    
    '''
    assert((right1-left1)%smallstep==0)
    
    minpeaklen=(right1-left1)*minpeakrate//smallstep
    df=alldata[alldata[0]==chrname]
    df=alldata[(alldata[1]>=left1)&(alldata[2]<=right1)]
        
    split1=np.zeros((right1-left1)//smallstep).astype(int)
    split2=np.zeros((right1-left1)//smallstep).astype(int)
    arealen=np.ones((right1-left1)//smallstep).astype(int)
    mat1=np.zeros([(right1-left1)//smallstep,(right1-left1)//smallstep])
    for index,row in df.iterrows():
        lindex=((row[1]-left1+smallstep//2)//smallstep)
        rindex=((row[2]-left1+smallstep//2)//smallstep)
        t=np.zeros((right1-left1)//smallstep).astype(int)
        t[lindex:rindex+1]=1
        mat1+=np.outer(t,t)
        split1[((row[1]-left1+smallstep//2)//smallstep)]+=1
        split2[((row[2]-left1+smallstep//2)//smallstep)]+=1
        
    mat1=mat1/mat1.diagonal()
    
    name=[str(i) for i in range(len(mat1))]
    sumsplit1=split1
    sumsplit2=split2
    uplimit=int(df.shape[0]*uplimitrate)
    '''
    1st step: merge the peak which has high condition expression rate
    '''
    while True:
        flag=0
        for i in range(len(mat1)-1):
            if mat1[i,i+1]>0.8 and mat1[i+1,i]>0.8:
                flag=1
                name[i]=name[i]+'-'+name[i+1]
                mat1=np.delete(mat1,i+1,axis=0)
                mat1=np.delete(mat1,i+1,axis=1)
                sumsplit1[i]+=sumsplit1[i+1]
                sumsplit2[i]+=sumsplit2[i+1]
                name.pop(i+1)
                arealen[i]+=arealen[i+1]
                arealen=np.delete(arealen,i+1)
                sumsplit1=np.delete(sumsplit1,i+1)
                sumsplit2=np.delete(sumsplit2,i+1)
                break
        if flag==0:
            break
        
    '''
    2nd step: merge the peak which has high co-expression rate, the peak length should be less than uplimit
    '''
    while True:
        flag=0
        for i in range(len(mat1)-1):
            if mat1[i,i+1]>0.8 and sumsplit1[i]+sumsplit1[i+1]<uplimit:
                flag=1
                name[i]=name[i]+'-'+name[i+1]
                mat1=np.delete(mat1,i+1,axis=0)
                mat1=np.delete(mat1,i+1,axis=1)
                sumsplit1[i]+=sumsplit1[i+1]
                sumsplit2[i]+=sumsplit2[i+1]
                name.pop(i+1)
                arealen[i]+=arealen[i+1]
                arealen=np.delete(arealen,i+1)
                sumsplit1=np.delete(sumsplit1,i+1)
                sumsplit2=np.delete(sumsplit2,i+1)
                break
            if mat1[i+1,i]>0.8 and sumsplit2[i+1]+sumsplit2[i]<uplimit:
                flag=1
                name[i]=name[i]+'-'+name[i+1]
                mat1=np.delete(mat1,i,axis=0)
                mat1=np.delete(mat1,i,axis=1)
                sumsplit1[i]+=sumsplit1[i+1]
                sumsplit2[i]+=sumsplit2[i+1]
                sumsplit1=np.delete(sumsplit1,i+1)
                sumsplit2=np.delete(sumsplit2,i+1)
                name.pop(i+1)
                arealen[i]+=arealen[i+1]
                arealen=np.delete(arealen,i+1)
                break
        if flag==0:
            break
    '''
    3rd step: merge the peak which has small length
    '''
    while True:
        flag=0
        i=np.argmin(arealen)
        if arealen[i]<minpeaklen:
            if i!=len(arealen)-1 and (i==0 or arealen[i+1]<=arealen[i-1]):
                name[i]=name[i]+'-'+name[i+1]
                name.pop(i+1)
                arealen[i]+=arealen[i+1]
                arealen=np.delete(arealen,i+1)
            if i!=0 and (i==len(arealen)-1 or arealen[i-1]<arealen[i+1]):
                name[i-1]=name[i-1]+'-'+name[i]
                name.pop(i)
                arealen[i-1]+=arealen[i]
                arealen=np.delete(arealen,i)
        else:
            break
    
    comp=np.zeros((right1-left1)//smallstep).astype(int)
    start=0
    for i in range(len(arealen)):
        comp[start:arealen[i]+1]=i
        start=arealen[i]+1
        
    mat=np.zeros([len(arealen),len(arealen)])
    for i in beds:
        df=pd.read_csv(path+i,sep='\t',header=None)
        df=df.drop_duplicates(subset=[1,2],keep='first')
        df=df[(df[0]==chrname)&(df[1]>=left1)&(df[2]<=right1)]
        t=np.zeros((right1-left1)//smallstep).astype(int)
        for index,row in df.iterrows():
            lindex=((row[1]-left1+smallstep//2)//smallstep)
            rindex=((row[2]-left1+smallstep//2)//smallstep)
            t[comp[lindex]:comp[rindex]+1]=1
        mat+=np.outer(t,t)
            
    mat1=mat/mat.diagonal()
    mat2=mat/len(beds)
    
    newnamelist=[]
    for i in name:
        newname=str(int(i.split('-')[0])*smallstep+left1)+'-'+str((int(i.split('-')[-1])+1)*smallstep+left1)
        newnamelist.append(newname)
    return newnamelist,mat1,mat2

if False: 
    def query(chrname,left1,right1,smallstep=20,uplimitrate=0.1,minpeakrate=0.1):
        '''
        input:
        chrname: string;chromosome name
        left1: int;left boundary
        right1: int;right boundary
        smallstep: int;small step length,assert((right1-left1)%smallstep==0)
        uplimitrate:float;the uplimit of peak length in 2nd step
        minpeakrate:float;in 3rd step,if the peak length is less than minpeakrate*(right1-left1),merge it with its neighbor peak
        
        output:
        newnamelist: list;new peak name list,like ['938020-938980','938980-939940']
        mat2: np.array;co-expression matrix,mat2[i,j] means rate open from i to j
        
        other value:
        mat1: np.array;mat1[i,j] means p(i|j)
        split1,sumsplit1: the start position of peak
        split2,sumsplit2: the end position of peak
        arealen: the length of peak
        
        '''
        assert((right1-left1)%smallstep==0)
        
        minpeaklen=(right1-left1)*minpeakrate//smallstep
        df=alldata[alldata[0]==chrname]
        df=alldata[(alldata[1]>=left1)&(alldata[2]<=right1)]
            
        split1=np.zeros((right1-left1)//smallstep+1)
        split2=np.zeros((right1-left1)//smallstep+1)
        arealen=np.ones((right1-left1)//smallstep+1).astype(int)
        mat=np.zeros([(right1-left1)//smallstep+1,(right1-left1)//smallstep+1])
        for index,row in df.iterrows():
            lindex=((row[1]-left1+smallstep//2)//smallstep)
            rindex=((row[2]-left1+smallstep//2)//smallstep)
            for i in range(lindex,rindex+1):
                for j in range(lindex,rindex+1):
                    mat[i,j]+=1
            split1[((row[1]-left1+smallstep//2)//smallstep)]+=1
            split2[((row[2]-left1+smallstep//2)//smallstep)]+=1
            
        mat2=mat.copy()/len(beds)
        mat1=mat.copy()
        for i in range((right1-left1)//smallstep+1):
            norm=mat[i,i]
            if norm==0:
                norm=1
            for j in range((right1-left1)//smallstep+1):
                mat1[i,j]=mat[i,j]/norm
        name=[str(i) for i in range(len(mat))]
        sumsplit1=split1
        sumsplit2=split2
        uplimit=int(df.shape[0]*uplimitrate)
        '''
        1st step: merge the peak which has high condition expression rate
        '''
        while True:
            flag=0
            for i in range(len(mat1)-1):
                if mat1[i,i+1]>0.8 and mat1[i+1,i]>0.8:
                    flag=1
                    matt=mat2.copy()
                    for j in range(0,len(mat1)):
                        if j>i:
                            mat2[i,j]=matt[i+1,j]
                        elif j<i:
                            mat2[i,j]=matt[i,j]
                        else:
                            mat2[i,i]=matt[i,i]+matt[i+1,i+1]-matt[i,i+1]
                        mat2[j,i]=mat2[i,j]
                    name[i]=name[i]+'-'+name[i+1]
                    mat1=np.delete(mat1,i+1,axis=0)
                    mat2=np.delete(mat2,i+1,axis=0)
                    mat1=np.delete(mat1,i+1,axis=1)
                    mat2=np.delete(mat2,i+1,axis=1)
                    sumsplit1[i]+=sumsplit1[i+1]
                    sumsplit2[i]+=sumsplit2[i+1]
                    name.pop(i+1)
                    arealen[i]+=arealen[i+1]
                    arealen=np.delete(arealen,i+1)
                    sumsplit1=np.delete(sumsplit1,i+1)
                    sumsplit2=np.delete(sumsplit2,i+1)
                    break
            if flag==0:
                break
            
        '''
        2nd step: merge the peak which has high co-expression rate, the peak length should be less than uplimit
        '''
        while True:
            flag=0
            for i in range(len(mat1)-1):
                if mat1[i,i+1]>0.8 and sumsplit1[i]+sumsplit1[i+1]<uplimit:
                    flag=1
                    name[i]=name[i]+'-'+name[i+1]
                    matt=mat2.copy()
                    for j in range(0,len(mat1)):
                        if j>i:
                            mat2[i,j]=matt[i+1,j]
                        elif j<i:
                            mat2[i,j]=matt[i,j]
                        else:
                            mat2[i,i]=matt[i,i]+matt[i+1,i+1]-matt[i,i+1]
                        mat2[j,i]=mat2[i,j]
                    mat1=np.delete(mat1,i+1,axis=0)
                    mat1=np.delete(mat1,i+1,axis=1)
                    mat2=np.delete(mat2,i+1,axis=0)
                    mat2=np.delete(mat2,i+1,axis=1)
                    sumsplit1[i]+=sumsplit1[i+1]
                    sumsplit2[i]+=sumsplit2[i+1]
                    name.pop(i+1)
                    arealen[i]+=arealen[i+1]
                    arealen=np.delete(arealen,i+1)
                    sumsplit1=np.delete(sumsplit1,i+1)
                    sumsplit2=np.delete(sumsplit2,i+1)
                    break
                if mat1[i+1,i]>0.8 and sumsplit2[i+1]+sumsplit2[i]<uplimit:
                    flag=1
                    name[i]=name[i]+'-'+name[i+1]
                    matt=mat2.copy()
                    for j in range(0,len(mat1)):
                        if j>i:
                            mat2[i,j]=matt[i+1,j]
                        elif j<i:
                            mat2[i,j]=matt[i,j]
                        else:
                            mat2[i,i]=matt[i,i]+matt[i+1,i+1]-matt[i,i+1]
                        mat2[j,i]=mat2[i,j]
                    mat1=np.delete(mat1,i,axis=0)
                    mat2=np.delete(mat2,i+1,axis=0)
                    mat1=np.delete(mat1,i,axis=1)
                    mat2=np.delete(mat2,i+1,axis=1)
                    sumsplit1[i]+=sumsplit1[i+1]
                    sumsplit2[i]+=sumsplit2[i+1]
                    sumsplit1=np.delete(sumsplit1,i+1)
                    sumsplit2=np.delete(sumsplit2,i+1)
                    name.pop(i+1)
                    arealen[i]+=arealen[i+1]
                    arealen=np.delete(arealen,i+1)
                    break
            if flag==0:
                break
        '''
        3rd step: merge the peak which has small length
        '''
        while True:
            flag=0
            i=np.argmin(arealen)
            if arealen[i]<minpeaklen:
                if i!=len(arealen)-1 and (i==0 or arealen[i+1]<=arealen[i-1]):
                    name[i]=name[i]+'-'+name[i+1]
                    name.pop(i+1)
                    matt=mat2.copy()
                    for j in range(0,len(mat2)):
                        if j>i:
                            mat2[i,j]=matt[i+1,j]
                        elif j<i:
                            mat2[i,j]=matt[i,j]
                        else:
                            mat2[i,i]=matt[i,i]+matt[i+1,i+1]-matt[i,i+1]
                        mat2[j,i]=mat2[i,j]
                    mat2=np.delete(mat2,i+1,axis=0)
                    mat2=np.delete(mat2,i+1,axis=1)
                    arealen[i]+=arealen[i+1]
                    arealen=np.delete(arealen,i+1)
                if i!=0 and (i==len(arealen)-1 or arealen[i-1]<arealen[i+1]):
                    name[i-1]=name[i-1]+'-'+name[i]
                    name.pop(i)
                    matt=mat2.copy()
                    for j in range(0,len(mat2)):
                        if j>i-1:
                            mat2[i-1,j]=matt[i,j]
                        elif j<i-1:
                            mat2[i-1,j]=matt[i-1,j]
                        else:
                            mat2[i-1,i-1]=matt[i-1,i-1]+matt[i,i]-matt[i-1,i]
                        mat2[j,i]=mat2[i,j]
                    mat2=np.delete(mat2,i,axis=0)
                    mat2=np.delete(mat2,i,axis=1)
                    arealen[i-1]+=arealen[i]
                    arealen=np.delete(arealen,i)
            else:
                break
        
        newnamelist=[]
        for i in name:
            newname=str(int(i.split('-')[0])*smallstep+left1)+'-'+str((int(i.split('-')[-1])+1)*smallstep+left1)
            newnamelist.append(newname)
        return newnamelist,mat2

if __name__=='main':
    newnamelist,mat2=query('chr1',938020,938980,smallstep=10,minpeakrate=0.1)
    heatmap(mat2,xticklabels=newnamelist,yticklabels=newnamelist,annot=True, fmt=".2f", cmap="YlGnBu",ax=None)
    plt.savefig('graph/graph938020-938980.png')
