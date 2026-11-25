import torch
from torch import nn
import torch.nn.functional as F
from importantonlymit import surrogate_spike, PowerOfTwoQuantizer

class V2(nn.Module):
    def __init__(self,pool=10,width=96,multiscale=True,hierarchy=True,adaptive=True):
        super().__init__()
        self.quant=False
        self.hierarchy=hierarchy
        self.adaptive=adaptive
        self.stems=nn.ModuleList([nn.Conv1d(1,c,k,padding=k//2) for c,k in ([(8,3),(8,5),(8,9)] if multiscale else [(24,5)])])
        self.conv=nn.Conv1d(24,24,3,padding=1)
        self.q=nn.Conv1d(24,16,1); self.k=nn.Conv1d(24,16,1); self.v=nn.Conv1d(24,24,1)
        self.pool=pool
        self.proj=nn.Linear(24*pool+6,width)
        self.head=nn.Linear(width,4)
        self.binary=nn.Linear(width,2)
        self.special=nn.Linear(width,3)
        self.threshold=nn.Parameter(torch.ones(5),requires_grad=adaptive)
    def affine(self,l,x):
        w=l.weight
        if self.quant:
            q=torch.sign(w)*torch.pow(2.,torch.round(torch.log2(w.abs().clamp_min(2**-7))).clamp(-7,7))
            q=torch.where(w.abs()<2**-8,torch.zeros_like(q),q)
            w=w+(q-w).detach()
        if isinstance(l,nn.Conv1d): return F.conv1d(x,w,l.bias,padding=l.padding)
        return F.linear(x,w,l.bias)
    def forward(self,x,rr,steps=10):
        mem=[0]*5; exits=[]; total=0; activity=0
        def lif(i,z):
            th=self.threshold[i].clamp(.25,4)
            mem[i]=mem[i]*.875+z
            a=surrogate_spike(mem[i]-th)
            mem[i]=mem[i]-a*th
            return a
        stem=F.max_pool1d(torch.cat([self.affine(l,x) for l in self.stems],1),2)
        for t in range(steps):
            a=lif(0,stem)
            b=lif(1,F.max_pool1d(self.affine(self.conv,a),2))
            q=lif(2,self.affine(self.q,b)); k=lif(3,self.affine(self.k,b)); v=lif(4,self.affine(self.v,b))
            z=torch.bmm(q.transpose(1,2),torch.bmm(k,v.transpose(1,2))).transpose(1,2)/4
            h=F.relu(self.affine(self.proj,torch.cat([F.adaptive_avg_pool1d(z+b,self.pool).flatten(1),rr],1)))
            if self.hierarchy:
                g=F.log_softmax(self.affine(self.binary,h),1)
                p=F.log_softmax(self.affine(self.special,h),1)
                logits=torch.cat([g[:,:1],g[:,1:]+p],1)
            else: logits=self.affine(self.head,h)
            total=total+logits
            activity=activity+(a.mean()+b.mean()+q.mean()+k.mean()+v.mean())/5
            if t+1 in (3,6,steps): exits.append(total/(t+1))
        return exits,activity/steps

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(1,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(15))
        self.head=nn.Sequential(nn.Linear(966,128),nn.ReLU(),nn.Linear(128,4))
    def forward(self,x,rr):
        return self.head(torch.cat([self.features(x).flatten(1),rr],1))
