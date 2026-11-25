"""Experimental compact CA-SSAN; floating-point QAT simulation, not RTL."""
import torch
from torch import nn
import torch.nn.functional as F
from importantonlymit import surrogate_spike, PowerOfTwoQuantizer

class CASSAN(nn.Module):
    def __init__(self, compact=True, quant=True, shift=True):
        super().__init__()
        self.quant, self.shift = quant, shift
        self.qat = PowerOfTwoQuantizer()
        self.c1 = nn.Conv1d(1,16,5,padding=2)
        self.c2 = nn.Conv1d(16,24,3,padding=1)
        self.q = nn.Conv1d(24,16,1)
        self.k = nn.Conv1d(24,16,1)
        self.v = nn.Conv1d(24,24,1)
        self.pool = nn.AdaptiveAvgPool1d(5 if compact else 45)
        self.proj = nn.Linear(120 if compact else 1080,64 if compact else 480)
        self.head = nn.Sequential(nn.Linear(70 if compact else 486,32),nn.ReLU(),nn.Linear(32,4))

    def conv(self, layer, x):
        return F.conv1d(x,self.qat(layer.weight) if self.quant else layer.weight,
                        layer.bias,padding=layer.padding)
    def linear(self, layer, x):
        return F.linear(x,self.qat(layer.weight) if self.quant else layer.weight,layer.bias)
    def lif(self, x, m):
        if m is None: m = torch.zeros_like(x)
        m = (m-m/8 if self.shift else .9*m)+x
        s = surrogate_spike(m-1)
        return s,m-s

    def step(self,x,rr,state):
        a,m1=self.lif(F.max_pool1d(self.conv(self.c1,x),2),state[0])
        b,m2=self.lif(F.max_pool1d(self.conv(self.c2,a),2),state[1])
        q,mq=self.lif(self.conv(self.q,b),state[2])
        k,mk=self.lif(self.conv(self.k,b),state[3])
        v,mv=self.lif(self.conv(self.v,b),state[4])
        z=torch.bmm(q.transpose(1,2),torch.bmm(k,v.transpose(1,2))).transpose(1,2)/4
        f=self.linear(self.proj,self.pool(z+b).flatten(1))
        h=F.relu(self.linear(self.head[0],torch.cat([f,rr],1)))
        return self.linear(self.head[2],h),h,[m1,m2,mq,mk,mv],(a.mean()+b.mean()+q.mean()+k.mean()+v.mean())/5

    def forward(self,x,rr):
        state=[None]*5
        total=0
        exits=[]
        activity=0
        for t in range(1,11):
            z,h,state,s=self.step(x,rr,state)
            total=total+z
            activity=activity+s
            if t in (3,6,10): exits.append(total/t)
        return exits,h,activity/10

    @torch.no_grad()
    def adaptive(self,x,rr,thresholds):
        # Actual active-batch compaction; thresholds are probability margins.
        idx=torch.arange(len(x),device=x.device)
        result=x.new_zeros((len(x),4))
        steps=torch.zeros(len(x),dtype=torch.long,device=x.device)
        state=[None]*5
        total=0
        for t in range(1,11):
            z,_,state,_=self.step(x,rr,state)
            total=total+z
            if t not in (3,6,10): continue
            logits=total/t
            p=logits.softmax(1)
            top,cls=p.topk(2,dim=1)
            done=(top[:,0]-top[:,1]>=thresholds[cls[:,0]]) if t<10 else torch.ones(len(x),dtype=torch.bool,device=x.device)
            result[idx[done]]=logits[done]
            steps[idx[done]]=t
            keep=~done
            if not keep.any(): break
            idx,x,rr,total=idx[keep],x[keep],rr[keep],total[keep]
            state=[m[keep] for m in state]
        return result,steps
