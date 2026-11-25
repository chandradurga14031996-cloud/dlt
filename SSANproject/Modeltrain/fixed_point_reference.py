"""Integer reference for compact QAT CA-SSAN. Q16.16, int64 sums, int32 saturation.
Confidence uses host softmax; this is not yet an integer hardware exit controller.
"""
import json
from pathlib import Path
import numpy as np
S=65536
LO,HI=-(1<<31),(1<<31)-1
LAYERS={'c1':(16,1,5),'c2':(24,16,3),'q':(16,24,1),'k':(16,24,1),'v':(24,24,1),'proj':(64,120),'head.0':(32,70),'head.2':(4,32)}

def sat(x): return np.clip(x,LO,HI).astype(np.int64)
def fixed(x):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite value')
    return sat(np.rint(x*S))
def read_hex(path,bits):
    values=[int(s,16) for s in Path(path).read_text().split()]
    if any(v<0 or v>=1<<bits for v in values): raise ValueError('Word out of range')
    a=np.array(values,dtype=np.int64)
    return np.where(a>=1<<(bits-1),a-(1<<bits),a) if bits==32 else a
def decode(code):
    if code==128: return 0,0
    if code&128: raise ValueError('Invalid zero code')
    e=code&63
    e=e-64 if e>=32 else e
    if not -7<=e<=7: raise ValueError('Invalid exponent')
    return (-1 if code&64 else 1),e

def dot(w,x,b):
    # x: input features by positions. Shift each product before summing.
    out=np.broadcast_to(b[:,None],(len(b),x.shape[1])).copy()
    for code in np.unique(w):
        sign,e=decode(int(code))
        if not sign: continue
        term=(x<<e) if e>=0 else (x>>(-e))
        out+=sign*((w==code).astype(np.int64)@term)
    return sat(out)

class FixedReference:
    def __init__(self,directory):
        p=Path(directory)
        self.meta=json.loads((p/'manifest.json').read_text())
        if self.meta['format']!='ca-ssan-q16-v1': raise ValueError('Unknown format')
        self.layers={}
        for name,shape in LAYERS.items():
            f=self.meta['layers'][name]
            w=read_hex(p/f['weights'],8).reshape(shape)
            for code in np.unique(w): decode(int(code))
            b=read_hex(p/f['bias'],32).reshape(shape[0])
            self.layers[name]=(w,b)
        self.threshold=np.array(self.meta['threshold_q16'],dtype=np.int64)/S
    def conv(self,name,x):
        w,b=self.layers[name]; k=w.shape[2]
        padded=np.pad(x,((0,0),(k//2,k//2)))
        windows=np.lib.stride_tricks.sliding_window_view(padded,k,axis=1).transpose(0,2,1)
        return dot(w.reshape(len(b),-1),windows.reshape(-1,x.shape[1]),b)
    def linear(self,name,x):
        w,b=self.layers[name]
        return dot(w,x.reshape(-1,1),b)[:,0]
    def lif(self,x,m):
        if m is None: m=np.zeros_like(x)
        m=sat(m-(m>>3)+x)
        s=(m>=S).astype(np.int64)
        return s,sat(m-s*S)
    def infer(self,wave,rr,adaptive=True):
        x=np.asarray(wave,dtype=np.int64).reshape(1,180)
        rr=np.asarray(rr,dtype=np.int64).reshape(6)
        m=[None]*5; total=np.zeros(4,dtype=np.int64)
        for t in range(1,11):
            a,m[0]=self.lif(self.conv('c1',x).reshape(16,90,2).max(2),m[0])
            b,m[1]=self.lif(self.conv('c2',a*S).reshape(24,45,2).max(2),m[1])
            q,m[2]=self.lif(self.conv('q',b*S),m[2])
            k,m[3]=self.lif(self.conv('k',b*S),m[3])
            v,m[4]=self.lif(self.conv('v',b*S),m[4])
            z=sat((q.T@(k@v.T)).T*(S//4)+b*S)
            pooled=z.reshape(24,5,9).sum(2)//9
            f=self.linear('proj',pooled.flatten())
            h=np.maximum(self.linear('head.0',np.concatenate([f,rr])),0)
            total+=self.linear('head.2',h)
            if t in (3,6,10):
                logits=sat(total//t)
                cls=int(np.argmax(logits))
                p=np.exp((logits-logits.max())/S); p/=p.sum()
                top=np.sort(p)
                if t==10 or (adaptive and top[-1]-top[-2]>=self.threshold[cls]):
                    return logits,cls,t
