"""V4 teacher-first qualification. No test data used."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from preprocessing_v4 import ECGDataset, load_real_mitbih_dataset

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(c,c,5,padding=2,bias=False),nn.BatchNorm1d(c),nn.ReLU(),nn.Conv1d(c,c,3,padding=1,bias=False),nn.BatchNorm1d(c))
    def forward(self,x): return F.relu(x+self.net(x))

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.ModuleList([nn.Conv1d(1,16,k,padding=k//2,bias=False) for k in (3,7,15)])
        self.body=nn.Sequential(nn.BatchNorm1d(48),nn.ReLU(),Block(48),nn.MaxPool1d(2),nn.Conv1d(48,64,3,padding=1,bias=False),nn.BatchNorm1d(64),nn.ReLU(),Block(64),nn.MaxPool1d(2),Block(64))
        self.embed=nn.Sequential(nn.Linear(64*10+6,128),nn.LayerNorm(128),nn.ReLU())
        self.head=nn.Linear(128,4)
    def forward(self,x,r):
        h=self.body(torch.cat([m(x) for m in self.stem],1))
        h=self.embed(torch.cat([F.adaptive_avg_pool1d(h,5).flatten(1),F.adaptive_max_pool1d(h,5).flatten(1),r],1))
        return self.head(h),h

def contrast(h,y):
    z=F.normalize(h,dim=1)
    s=z@z.T/.1
    eye=torch.eye(len(y),device=y.device,dtype=torch.bool)
    pos=(y[:,None]==y[None,:]) & ~eye
    logp=s-torch.logsumexp(s.masked_fill(eye,float('-inf')),dim=1,keepdim=True)
    valid=pos.any(1)
    return -(logp.masked_fill(~pos,0).sum(1)/pos.sum(1).clamp_min(1))[valid].mean() if valid.any() else h.sum()*0

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    dev=torch.device('cpu' if a.smoke or not torch.cuda.is_available() else 'cuda')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if a.smoke:
        m=Teacher();x=torch.randn(16,1,180);r=torch.randn(16,6);y=torch.arange(16)%4
        z,h=m(x,r);loss=F.cross_entropy(z,y)+.05*contrast(h,y);loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
        print('Synthetic teacher forward/backward passed');return
    tr=[101,106,109,112,115,116,119,122,201,202,205,207,208,209,215,220,230]
    va=[108,114,118,124,203,223]
    def load(recs):
        w,r,y,ids=load_real_mitbih_dataset(recs,db_dir=a.data)
        assert set(np.unique(ids))==set(recs)
        assert np.isfinite(w).all() and np.isfinite(r).all()
        assert w.shape==(len(y),180) and r.shape==(len(y),6)
        assert min(np.bincount(y,minlength=4))>0
        return w,r,y
    w,r,y=load(tr);vw,vr,vy=load(va)
    mean=r.mean(0);std=r.std(0).clip(1e-5)
    train=ECGDataset(w,((r-mean)/std).clip(-10,10),y)
    val=DataLoader(ECGDataset(vw,((vr-mean)/std).clip(-10,10),vy),batch_size=a.batch)
    counts=np.bincount(y,minlength=4)
    m=Teacher().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=4)
    best=-1
    for e in range(a.epochs):
        m.train()
        sampler=WeightedRandomSampler(torch.as_tensor(counts[y]**(-.25),dtype=torch.double),len(y),replacement=True) if e>=5 else None
        for x,r,t in DataLoader(train,batch_size=a.batch,sampler=sampler,shuffle=sampler is None):
            x,r,t=x.to(dev),r.to(dev),t.to(dev)
            opt.zero_grad();z,h=m(x,r);loss=F.cross_entropy(z,t)+.05*contrast(h,t)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        m.eval();pred=[]
        with torch.no_grad():
            for x,r,t in val:pred.extend(m(x.to(dev),r.to(dev))[0].argmax(1).cpu().tolist())
        f=f1_score(vy,pred,labels=[0,1,2,3],average=None,zero_division=0)
        res=dict(epoch=e+1,accuracy=accuracy_score(vy,pred),macro_f1=float(f.mean()),class_f1=f.tolist(),confusion=confusion_matrix(vy,pred,labels=[0,1,2,3]).tolist())
        score=.8*f.mean()+.2*f.min();sched.step(score)
        print(json.dumps(res),flush=True)
        state=dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=e+1,rr_mean=mean.tolist(),rr_std=std.tolist(),train_records=tr,validation_records=va,metrics=res)
        torch.save(state,out/'last.pt')
        if score>best:
            best=score;torch.save(state,out/'best.pt');(out/'teacher.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
