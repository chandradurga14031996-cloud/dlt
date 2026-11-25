"""V3 minority-focused experiment. Existing V1/V2 remain unchanged."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit,train_test_split
from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix
from train_ca_ssan import DS1,DS2,patient,controlled_smote
from importantonlymit import ECGDataset,load_real_mitbih_dataset
from ca_ssan_v2 import V2,Teacher

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--protocol',choices=['inter','intra'],default='inter')
    p.add_argument('--epochs',type=int,default=80); p.add_argument('--teacher-epochs',type=int,default=40)
    p.add_argument('--qat-epochs',type=int,default=30); p.add_argument('--batch',type=int,default=64)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--pool',type=int,default=15)
    p.add_argument('--width',type=int,default=128); p.add_argument('--smote',action='store_true')
    p.add_argument('--gamma',type=float,default=1.5); p.add_argument('--test',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.teacher_epochs,a.qat_epochs,a.batch,a.pool,a.width)<1 or a.pool>45 or a.gamma<0: p.error('Invalid settings')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    records=DS1+[202] if a.protocol=='inter' else DS1+DS2
    w,r,y,ids=load_real_mitbih_dataset(records,db_dir=a.data)
    if set(map(int,np.unique(ids)))!=set(records): raise ValueError('Missing records')
    if not np.isfinite(w).all() or not np.isfinite(r).all(): raise ValueError('Nonfinite input')
    groups=np.array([patient(i) for i in ids])
    if a.protocol=='inter':
        candidates=[]
        for tr,va in GroupShuffleSplit(n_splits=100,test_size=.25,random_state=a.seed).split(w,y,groups):
            c=np.bincount(y[va],minlength=4); d=np.bincount(y[tr],minlength=4)
            if min(c)<20 or min(d)<20: continue
            candidates.append((np.mean(abs(c-np.bincount(y,minlength=4)*.25)/np.maximum(np.bincount(y,minlength=4),1)),tr,va))
        if not candidates: raise ValueError('No adequate grouped split')
        _,tr,va=min(candidates,key=lambda z:z[0])
        assert not set(groups[tr])&set(groups[va])
    else:
        tr,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=a.seed)
        tr,va=train_test_split(tr,test_size=.2,stratify=y[tr],random_state=a.seed)
    tw,rr,ty=w[tr],r[tr],y[tr]
    if a.smote: tw,rr,ty=controlled_smote(tw,rr,ty,[.6,.35,.5],np.random.default_rng(a.seed))
    counts=np.bincount(ty,minlength=4)
    if min(counts)==0: raise ValueError('Missing training class')
    sampler=WeightedRandomSampler(torch.as_tensor(counts[ty]**(-.65),dtype=torch.double),len(ty),replacement=True)
    train=DataLoader(ECGDataset(tw,rr,ty),batch_size=a.batch,sampler=sampler)
    val=DataLoader(ECGDataset(w[va],r[va],y[va]),batch_size=a.batch)
    weights=torch.tensor([1.,1.5,1.,1.5],device=dev)
    def loss(z,y):
        ce=F.cross_entropy(z,y,reduction='none')
        return (weights[y]*(1-torch.exp(-ce))**a.gamma*ce).mean()
    def evaluate(m,loader,teacher=False):
        m.eval(); yy=[]; pp=[]
        with torch.no_grad():
            for x,r,y in loader:
                x,r=x.to(dev),r.to(dev)
                z=m(x,r) if teacher else m(x,r)[0][-1]
                yy.extend(y.tolist()); pp.extend(z.argmax(1).cpu().tolist())
        f=f1_score(yy,pp,labels=[0,1,2,3],average=None,zero_division=0)
        return dict(accuracy=float(accuracy_score(yy,pp)),macro_f1=float(f.mean()),min_f1=float(f.min()),class_f1=f.tolist(),confusion=confusion_matrix(yy,pp,labels=[0,1,2,3]).tolist(),report=classification_report(yy,pp,labels=[0,1,2,3],target_names=list('NSVF'),output_dict=True,zero_division=0))
    def fit(name,m,epochs,teacher=None,is_teacher=False,lr=.001):
        m.to(dev); opt=torch.optim.AdamW(m.parameters(),lr=lr)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=4,factor=.5)
        best=-1; stale=0
        for e in range(epochs):
            m.train()
            for x,r,y in train:
                x,r,y=x.to(dev),r.to(dev),y.to(dev); opt.zero_grad()
                if is_teacher: l=loss(m(x,r),y)
                else:
                    z,act=m(x,r)
                    l=.15*loss(z[0],y)+.25*loss(z[1],y)+.60*loss(z[-1],y)+.001*act
                    if teacher is not None:
                        with torch.no_grad(): target=F.softmax(teacher(x,r)/2,dim=1)
                        l=.8*l+.2*4*F.kl_div(F.log_softmax(z[-1]/2,dim=1),target,reduction='batchmean')
                l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
            res=evaluate(m,val,is_teacher); score=.5*res['macro_f1']+.5*res['min_f1']
            print(name,e+1,res['class_f1'],score,flush=True); sched.step(score)
            if score>best:
                best=score; stale=0; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            else: stale+=1
            if stale>=15: break
        m.load_state_dict(state); m.eval()
        res=evaluate(m,val,is_teacher)
        torch.save(dict(state=state,config=vars(a),quant=getattr(m,'quant',False)),out/(name+'.pt'))
        (out/(name+'.json')).write_text(json.dumps(res,indent=2))
        return res
    teacher=Teacher()
    t=fit('teacher',teacher,a.teacher_epochs,is_teacher=True)
    model=V2(a.pool,a.width,True,True,True)
    f=fit('float',model,a.epochs,teacher)
    model.quant=True
    q=fit('qat',model,a.qat_epochs,teacher,lr=.0002)
    result=dict(config=vars(a),teacher=t,float=f,qat=q,train_records=np.unique(ids[tr]).tolist(),validation_records=np.unique(ids[va]).tolist(),train_counts=counts.tolist(),validation_counts=np.bincount(y[va],minlength=4).tolist())
    if a.test:
        if a.protocol=='inter':
            rec=[i for i in DS2 if i!=202]
            x,z,l,i=load_real_mitbih_dataset(rec,db_dir=a.data)
            assert set(map(int,np.unique(i)))==set(rec)
            assert not set(map(patient,i))&set(groups)
        else: x,z,l=w[te],r[te],y[te]
        result['test']=evaluate(model,DataLoader(ECGDataset(x,z,l),batch_size=a.batch))
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
