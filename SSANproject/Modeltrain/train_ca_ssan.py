"""Patient-disjoint experimental training; test is opt-in and never used for tuning."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from importantonlymit import ECGDataset, load_real_mitbih_dataset
from ca_ssan_model import CASSAN

DS1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

def patient(rec):
    return 201 if int(rec) in (201,202) else int(rec)

def contrast(h,y):
    h=F.normalize(h,dim=1)
    z=h@h.T/.1
    eye=torch.eye(len(y),dtype=torch.bool,device=y.device)
    pos=(y[:,None]==y[None,:]) & ~eye
    logp=z-torch.logsumexp(z.masked_fill(eye,-1e9),dim=1,keepdim=True)
    valid=pos.any(1)
    return -(logp*pos).sum(1)[valid].div(pos.sum(1)[valid]).mean() if valid.any() else h.sum()*0

def controlled_smote(w,r,y,ratios,rng):
    x=np.concatenate([w,r],axis=1).astype(np.float32)
    scale=np.maximum(x.std(axis=0),1e-6)
    counts=np.bincount(y,minlength=4)
    chunks=[x]; labels=[y]
    for c,ratio in enumerate(ratios,1):
        z=x[y==c]; n=max(0,int(np.ceil(counts[0]*ratio))-len(z))
        if not n: continue
        if len(z)<2: raise ValueError(f'SMOTE needs two real samples for class {c}')
        k=min(5,len(z)-1)
        nn=NearestNeighbors(n_neighbors=k+1).fit(z/scale)
        neighbors=nn.kneighbors(z/scale,return_distance=False)
        neighbors=np.stack([row[row!=i][:k] for i,row in enumerate(neighbors)])
        i=rng.integers(len(z),size=n)
        j=neighbors[i,rng.integers(k,size=n)]
        lam=rng.random((n,1)).astype(np.float32)
        chunks.append(z[i]+lam*(z[j]-z[i]))
        labels.append(np.full(n,c,dtype=y.dtype))
    x=np.concatenate(chunks); y=np.concatenate(labels)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite synthetic features')
    return x[:,:w.shape[1]],x[:,w.shape[1]:],y

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--limit',type=int,default=0,help='Maximum beats per partition; smoke only')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--variant',choices=['reference','balanced','compact','qat','full'],default='full')
    p.add_argument('--out',required=True)
    p.add_argument('--test',action='store_true',help='Evaluate held-out test after configuration is frozen')
    p.add_argument('--smote',action='store_true')
    p.add_argument('--ratios',type=float,nargs=3,default=[.25,.35,.15],metavar=('S','V','F'))
    p.add_argument('--patience',type=int,default=0)
    p.add_argument('--schedule',action='store_true')
    a=p.parse_args()
    if a.patience<0 or not all(0<=v<=1 for v in a.ratios): p.error('Invalid patience or ratios')
    if a.smote and (a.variant=='reference' or a.limit): p.error('SMOTE requires full development data and balanced variant')
    if a.limit and a.test: p.error('Smoke runs cannot evaluate test')
    if a.epochs<1 or a.batch<2: p.error('Invalid epochs or batch')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move record 202 to development to keep the shared patient together.
    development=DS1+[202]; test=[r for r in DS2 if r!=202]
    groups=np.array([patient(r) for r in development])
    # Select split by record groups, not by individual beats.
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=a.seed).split(development,groups=groups))
    records=[[development[i] for i in tr],[development[i] for i in va],test]
    sets=[set(map(patient,r)) for r in records]
    assert not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    print('Record splits:',records,flush=True)
    def load(recs,augment=False):
        w,r,y,ids=load_real_mitbih_dataset(recs,db_dir=a.data)
        if set(map(int,np.unique(ids)))!=set(recs): raise ValueError('Missing records')
        if not np.isfinite(w).all() or not np.isfinite(r).all(): raise ValueError('Nonfinite inputs')
        if a.limit and len(y)>a.limit:
            ix=rng.choice(len(y),a.limit,replace=False); w,r,y=w[ix],r[ix],y[ix]
        print('Class counts:',np.bincount(y,minlength=4),flush=True)
        if augment:
            w,r,y=controlled_smote(w,r,y,a.ratios,rng)
            print('After training-only SMOTE:',np.bincount(y,minlength=4),flush=True)
        return ECGDataset(w,r,y),y
    train,y=load(records[0],augment=a.smote); val,vy=load(records[1])
    counts=np.bincount(y,minlength=4)
    if not a.limit and (min(counts)==0 or min(np.bincount(vy,minlength=4))==0):
        raise ValueError('Missing class in development split; predefine a suitable grouped split')
    balanced=a.variant!='reference'
    weights=(1-.9999)/(1-np.power(.9999,np.maximum(counts,1)))
    weights=np.sqrt(weights); weights=weights/weights.mean()
    weights=torch.tensor(weights,dtype=torch.float32,device=dev)
    sampler=WeightedRandomSampler(torch.tensor(np.power(np.maximum(counts,1),-.25)[y],dtype=torch.double),len(y),replacement=True) if balanced and not a.smote else None
    tl=DataLoader(train,batch_size=a.batch,sampler=sampler,shuffle=sampler is None)
    vl=DataLoader(val,batch_size=a.batch)
    m=CASSAN(compact=a.variant in ('compact','qat','full'),quant=a.variant in ('qat','full'),shift=a.variant in ('qat','full')).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3,min_lr=1e-5) if a.schedule else None
    history=[]; stale=0; best_epoch=0
    def collect(loader,threshold=None):
        m.eval(); ys=[]; ps=[]; ts=[]
        with torch.no_grad():
            for x,r,y in loader:
                x,r=x.to(dev),r.to(dev)
                if threshold is None:
                    z=m(x,r)[0][-1]; t=torch.full((len(x),),10)
                else: z,t=m.adaptive(x,r,threshold)
                ys.extend(y.tolist()); ps.extend(z.argmax(1).cpu().tolist()); ts.extend(t.cpu().tolist())
        return np.array(ys),np.array(ps),float(np.mean(ts))
    def score(y,p): return f1_score(y,p,labels=[0,1,2,3],average='macro',zero_division=0)
    best=-1
    for epoch in range(a.epochs):
        m.train(); loss_sum=0
        for x,r,y in tl:
            x,r,y=x.to(dev),r.to(dev),y.to(dev)
            opt.zero_grad()
            exits,h,activity=m(x,r)
            losses=[]
            for z in exits:
                ce=F.cross_entropy(z,y,reduction='none')
                losses.append((weights[y]*(1-torch.exp(-ce))**2*ce).mean() if balanced else ce.mean())
            loss=sum(losses)/len(losses)
            if a.variant=='full': loss=loss+.05*contrast(h,y)+.001*activity
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
            loss_sum+=loss.item()
        yy,pp,_=collect(vl); f=score(yy,pp)
        print(f'Epoch {epoch+1}: loss={loss_sum/len(tl):.4f} val macro F1={f:.4f}',flush=True)
        history.append(dict(epoch=epoch+1,loss=loss_sum/len(tl),macro_f1=float(f),lr=opt.param_groups[0]['lr']))
        if f>best:
            best=f; best_epoch=epoch+1; stale=0
            state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else: stale+=1
        if scheduler is not None: scheduler.step(f)
        if a.patience and stale>=a.patience:
            print('Early stopping at epoch',epoch+1,flush=True); break
    m.load_state_dict(state)
    yy,pp,_=collect(vl)
    base=f1_score(yy,pp,labels=[0,1,2,3],average=None,zero_division=0)
    thresholds=torch.full((4,),2.,device=dev)
    if a.variant=='full':
        # Greedy validation-only calibration; retain each class F1 within 1 point.
        for c in range(4):
            if not np.any(yy==c): continue
            for margin in (.2,.4,.6,.8,.95):
                candidate=thresholds.clone(); candidate[c]=margin
                yv,pv,_=collect(vl,candidate)
                f=f1_score(yv,pv,labels=[0,1,2,3],average=None,zero_division=0)
                if np.all(f>=base-.01): thresholds=candidate; break
    def report(loader):
        yy,pp,t=collect(loader,thresholds)
        return dict(accuracy=float(accuracy_score(yy,pp)),macro_f1=float(score(yy,pp)),avg_steps=t,
                    report=classification_report(yy,pp,labels=[0,1,2,3],target_names=['N','S','V','F'],zero_division=0,output_dict=True))
    result=dict(history=history,best_epoch=best_epoch,training_counts=counts.tolist(),config=vars(a),records=records,thresholds=thresholds.cpu().tolist(),parameters=sum(p.numel() for p in m.parameters()),validation=report(vl))
    if a.test:
        ds,_=load(records[2]); result['test']=report(DataLoader(ds,batch_size=a.batch))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state=state,thresholds=thresholds.cpu(),config=vars(a)),out/'model.pt')
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
