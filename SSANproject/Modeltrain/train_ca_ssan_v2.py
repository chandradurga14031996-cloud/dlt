import argparse,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report,f1_score,accuracy_score
from train_ca_ssan import DS1,DS2,patient,controlled_smote
from importantonlymit import ECGDataset,load_real_mitbih_dataset
from ca_ssan_v2 import V2,Teacher

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=10); p.add_argument('--teacher-epochs',type=int,default=10)
    p.add_argument('--pool',type=int,default=10); p.add_argument('--width',type=int,default=96)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--batch',type=int,default=32)
    p.add_argument('--sweep',action='store_true'); p.add_argument('--smote',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.teacher_epochs,a.pool,a.width,a.batch)<1 or a.pool>45: p.error('Invalid configuration')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def seed():
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    seed()
    records=DS1+[202]
    w,r,y,ids=load_real_mitbih_dataset(records,db_dir=a.data)
    groups=np.array([patient(i) for i in ids])
    if set(map(int,np.unique(ids)))!=set(records): raise ValueError('Missing development records')
    # Choose a split using class counts only, never model predictions.
    candidates=[]
    for tr,va in GroupShuffleSplit(n_splits=100,test_size=.25,random_state=a.seed).split(w,y,groups):
        c=np.bincount(y[va],minlength=4); d=np.bincount(y[tr],minlength=4)
        if min(c)<20 or min(d)<20: continue
        target=np.bincount(y,minlength=4)*.25
        candidates.append((float(np.mean(abs(c-target)/target)),tr,va))
    if not candidates: raise ValueError('No grouped split meets minimum class support')
    _,tr,va=min(candidates,key=lambda z:z[0])
    assert not set(groups[tr])&set(groups[va])
    assert not set(groups[va])&set(map(patient,[i for i in DS2 if i!=202]))
    split=dict(train_records=np.unique(ids[tr]).tolist(),validation_records=np.unique(ids[va]).tolist(),train_counts=np.bincount(y[tr],minlength=4).tolist(),validation_counts=np.bincount(y[va],minlength=4).tolist())
    (out/'split.json').write_text(json.dumps(split,indent=2)); print(split,flush=True)
    tw,rr,ty=w[tr],r[tr],y[tr]
    if a.smote: tw,rr,ty=controlled_smote(tw,rr,ty,[.25,.35,.15],np.random.default_rng(a.seed))
    train=ECGDataset(tw,rr,ty); val=DataLoader(ECGDataset(w[va],r[va],y[va]),batch_size=a.batch)
    def loader(): return DataLoader(train,batch_size=a.batch,shuffle=True,generator=torch.Generator().manual_seed(a.seed))
    def evaluate(m,teacher=False):
        m.eval(); yy=[]; pp=[]
        with torch.no_grad():
            for x,r,y in val:
                x,r=x.to(dev),r.to(dev)
                z=m(x,r) if teacher else m(x,r)[0][-1]
                yy.extend(y.tolist()); pp.extend(z.argmax(1).cpu().tolist())
        return dict(accuracy=accuracy_score(yy,pp),macro_f1=f1_score(yy,pp,labels=[0,1,2,3],average='macro',zero_division=0),report=classification_report(yy,pp,labels=[0,1,2,3],target_names=['N','S','V','F'],output_dict=True,zero_division=0))
    def fit(name,m,epochs,teacher=None,qat=False,is_teacher=False):
        m=m.to(dev); opt=torch.optim.AdamW(m.parameters(),lr=.001)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=3,factor=.5,min_lr=1e-5)
        best=-1; stale=0; history=[]
        for e in range(epochs):
            quant=qat and e>=epochs//2
            if not is_teacher:
                if quant and not m.quant: best=-1; stale=0
                m.quant=quant
            m.train(); total=0
            for x,r,y in loader():
                x,r,y=x.to(dev),r.to(dev),y.to(dev); opt.zero_grad()
                if is_teacher: loss=F.cross_entropy(m(x,r),y)
                else:
                    exits,activity=m(x,r)
                    loss=sum(F.cross_entropy(z,y) for z in exits)/len(exits)+.001*activity
                    if teacher is not None:
                        with torch.no_grad(): target=F.softmax(teacher(x,r)/2,dim=1)
                        loss=.7*loss+.3*4*sum(F.kl_div(F.log_softmax(z/2,dim=1),target,reduction='batchmean') for z in exits)/len(exits)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step(); total+=loss.item()
            result=evaluate(m,is_teacher); score=result['macro_f1']
            history.append(dict(epoch=e+1,loss=total/len(loader()),macro_f1=score,quant=quant))
            print(name,history[-1],flush=True); sched.step(score)
            if score>best:
                best=score; stale=0; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; best_epoch=e+1
            else: stale+=1
            if stale>=15 and (not qat or quant): break
        m.load_state_dict(state); m.eval()
        result=evaluate(m,is_teacher)
        result.update(name=name,best_epoch=best_epoch,parameters=sum(p.numel() for p in m.parameters()),history=history)
        torch.save(dict(state=state,quant=qat,config=vars(a),name=name),out/(name+'.pt'))
        (out/(name+'.json')).write_text(json.dumps(result,indent=2))
        return m,result
    teacher,t=fit('teacher',Teacher(),a.teacher_epochs,is_teacher=True)
    cases=[('v2_full',True,True,True,True,True)]
    if a.sweep: cases=[('control',False,False,False,False,False),('multiscale',True,False,False,False,False),('hierarchy',True,True,False,False,False),('adaptive',True,True,True,False,False),('distilled',True,True,True,True,False)]+cases
    rows=[t]
    for name,ms,h,ad,kd,qat in cases:
        seed()
        _,result=fit(name,V2(a.pool,a.width,ms,h,ad),a.epochs,teacher if kd else None,qat)
        rows.append(result)
        (out/'comparison.json').write_text(json.dumps(rows,indent=2))
        import csv
        with (out/'comparison.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['model','accuracy','macro_f1','N_F1','S_F1','V_F1','F_F1','parameters','best_epoch'])
            for z in rows: writer.writerow([z['name'],z['accuracy'],z['macro_f1'],*[z['report'][c]['f1-score'] for c in 'NSVF'],z['parameters'],z['best_epoch']])
if __name__=='__main__': main()
