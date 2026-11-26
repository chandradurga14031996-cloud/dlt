"""Original SSAN, independent validation, CPU checkpoints. Historical RR preprocessing."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse, json, random, hashlib
from pathlib import Path
import numpy as np
import torch
import importantonlymit as base
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--protocol",choices=["intra","inter"],required=True)
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--test",action="store_true")
    p.add_argument("--cache",required=True)
    p.add_argument("--balance",choices=["original","mild"],default="original")
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    device=torch.device("cpu")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(a).items() if k not in ("resume","test")}
    cfg=out/"config.json"
    if cfg.exists():
        if json.loads(cfg.read_text())!=config or not (a.resume or a.test):
            raise RuntimeError("Existing run: use identical arguments and --resume or --test")
    else:
        if a.resume or a.test: raise RuntimeError("No existing run")
        cfg.write_text(json.dumps(config,indent=2))
    cache=Path(a.cache); cache.parent.mkdir(parents=True,exist_ok=True)
    records=sorted(base.DS1_RECORDS+base.DS2_RECORDS)
    fingerprint=hashlib.sha256((Path(base.__file__).read_bytes()+str(Path(a.data).resolve()).encode())).hexdigest()
    if cache.exists():
        with np.load(cache,allow_pickle=False) as d:
            assert str(d["fingerprint"])==fingerprint,"Cache source mismatch"
            w,r,y,ids=[d[k] for k in ("w","r","y","ids")]
    else:
        w,r,y,ids=base.load_real_mitbih_dataset(records,db_dir=a.data)
        ids=np.asarray(ids,dtype=int)
        assert set(ids)==set(records)
        np.savez_compressed(cache,w=w,r=r,y=y,ids=ids,fingerprint=fingerprint)
    assert set(ids)==set(records) and w.shape==(len(y),180) and r.shape==(len(y),6)
    assert np.isfinite(w).all() and np.isfinite(r).all()
    if a.protocol=="intra":
        pool,te=base.patient_specific_split(w,r,y,ids,seed=a.seed)
        tr0,va0=base.patient_specific_split(w[pool],r[pool],y[pool],ids[pool],seed=a.seed+1)
        tr,va=pool[tr0],pool[va0]
    else:
        vr=[108,114,118,124,203,223]
        train_records=set(base.DS1_RECORDS+[202])-set(vr)
        test_records=set(base.DS2_RECORDS)-{202}
        tr=np.flatnonzero(np.isin(ids,list(train_records)))
        va=np.flatnonzero(np.isin(ids,vr))
        te=np.flatnonzero(np.isin(ids,list(test_records)))
        assert not set(ids[tr]) & set(ids[te])
    assert len(np.unique(np.concatenate([tr,va,te])))==len(y)
    for name,idx in [("train",tr),("validation",va),("test",te)]:
        counts=np.bincount(y[idx],minlength=4)
        assert min(counts)>0
        print(name,len(idx), "class counts",counts.tolist(),flush=True)
    np.savez(out/"split.npz",train=tr,validation=va,test=te)
    def loader(idx,shuffle=False):
        return DataLoader(base.ECGDataset(w[idx],r[idx],y[idx]),batch_size=a.batch,shuffle=shuffle,drop_last=shuffle and len(idx)%a.batch==1)
    train_loader=loader(tr,True); val_loader=loader(va)
    model=base.Proposed_SSAN_Model().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=1e-5)
    criterion_s1=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,2.]))
    weights=[1.,2.2,1.8,3.2] if a.balance=="original" else [1.,3.,1.8,4.]
    criterion_s2=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
    start=0; best=-1.
    def save(path,state):
        tmp=path.with_suffix(".tmp")
        torch.save(state,tmp); os.replace(tmp,path)
    def evaluate(dl):
        acc,f1,_,_,d=base.evaluate_proposed_ssan(model,dl)
        return {"accuracy":float(acc),"macro_f1":float(f1),"per_class":{k:{m:float(v[m]) for m in ("precision_pp","recall","f1")} for k,v in d["per_class"].items()},"confusion":d["confusion_matrix"].tolist()}
    if a.test:
        ck=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
        if ck["epoch"]<1: raise RuntimeError("No trained model")
        model.load_state_dict(ck["model"])
        result=evaluate(loader(te))
        (out/"test.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True); return
    if a.resume:
        ck=torch.load(out/"last.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        start=ck["epoch"]; best=ck["best"]; torch.set_rng_state(ck["rng"])
    for epoch in range(start,a.epochs):
        model.train(); total_loss=0.; total=0; correct=0
        for x,rr_batch,y_batch in train_loader:
            optimizer.zero_grad()
            B=x.size(0)
            m1 = model.lif1.init_leaky(B, 16 * 90, device)
            m2 = model.lif2.init_leaky(B, 24 * 45, device)
            mq = model.ssa.lif_q.init_leaky(B, 16 * 45, device)
            mk = model.ssa.lif_k.init_leaky(B, 16 * 45, device)
            mv = model.ssa.lif_v.init_leaky(B, 24 * 45, device)
            ms1 = model.s1_lif.init_leaky(B, 2, device)

            accum_f, s1_spks = [], []
            for t in range(model.T_max):
                c1_out = model.mp1(model.bn1(model.c1(x)))
                spk1, m1 = model.lif1(c1_out.reshape(B, -1), m1)
                spk1_spatial = spk1.reshape(B, 16, 90)

                c2_out = model.mp2(model.bn2(model.c2(spk1_spatial)))
                spk2, m2 = model.lif2(c2_out.reshape(B, -1), m2)
                spk2_spatial = spk2.reshape(B, 24, 45)

                ssa_out, mq, mk, mv = model.ssa(spk2_spatial, mq, mk, mv)
                flat = model.fc_flat(ssa_out.reshape(B, -1))
                fused = model.bn_fused(torch.cat([flat, rr_batch], dim=-1))
                accum_f.append(fused)

                spk_s1, ms1 = model.s1_lif(model.s1_head(fused), ms1)
                s1_spks.append(spk_s1)

            s1_out = torch.stack(s1_spks).sum(dim=0)
            logits = model.stage2_specialist(torch.stack(accum_f).mean(dim=0))
            y_s1 = (y_batch != 0).long()
            
            loss = criterion_s1(s1_out, y_s1) + 1.5 * criterion_s2(logits, y_batch)


            if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total_loss+=loss.item()*B; total+=B
            correct+=(logits.argmax(1)==y_batch).sum().item()
        scheduler.step()
        metrics=evaluate(val_loader)
        metrics.update(epoch=epoch+1,loss=total_loss/total,train_accuracy=100*correct/total)
        improved=metrics["macro_f1"]>best
        if improved: best=metrics["macro_f1"]
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,best=best,rng=torch.get_rng_state())
        if improved: save(out/"best.pt",state)
        save(out/"last.pt",state)
        with (out/"history.jsonl").open("a") as f: f.write(json.dumps(metrics)+"\n")
        print(json.dumps(metrics),flush=True)

if __name__=="__main__":
    main()
