"""Export compact QAT checkpoint to signed Po2 ROM and Q16.16 bias/test files."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from ca_ssan_model import CASSAN
from fixed_point_reference import FixedReference,LAYERS,S,fixed

def write_hex(path,values,bits):
    mask=(1<<bits)-1
    path.write_text(''.join(f'{int(v)&mask:0{bits//4}X}\n' for v in np.asarray(values).flatten()))

def encode(w):
    w=np.asarray(w,dtype=np.float64)
    if not np.isfinite(w).all(): raise ValueError('Nonfinite weight')
    e=np.clip(np.rint(np.log2(np.maximum(np.abs(w),1e-12))),-7,7).astype(np.int64)
    return np.where(np.abs(w)<2**-8,128,((w<0).astype(np.int64)<<6)|(e&63))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--data',help='Optional MIT-BIH directory for development-only vectors')
    a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck['config']['variant'] not in ('qat','full'): raise ValueError('Only compact QAT variants supported')
    m=CASSAN().eval(); m.load_state_dict(ck['state'],strict=True)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    threshold=ck['thresholds'].numpy()
    if threshold.shape!=(4,) or not np.isfinite(threshold).all() or np.any((threshold<0)|(threshold>2)): raise ValueError('Invalid thresholds')
    meta=dict(format='ca-ssan-q16-v1',checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              fractional_bits=16,word_bits=32,accumulator_bits=64,weight_bits=8,
              zero_code='80',exponent_range=[-7,7],layout='C order: output,input,kernel; channel-major features',
              rounding='nearest-even at export; arithmetic right shift and floor division during inference',
              saturation='signed int32 after affine, attention, membrane and final average',
              threshold_q16=fixed(threshold).tolist(),confidence='host softmax probability margin; not RTL',
              timesteps=[3,6,10],layers={},vectors=[])
    write_hex(out/'thresholds.hex',meta['threshold_q16'],32)
    for name,shape in LAYERS.items():
        layer=m.get_submodule(name)
        w=layer.weight.detach().numpy(); b=layer.bias.detach().numpy()
        if w.shape!=shape: raise ValueError(name)
        stem=name.replace('.','_')
        wf,bf=stem+'_weights.hex',stem+'_bias.hex'
        write_hex(out/wf,encode(w),8); write_hex(out/bf,fixed(b),32)
        meta['layers'][name]=dict(shape=list(shape),weights=wf,bias=bf)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    ref=FixedReference(out)
    rng=np.random.default_rng(42)
    vectors=[('zero',np.zeros(180),np.zeros(6)),('random',rng.normal(0,.5,180),np.ones(6)),('impulse',np.eye(1,180,90).ravel(),np.ones(6))]
    if a.data:
        from importantonlymit import load_real_mitbih_dataset
        w,r,y,_=load_real_mitbih_dataset([101,106],db_dir=a.data)
        for c in range(4):
            ix=np.flatnonzero(y==c)
            if len(ix): vectors.append(('development_class_'+str(c),w[ix[0]],r[ix[0]]))
    for name,x,r in vectors:
        x,r=fixed(x),fixed(r)
        logits,cls,t=ref.infer(x,r)
        with torch.no_grad():
            z,steps=m.adaptive(torch.tensor(x/S,dtype=torch.float32).reshape(1,1,180),torch.tensor(r/S,dtype=torch.float32).reshape(1,6),ck['thresholds'])
        write_hex(out/(name+'_input.hex'),np.concatenate([x,r]),32)
        write_hex(out/(name+'_logits.hex'),logits,32)
        meta['vectors'].append(dict(name=name,input=name+'_input.hex',logits=name+'_logits.hex',class_id=cls,timesteps=t,
                                    torch_class=int(z.argmax(1).item()),torch_timesteps=int(steps.item()),
                                    max_logit_error=float(np.max(np.abs(logits/S-z.numpy().ravel())))))
    meta['files_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.hex')}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta['vectors'],indent=2))
    print('Exported:',out)
if __name__=='__main__': main()
