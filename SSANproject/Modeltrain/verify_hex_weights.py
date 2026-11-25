"""Verify exported ROM integrity, checkpoint encoding and integer golden vectors."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from fixed_point_reference import FixedReference,read_hex,LAYERS,fixed,decode
from export_weights import encode

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dir',required=True)
    p.add_argument('--checkpoint',required=True)
    a=p.parse_args()
    root=Path(a.dir); ref=FixedReference(root); meta=ref.meta
    if hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()!=meta['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    for name,digest in meta['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest: raise ValueError('Hash mismatch: '+name)
    for name in LAYERS:
        w,b=ref.layers[name]
        np.testing.assert_array_equal(w,encode(ck['state'][name+'.weight'].numpy()))
        np.testing.assert_array_equal(b,fixed(ck['state'][name+'.bias'].numpy()))
    np.testing.assert_array_equal(read_hex(root/'thresholds.hex',32),fixed(ck['thresholds'].numpy()))
    np.testing.assert_array_equal(meta['threshold_q16'],fixed(ck['thresholds'].numpy()))
    for value,code in [(0,128),(1,0),(2,1),(.5,63),(-2,65),(-.5,127)]:
        assert int(encode(value))==code
        sign,e=decode(code); assert sign*2**e==value
    mismatches=0
    for v in meta['vectors']:
        x=read_hex(root/v['input'],32)
        z,c,t=ref.infer(x[:180],x[180:])
        np.testing.assert_array_equal(z,read_hex(root/v['logits'],32))
        assert (c,t)==(v['class_id'],v['timesteps'])
        same=(c,t)==(v['torch_class'],v['torch_timesteps'])
        mismatches+=not same
        print(v['name'], 'integer replay PASS', 'float agreement',same,'max logit error',v['max_logit_error'])
    print('ROM/checkpoint and integer replay PASS; float class/step mismatches:',mismatches)
    if mismatches: raise SystemExit('Float agreement not achieved; investigate before RTL deployment')
if __name__=='__main__': main()
