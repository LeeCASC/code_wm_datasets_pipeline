#!/usr/bin/env python3
"""Convert local HF H3 text/VAEs into CWM-compatible standalone Safetensors.

Streams raw/cast tensors to bounded-memory output. Source weights stay read-only.
Published header schemas define target keys/shapes/dtypes, not parameter values.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

from world_model_pipeline.paths import ROOT
HEADERS = ROOT/'assets/model_conversion_headers'
NAMES = {'text_encoder': ('text_encoders', 'qwen3vl_32b_minimax_h3_bf16'),
         'video_vae': ('vae', 'minimax_h3_video_vae_fp16'),
         'audio_vae': ('vae', 'minimax_h3_audio_vae_fp32')}
DTYPES = {'F32': ('<f4',4), 'F16': ('<f2',2), 'BF16': ('<u2',2), 'I64': ('<i8',8)}


def header(path):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if size > 32*1024*1024:
            raise ValueError('Unexpectedly large Safetensors header')
        raw = f.read(size)
    data = json.loads(raw)
    result = {}
    for key, value in data.items():
        if key != '__metadata__':
            result[key] = dict(value, file=path, start=8+size+value['data_offsets'][0])
    return result


def shard_headers(folder):
    index = json.loads((folder/'model.safetensors.index.json').read_text())['weight_map']
    result = {}
    for filename in sorted(set(index.values())):
        current = header(folder/filename)
        for key, value in current.items():
            if key in result or index.get(key) != filename:
                raise ValueError(f'Shard/index mismatch: {key}')
            result[key] = value
    if set(result) != set(index):
        raise ValueError('Incomplete shard index')
    return result


def text_target(key):
    if key.startswith('model.language_model.'):
        return 'model.'+key[len('model.language_model.'):]
    if key.startswith('model.visual.'):
        return key[len('model.'):]
    return key


def recipe(source, role, expected):
    if role == 'text_encoder':
        raw = shard_headers(source/'text_encoder')
        mapping = {text_target(k): dict(v, operation='raw', source_key=k)
                   for k,v in raw.items() if text_target(k) in expected}
        # Do not silently discard an unrelated model's parameters.
        import re
        for k in raw:
            if text_target(k) not in expected:
                match = re.match(r'model\.language_model\.layers\.(\d+)\.', k)
                if k not in {'lm_head.weight','model.language_model.norm.weight'} and not (match and int(match[1]) >= 50):
                    raise ValueError(f'Unexpected text parameter: {k}')
    else:
        cfg = json.loads((source/role/'config.json').read_text())
        raw = header(source/role/('source/model.safetensors' if role == 'video_vae' else 'model.safetensors'))
        mapping = {}
        for key, spec in raw.items():
            if role == 'audio_vae' and key.endswith('.weight_g'):
                continue
            if role == 'audio_vae' and key.endswith('.weight_v'):
                target = key[:-2]
                gain = raw[target+'_g']
                if gain['shape'] != [spec['shape'][0]]+[1]*(len(spec['shape'])-1):
                    raise ValueError(f'Weight norm dimension is not 0: {key}')
                mapping[target] = dict(spec, operation='weight_norm', gain=gain, source_key=key)
            else:
                mapping[key] = dict(spec, operation='cast' if role == 'video_vae' else 'raw', source_key=key)
        for key in ['latents_mean','latents_std']:
            mapping[key] = {'shape': [len(cfg[key])], 'dtype':'F32',
                            'operation':'constant', 'value':cfg[key]}
    if set(mapping) != set(expected):
        raise ValueError(f'Target key mismatch: missing={set(expected)-set(mapping)}, extra={set(mapping)-set(expected)}')
    for key, target in expected.items():
        if mapping[key]['shape'] != target['shape']:
            raise ValueError(f'Shape mismatch: {key}')
        if mapping[key]['operation'] == 'raw' and mapping[key]['dtype'] != target['dtype']:
            raise ValueError(f'Dtype mismatch: {key}')
    return mapping


def write_stream(f, digest, source, count, transform=None):
    remain = count
    while remain:
        block = source.read(min(remain,16*1024*1024))
        if not block:
            raise EOFError('Unexpected source tensor EOF')
        remain -= len(block)
        if transform:
            values = np.frombuffer(block,dtype='<f4')
            converted = values.astype('<f2')
            if not np.isfinite(converted).all():
                raise ValueError('Non-finite value after FP16 conversion')
            block = converted.tobytes()
        f.write(block)
        digest.update(block)


def convert(source, output, role):
    directory, name = NAMES[role]
    reference = json.loads((HEADERS/(name+'.json')).read_text())
    expected = reference['tensors']
    mapping = recipe(source,role,expected)
    target = output/'models'/directory/(name+'.safetensors')
    receipt_path = target.with_suffix('.conversion.json')
    if target.exists():
        if not receipt_path.exists():
            raise FileExistsError(f'Existing target has no conversion receipt: {target}')
        print(f'Using completed conversion: {target}',flush=True)
        return json.loads(receipt_path.read_text())
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.safetensors.converting')
    tensors = sorted(expected,key=lambda k:expected[k]['data_offsets'][0])
    metadata = dict(reference.get('metadata') or {})
    metadata.update(local_conversion_source=str(source),local_conversion_role=role)
    target_header = {'__metadata__':metadata}
    offset = 0
    for key in tensors:
        spec = expected[key]
        size = math.prod(spec['shape'])*DTYPES[spec['dtype']][1]
        target_header[key] = {'dtype':spec['dtype'],'shape':spec['shape'],'data_offsets':[offset,offset+size]}
        offset += size
    encoded = json.dumps(target_header,separators=(',',':')).encode()
    encoded += b' '*((-len(encoded))%8)
    prefix = struct.pack('<Q',len(encoded))+encoded
    digest = hashlib.sha256()
    started = time.monotonic()
    if role == 'audio_vae':
        import torch
        from safetensors import safe_open
        torch.set_num_threads(4)
    with temporary.open('xb') as f:
        f.write(prefix); digest.update(prefix)
        for index,key in enumerate(tensors):
            spec, dest = mapping[key], expected[key]
            op = spec['operation']
            if op == 'constant':
                block = np.asarray(spec['value'],dtype=DTYPES[dest['dtype']][0]).tobytes()
                f.write(block); digest.update(block)
            elif op == 'weight_norm':
                with safe_open(str(spec['file']),framework='pt',device='cpu') as handle:
                    v = handle.get_tensor(spec['source_key'])
                    g = handle.get_tensor(spec['source_key'][:-2]+'_g')
                    with torch.no_grad():
                        folded = torch._weight_norm(v,g,0).contiguous()
                    if not torch.isfinite(folded).all():
                        raise ValueError(f'Non-finite folded parameter: {key}')
                    block = folded.numpy().tobytes()
                f.write(block); digest.update(block)
                del v,g,folded,block
            else:
                with spec['file'].open('rb') as reader:
                    reader.seek(spec['start'])
                    count = spec['data_offsets'][1]-spec['data_offsets'][0]
                    write_stream(f,digest,reader,count,transform=(op=='cast'))
            if index%50 == 0 or index == len(tensors)-1:
                print(f'{role}: {index+1}/{len(tensors)} tensors; {f.tell()/1e9:.2f} GB written',flush=True)
        if f.tell() != len(prefix)+offset:
            raise ValueError('Converted payload byte count mismatch')
        f.flush()
        import os
        os.fsync(f.fileno())
    # Parse the actual file through the safetensors library before promotion.
    from safetensors import safe_open
    with safe_open(str(temporary),framework='pt',device='cpu') as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError('Written Safetensors keys mismatch')
        for key in expected:
            if list(handle.get_slice(key).get_shape()) != expected[key]['shape']:
                raise ValueError(f'Written shape mismatch: {key}')
    receipt = {'status':'converted','role':role,'source':str(source),'target':str(target),
               'bytes':temporary.stat().st_size,'sha256':digest.hexdigest(),'tensors':len(tensors),
               'operations':dict(__import__('collections').Counter(v['operation'] for v in mapping.values())),
               'reference_header':str(HEADERS/(name+'.json')),
               'source_weights_modified':False,'official_weight_values_compared':False,
               'wall_seconds':round(time.monotonic()-started,2)}
    temporary.rename(target)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--release-root',type=Path,default=ROOT)
    p.add_argument('--only',nargs='+',choices=list(NAMES),default=list(NAMES))
    args=p.parse_args()
    receipts=[convert(args.source.resolve(),args.release_root.resolve(),role) for role in args.only]
    print(json.dumps({'conversions':receipts},indent=2))


if __name__=='__main__':
    main()
