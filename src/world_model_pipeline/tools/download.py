#!/usr/bin/env python3
"""Resumable parallel public-file download with proxies disabled and SHA256 check."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

import requests


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def download(url,target,size,expected,workers=8,chunk_size=32*1024*1024):
    target=Path(target).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size==size and sha256(target)==expected:
            print(f'Existing verified: {target}',flush=True);return
        raise ValueError(f'Existing file does not match expected checksum: {target}')
    partial=target.with_suffix(target.suffix+'.part')
    state_path=target.with_suffix(target.suffix+'.parts.json')
    chunks=(size+chunk_size-1)//chunk_size
    identity={'url':url,'size':size,'sha256':expected,'chunk_size':chunk_size}
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if any(state.get(k)!=v for k,v in identity.items()) or not partial.exists() or partial.stat().st_size!=size:
            raise ValueError('Partial download state does not match this request')
    else:
        with partial.open('xb') as f:f.truncate(size)
        state=dict(identity,completed=[])
        state_path.write_text(json.dumps(state))
    completed=set(state['completed'])
    lock=threading.Lock()
    fd=os.open(partial,os.O_RDWR)
    started=time.monotonic();initial=len(completed);last_print=[0.]
    local=threading.local();redirect_hosts=set()

    def fetch(index):
        start=index*chunk_size;end=min(start+chunk_size,size)-1
        if not hasattr(local,'session'):
            local.session=requests.Session()
            local.session.trust_env=False  # Never consume HTTP(S)/ALL_PROXY or proxy credentials.
        for attempt in range(5):
            try:
                with local.session.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                                       stream=True,timeout=(15,90)) as response:
                    response.raise_for_status()
                    correct=f'bytes {start}-{end}/{size}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=correct:
                        raise ValueError(f'Server did not honor range {index}: {response.status_code}, {response.headers.get("Content-Range")}')
                    position=start
                    for block in response.iter_content(1024*1024):
                        if not block:continue
                        if position+len(block)>end+1:raise ValueError('Range response too large')
                        view=memoryview(block)
                        while view:
                            written=os.pwrite(fd,view,position)
                            if written<=0:raise OSError('Short pwrite')
                            position+=written;view=view[written:]
                    if position!=end+1:raise EOFError('Range response truncated')
                    host=urlparse(response.url).hostname
                with lock:
                    redirect_hosts.add(host);completed.add(index)
                    state['completed']=sorted(completed)
                    temporary=state_path.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps(state));temporary.replace(state_path)
                    now=time.monotonic()
                    if now-last_print[0]>10 or len(completed)==chunks:
                        done=sum(min(chunk_size,size-i*chunk_size) for i in completed)
                        print(f'{target.name}: {done/1e9:.2f}/{size/1e9:.2f} GB; {len(completed)}/{chunks} ranges; direct host={host}',flush=True)
                        last_print[0]=now
                return
            except Exception:
                if attempt==4:raise
                time.sleep(min(2**attempt,16))
    try:
        pending=[i for i in range(chunks) if i not in completed]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed(pool.submit(fetch,i) for i in pending):future.result()
        os.fsync(fd)
    finally:os.close(fd)
    print('Checking complete SHA256...',flush=True)
    observed=sha256(partial)
    if observed!=expected:raise ValueError(f'SHA256 mismatch: expected {expected}; got {observed}')
    partial.rename(target)
    receipt=dict(identity,status='verified',target=str(target),observed_sha256=observed,
                 proxies_disabled=True,trust_env=False,redirect_hosts=sorted(redirect_hosts),
                 resumed_chunks=initial,wall_seconds=round(time.monotonic()-started,2))
    target.with_suffix('.download.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--size',required=True,type=int)
    p.add_argument('--sha256',required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    download(a.url,a.output,a.size,a.sha256,a.workers)
