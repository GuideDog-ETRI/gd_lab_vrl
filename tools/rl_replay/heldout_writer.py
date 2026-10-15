"""UPDATE110 immutable batch-event replay; no simulator dependencies."""
import hashlib,json
from pathlib import Path
import torch

def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cpu(x) for x in v]
    return v

class HeldoutWriter:
    def __init__(self,path,metadata):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
        self.metadata=metadata;self.events=[];self.shards=[];self.packets=0
        (self.path/'INCOMPLETE').write_text('collection running\n')
    def add(self,kind,**kw):
        self.events.append(cpu(dict(kind=kind,**kw)))
        if kind=='packet':self.packets+=1
        if self.packets>=4:self.flush()
    def flush(self):
        if not self.events:return
        p=self.path/f'shard_{len(self.shards):05d}.pt';torch.save(self.events,p)
        self.shards.append(dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),events=len(self.events)))
        self.events=[];self.packets=0
    def finish(self):
        self.flush()
        (self.path/'manifest.json').write_text(json.dumps(dict(metadata=self.metadata,shards=self.shards,complete=True),indent=2))
        (self.path/'INCOMPLETE').unlink()
