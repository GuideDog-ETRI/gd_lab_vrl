"""Independent block seeds; reset draws indexed by env and reset ordinal."""
import functools,hashlib,random
import numpy as np
import torch
SEEDS=(42,)
def install(cfg):
    for name in dir(cfg.events):
        term=getattr(cfg.events,name)
        if getattr(term,'mode',None)!='reset':continue
        original=term.func
        if not callable(original):raise TypeError((name,original))
        def wrap(original,name):
            counts={}
            @functools.wraps(original)
            def call(env,env_ids,**kw):
                ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
                for e in ids.tolist():
                    ordinal=counts.get(e,0);counts[e]=ordinal+1
                    key=f'{SEEDS[e//32]}:{name}:{e%32}:{ordinal}'
                    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
                    py=random.getstate();npstate=np.random.get_state()
                    devices=[torch.device(env.device)] if str(env.device).startswith('cuda') else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
                        original(env,torch.tensor([e],device=env.device),**kw)
                    random.setstate(py);np.random.set_state(npstate)
            return call
        term.func=wrap(original,name)
