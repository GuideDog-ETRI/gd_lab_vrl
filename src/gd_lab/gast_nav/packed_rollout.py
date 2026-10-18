"""Lossless post-warp step maps; actor and critic share one GPU byte pool.
Age indices address exact per-frame float values, including historical elapsed
age. No categorical warp or parameter-dependent latent is replayed/cached.
"""
import torch

MAP_KEYS=('nav_wide','nav_near')
class CapturePool:
    def __init__(self, source):
        self.source=source;self.clear()

    def clear(self):
        self.records=[];self.bank=None;self.rows=0

    def references(self, obs):
        wide=obs['nav_wide'];n=len(wide)
        x=wide.reshape(n,3,3,3,43,49)
        age=x[:,:,:,2]
        frames=self.source.pipeline.mapper.history
        elapsed=torch.zeros(n,3,device=wide.device,dtype=torch.float64)
        for i,f in enumerate(frames):
            elapsed[:,i]=(frames[0].sample_time_ms-f.sample_time_ms).double()/1000
        codes=torch.arange(16,device=wide.device)
        # Match network_inputs arithmetic exactly: float32 upper, float64 elapsed.
        upper=codes.float()*.1
        lut=((upper.double()[None,None,:]+elapsed[:,:,None])/1.5).clamp(0,1).float()
        lut[:,:,15]=1.
        index=(age.double()*1.5-elapsed[:,:,None,None,None])/.1
        index=index.round().clamp(0,14).to(torch.uint8)
        index=torch.where(age==1,15,index)
        state=x[:,:,:,0].to(torch.uint8)*2+x[:,:,:,1].to(torch.uint8)
        bits=(state|(index<<2)).flatten(1,2).clone()
        near=(obs['nav_near']*255).round().to(torch.uint8)
        record={'bits':bits,'age_lut':lut,'near':near}
        # Fail closed if future observation semantics cannot be represented exactly.
        decoded=self.decode(record)
        if not torch.equal(decoded[0],wide) or not torch.equal(decoded[1],obs['nav_near']):
            raise ValueError('post-warp uint8 representation is not lossless')
        small=obs.exclude(*MAP_KEYS).clone()
        small['nav_capture_ids']=torch.arange(self.rows,self.rows+n,device=wide.device)
        if self.bank is not None:self.records=[self.bank];self.bank=None
        self.records.append(record);self.rows+=n
        return small

    @staticmethod
    def decode(b):
        bits=b['bits'];n=len(bits)
        idx=(bits>>2).reshape(n,3,-1).long()
        age=b['age_lut'].gather(2,idx).reshape(n,9,43,49)
        state=bits&3
        wide=torch.stack(((state==2).float(),(state==1).float(),age),2).flatten(1,2)
        return wide,b['near'].float()/255.

    def seal(self):
        if self.bank is None:
            self.bank={key:torch.cat([r[key] for r in self.records]) for key in self.records[0]}
            self.records.clear()

    def unpack(self, small):
        self.seal();ids=small['nav_capture_ids']
        wide,near=self.decode({k:v[ids] for k,v in self.bank.items()})
        result=small.clone(False);result['nav_wide']=wide;result['nav_near']=near
        return result

    def bytes(self):
        values=(v for r in self.records for v in r.values()) if self.bank is None else self.bank.values()
        return sum(v.numel()*v.element_size() for v in values)
