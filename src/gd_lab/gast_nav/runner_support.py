"""Pure launcher bookkeeping; no optimizer or simulator side effects."""
import math
from pathlib import Path


def rank_env_count(total,world,rank):
    if total<world or not 0<=rank<world:raise ValueError('distributed environment allocation')
    return total//world+int(rank<total%world)

class TopK:
    def __init__(self,directory,k=5):
        self.directory=Path(directory);self.k=k;self.records=[]
    def consider(self,score,iteration,save):
        if not math.isfinite(score):return False
        if len(self.records)>=self.k and score<=min(x[0] for x in self.records):return False
        self.directory.mkdir(parents=True,exist_ok=True)
        path=self.directory/f'nav_{iteration:06d}.pt'
        save(path)
        self.records.append((float(score),int(iteration),path));self.records.sort(reverse=True)
        for _,_,old in self.records[self.k:]:old.unlink()
        self.records=self.records[:self.k]
        return True
