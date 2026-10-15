"""Sixteen persistent CPU workers, each owning disjoint env maps."""
import os,pickle,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
class ParallelMap:
    def __init__(self,n,workers=16):
        self.ranges=[x for x in np.array_split(np.arange(n),min(workers,n)) if len(x)]
        self.pool=ThreadPoolExecutor(max_workers=len(self.ranges))
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.workers=[subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('heightmap118_worker.py')),str(len(x))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env) for x in self.ranges]
        self.cpu={'integrate':0.,'sample':0.}
    def request(self,item):
        w,op,args=item;pickle.dump((op,args),w.stdin,protocol=5);w.stdin.flush();return pickle.load(w.stdout)
    def dispatch(self,op,arguments):
        results=list(self.pool.map(self.request,[(w,op,args) for w,args in zip(self.workers,arguments)]))
        if op in self.cpu:self.cpu[op]+=sum(t for _,t in results)
        return [v for v,_ in results]
    def reset(self,ids):
        if not len(ids):return
        self.dispatch('reset',[(np.flatnonzero(np.isin(rows,ids)).tolist(),) for rows in self.ranges])
    def sample(self,poses,scanner,stamp):
        return np.concatenate(self.dispatch('sample',[(poses[rows],scanner[rows],stamp) for rows in self.ranges]))
    def integrate(self,ids,*args):
        jobs=[(w,'integrate',(np.flatnonzero(np.isin(rows,ids)).tolist(),*[a[rows] for a in args[:-1]],args[-1])) for w,rows in zip(self.workers,self.ranges) if np.isin(rows,ids).any()]
        results=list(self.pool.map(self.request,jobs));self.cpu['integrate']+=sum(t for _,t in results)
    def close(self):
        for w in self.workers:
            pickle.dump(('close',()),w.stdin,protocol=5);w.stdin.flush();w.stdin.close();w.wait();w.stdout.close()
        self.pool.shutdown()
