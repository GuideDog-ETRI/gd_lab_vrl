import sys,unittest,time
import numpy as np
from heightmap117 import BatchMap
from heightmap118 import ParallelMap
from heightmap116 import ElevationMap
class Equivalence(unittest.TestCase):
    def test_serial_parallel(self):
        n=16;rng=np.random.default_rng(118);serial=BatchMap(n);parallel=ParallelMap(n,16)
        depth=rng.uniform(.7,1.3,(n,4,45,80));valid=rng.random(depth.shape)>.1
        ir=np.ones_like(depth)*.8;ir[:,:,:,30:45]=.1
        pos=np.tile([0.,0.,.8],(n,4,1));q=np.tile([1.,0,0,0],(n,4,1))
        ks=np.tile([[45.,0,40],[0,45,22.5],[0,0,1.]],(n,4,1,1));body=np.tile([0.,0,.8],(n,1));bq=q[:,0]
        try:
            for tick in range(5):
                stamp=tick*.1;body[:,0]+=.025;ids=list(range(n)) if tick!=2 else list(range(3,8))
                args=(ids,depth,valid,ir,pos,q,ks,body,bq,stamp)
                serial.integrate(*args);parallel.integrate(*args)
                poses=np.c_[body[:,:2],np.linspace(-.3,.3,n)];scanner=np.full(n,.8)
                np.testing.assert_array_equal(serial.sample(poses,scanner,stamp),parallel.sample(poses,scanner,stamp))
                if tick==3:serial.reset([1,3]);parallel.reset([1,3])
            print('serial vs 16 CPU workers: exact array equality, 5 integrations + subset + reset',flush=True)
        finally:parallel.close()
    def test_transport(self):
        import importlib.util
        from pathlib import Path
        path=Path(__file__).resolve().parents[2]/'src/gd_lab/core/camera_transport.py'
        spec=importlib.util.spec_from_file_location('transport118test',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        def run():
            result=[]
            for seed in (7,123,2024,42):
                t=m.CameraTransport(m.CameraTransportConfig((70,100),(0,150),.05),.01,seed);n=t.next_capture(0);events=[]
                for step in range(1,3001):
                    if step==n:t.capture(step,None);events.append(('c',step,t.dropped));n=t.next_capture(step)
                    for p in t.receive(step):events.append(('r',p.capture_step,p.delivery_step,step))
                result.append(events)
            return result
        a=run();self.assertEqual(a,run());self.assertEqual(len({str(x) for x in a}),4)
if __name__=='__main__':unittest.main(verbosity=2)
