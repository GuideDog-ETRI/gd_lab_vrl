"""Pure CPU persistent map shard. No torch/Isaac import."""
import pickle,sys,time
from heightmap117 import BatchMap
inp=sys.stdin.buffer;out=sys.stdout.buffer
maps=BatchMap(int(sys.argv[1]))
while True:
    try:op,args=pickle.load(inp)
    except EOFError:break
    if op=='close':break
    tick=time.process_time()
    value=getattr(maps,op)(*args)
    pickle.dump((value,time.process_time()-tick),out,protocol=5);out.flush()
