"""Original selector function ASTs; no candidate annotations are read."""
import numpy as np
from runtime import require

def random_query(pool,q,dataset_code,seed,round_,purpose):
    pool=sorted(pool);require(len(pool)==len(set(pool)) and 0<q<=len(pool),'Invalid random pool/count')
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x464D5632,dataset_code,seed,round_,purpose])))
    return [pool[int(i)] for i in rng.choice(len(pool),q,replace=False)]

def farthest(ids,z,labeled,q):
    require(ids==sorted(ids) and len(ids)==len(set(ids)),'IDs not canonical')
    z=np.asarray(z,dtype=np.float64);require(z.shape[0]==len(ids) and np.isfinite(z).all(),'Invalid features')
    pos={v:i for i,v in enumerate(ids)};require(set(labeled)<=set(ids) and labeled,'Invalid reference')
    available=np.ones(len(ids),bool);available[[pos[x] for x in labeled]]=False
    require(0<q<=int(available.sum()),'Invalid farthest query')
    distance=np.full(len(ids),np.inf)
    for id in labeled:distance=np.minimum(distance,1-z@z[pos[id]])
    chosen=[]
    for _ in range(q):
        best=float(distance[available].max());index=int(np.flatnonzero(available & (distance>=best-1e-12))[0]);chosen.append(ids[index]);available[index]=False
        distance=np.minimum(distance,1-z@z[index])
    return chosen
