"""CPU-only falsification for exact affine box witness search (no network/GPU)."""
import os
for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"): os.environ[k]="1"
import time, math, itertools as it, random, json, pathlib, hashlib, statistics
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
EVAL_SEED=int(os.environ.get('THEORY_SYNTH_SEED','910231'))
B,M,D=10,14,8
START=time.perf_counter()
def guard(): 
    if time.perf_counter()-START>55: raise TimeoutError("55s CPU experiment wall guard")
S={k:np.array(list(it.combinations(range(M),k)),dtype=np.intp) for k in (2,3)}
G={2:np.array([(i,D-i) for i in range(1,D)]),
   3:np.array([(i,j,D-i-j) for i in range(1,D) for j in range(1,D-i)])}
GRIDS={k:np.ascontiguousarray(G[k].T) for k in (2,3)}

def clip1(P,row):
    a,b,c=map(int,row)
    if not P:return P
    Q=[]
    p=P[-1]; dp=a*p[0]+b*p[1]-c*p[2]
    for q in P:
        dq=a*q[0]+b*q[1]-c*q[2]
        if (dp>0)!=(dq>0):
            v=tuple(dp*q[i]-dq*p[i] for i in range(3))
            if v[2]<0:v=tuple(-x for x in v)
            g=math.gcd(math.gcd(abs(v[0]),abs(v[1])),v[2])
            v=tuple(x//g for x in v)
            if not Q or Q[-1]!=v: Q.append(v)
        if dq<=0 and (not Q or Q[-1]!=q): Q.append(q)
        p,dp=q,dq
    if len(Q)>1 and Q[-1]==Q[0]:Q.pop()
    return Q
P0=[(-B,-B,1),(B,-B,1),(B,B,1),(-B,B,1)]
def polygon(rows,prefix=None):
    P=P0 if prefix is None else prefix
    for r in rows:
        P=clip1(P,r)
        if not P:break
    return P

def margins(mat): return -mat[...,2]-B*(abs(mat[...,0])+abs(mat[...,1]))
def verify(rows,inds,weights):
    # Scalar proof checker, independent of vectorized floating ranking.
    if len(inds)!=len(weights) or sum(int(w) for w in weights)!=D or any(int(w)<0 for w in weights):return False
    a,b,c=[sum(int(rows[int(i),j])*int(w) for i,w in zip(inds,weights)) for j in range(3)]
    return -c-B*(abs(a)+abs(b))>0

def sweep(rows,k,sel=None):
    sub=S[k] if sel is None else S[k][sel]
    vals=rows[sub]                      # [supports,k,3]
    combo=np.einsum('skd,kg->sgd',vals,GRIDS[k],optimize=False)
    v=margins(combo)
    z=np.argmax(v,axis=1)
    return v[np.arange(len(v)),z],z

def sparse(row):
    sig=margins(row)
    for k in (2,3):
        best,w=sweep(row,k)
        if best.max()>0:
            i=int(best.argmax())
            assert verify(row,S[k][i],G[k][w[i]])
            return k
    return 1 if sig.max()>0 else 0

def draw(family,idx,rg):
    # Generate same-center paired hypotheses; good/bad ID randomized outside row generation.
    ox,oy=rg.randint(-3,3),rg.randint(-3,3)
    if family in ("train_pair","scaled_pair"):
        while (u:=(rg.randint(-4,4),rg.randint(-4,4)))==(0,0):pass
        normals=[u,(-u[0],-u[1])]
    elif family=="train_axis":
        sx,sy=rg.choice([-1,1]),rg.choice([-1,1]); normals=[(sx,0),(0,sy),(-sx,-sy)]
    else:
        while True:
            u=(rg.randint(-4,4),rg.randint(-4,4));v=(rg.randint(-4,4),rg.randint(-4,4))
            if u[0]*v[1]-u[1]*v[0] not in (0,1,-1) and all(u) and all(v) and all((u[0]+v[0],u[1]+v[1])):break
        normals=[u,v,(-u[0]-v[0],-u[1]-v[1])]
    sat_control=family=="boundary_feasible"
    q,r= [],[]
    for a,b in normals:
        good=rg.randint(0,3)
        bad=0 if sat_control else -rg.randint(1,3)
        scale=rg.randint(1,2) if family.startswith("train") else rg.randint(1,5)
        core=a*ox+b*oy
        q.append((a*scale,b*scale,(core+good)*scale))
        r.append((a*scale,b*scale,(core+bad)*scale))
    for _ in range(M-len(q)):
        while True:
            a,b=rg.randint(-5,5),rg.randint(-5,5)
            if (a,b)!=(0,0):break
        s=rg.randint(1,3)
        c=a*ox+b*oy+rg.randint(0,6)
        q.append((a*s,b*s,c*s));r.append((a*s,b*s,c*s))
    order=rg.sample(range(M),M)
    qa=np.array([q[i] for i in order],dtype=np.int64);rb=np.array([r[i] for i in order],dtype=np.int64)
    return (qa,rb) if rg.randrange(2) else (rb,qa)

def dataset(fams,seed,wanted):
    rg=random.Random(seed); cases=[]; attempts=0
    for f in fams:
        count=0
        while count<wanted:
            guard()
            if attempts>20*len(fams)*wanted: raise RuntimeError("relax declared split or increase sampling cap")
            attempts+=1
            p=draw(f,attempts,rg)
            yt=[not bool(polygon(r)) for r in p]
            if yt==[True,True] or not any(yt) and f!="boundary_feasible":continue
            ks=[sparse(r) for r in p]
            if any(margins(r).max()>0 for r in p):continue
            if f in ("train_axis","oblique_triple") and any(h and k==2 for h,k in zip(yt,ks)):continue
            if f=="scaled_pair" and any(h and k not in (0,2) for h,k in zip(yt,ks)):continue
            # Keep grid misses; exact LP and grid support are intentionally separate targets.
            for h,r in enumerate(p):
                pre=polygon(r[:7]); assert (not bool(polygon(r[7:],pre)))==yt[h]
                cases.append({"fam":f,"rows":r,"truth":yt[h],"sparse":ks[h],"prefix":pre,
                             "instance":(f,count),"hyp":h})
            count+=1
    return cases, attempts

def rankbase(rows,k):
    vals=rows[S[k]]
    p=vals.sum(axis=1)
    # Optimistic geometry-weight proxy using individually normalized normals.
    den=np.maximum(1,abs(vals[:,:,:2]).sum(axis=2))
    q=(vals/den[:,:,None]).sum(axis=1)
    m0=margins(p)/k
    m1=margins(q)/k
    return np.column_stack((m0,m1))

def feats(rows,k,base):
    vals=rows[S[k]]
    scale=np.maximum(1,abs(vals[:,:,:2]).sum(axis=2))
    direct=margins(vals)
    ratios=(vals[:,:,2]/scale)
    normed=vals[:,:,:2]/scale[:,:,None]
    pairs=list(it.combinations(range(k),2))
    align=np.stack([(normed[:,i]*normed[:,j]).sum(axis=1) for i,j in pairs],axis=1)
    cancellation=np.abs(normed.sum(axis=1)).sum(axis=1)
    return np.column_stack((base,scale.min(axis=1),scale.max(axis=1),ratios.min(axis=1),
       ratios.max(axis=1),direct.max(axis=1),cancellation,align.min(axis=1),align.max(axis=1)))

def fit(train):
    coef={}
    for k in (2,3):
        A=[]; z=[]
        for case in train:
            rows=case["rows"]; z0,_=sweep(rows,k)
            g=rankbase(rows,k);f=feats(rows,k,g)
            # Regression standardizes features below. Labels are best exact margin for support
            # over declared grid, not planted row IDs or hypothesis labels.
            A.append(f);z.append(z0/D)
        X=np.concatenate(A);y=np.concatenate(z)
        mu=X.mean(axis=0);s=X.std(axis=0)+1e-9
        T=np.column_stack(((X-mu)/s,np.ones(len(y))))
        # common offline procedure; emphasize margins near/above the cutoff
        w=np.where(y>0,30.,np.where(y>-2,4.,1.))
        a=np.linalg.solve(T.T@(w[:,None]*T)+1e-2*np.eye(T.shape[1]),T.T@(w*y))
        coef[k]=(mu,s,a)
    return coef

def search(row,K,kind,coef):
    calls=0
    for k in (2,3):
        base=rankbase(row,k)
        if kind=="learned":
            mu,s,a=coef[k]
            score=((feats(row,k,base)-mu)/s)@a[:-1]+a[-1]
        elif kind=="geom":
            # Properly normalized opposite-direction and uniform grid scouts.
            score=np.maximum(base[:,0]/np.maximum(1,np.abs(row[S[k],:2]).sum(axis=(1,2))/k),base[:,1])
        elif kind=="position": score=-np.arange(len(base))
        else: raise KeyError(kind)
        # stable deterministic ties. Any selected support gets exact optimal grid check.
        inds=np.argsort(-score,kind="stable")[:K]
        vals,ws=sweep(row,k,inds)
        calls+=len(inds)*len(G[k])
        if vals.max()>0:
            u=int(vals.argmax());cert=(S[k][inds[u]],G[k][ws[u]])
            assert verify(row,*cert)
            return True,calls
    return False,calls

def exact_sparse(row):
    calls=0
    for k in (2,3):
        vals,ws=sweep(row,k)
        calls+=len(vals)*len(G[k])
        if vals.max()>0:
            u=int(vals.argmax()); assert verify(row,S[k][u],G[k][ws[u]])
            return True,calls
    return False,calls

def methods(coef):
    z={}
    for t in (2,8,32):
        for k in ("learned","geom","position"):
            z[f"{k}-{t}"]=lambda r,c,t=t,k=k: search(r,t,k,coef)
    z["sparse-exact"]=lambda r,c:exact_sparse(r)
    z["lp-rational-cold"]=lambda r,c:(not bool(polygon(r)),0)
    z["lp-rational-warm"]=lambda r,c:(not bool(polygon(r[7:],c["prefix"])),0)
    return z

def perf(cases, met, repetitions):
    out={}
    for name,method in met.items():
        guard()
        rows=[]
        for case in cases:
            assert case["rows"].shape==(M,3)
            times=[]
            for _ in range(repetitions):
                t=time.perf_counter_ns()
                dec,calls=method(case["rows"],case)
                times.append(time.perf_counter_ns()-t)
            rows.append((str(case["fam"]),bool(case["truth"]),int(case["sparse"]),bool(dec),
                        int(calls),int(statistics.median(times))))
        out[name]=rows
    return out

def summarize(out):
    summary={}
    for name,rows in out.items():
        d={}
        for fam in sorted({r[0] for r in rows}|{"ALL"}):
            z=[r for r in rows if fam=="ALL" or r[0]==fam]
            elig=[r for r in z if r[1]]
            expr=[r for r in elig if r[2]]
            safe=[r for r in z if not r[1]]
            d[fam]={"lp_positive":len(elig),"grid_positive":len(expr),"feasible_neg":len(safe),
              "positive_found":sum(r[3] for r in elig),"false_excluded":sum(r[3] for r in safe),
              "median_us":round(float(np.median([r[-1] for r in z]))/1000,2),
              "p90_us":round(float(np.percentile([r[-1] for r in z],90))/1000,2),
              "avg_checked":round(float(np.mean([r[-2] for r in z])),1)}
        summary[name]=d
    return summary

def latency_measured(out):
    return {str(b):{name:{"timely_correct_positive":sum(r[1] and r[3] and r[-1]<=b*1000 for r in rows),
                         "timely_decision_total":sum(r[-1]<=b*1000 for r in rows)}
             for name,rows in out.items()} for b in (40,75,125,200,350,600,1000)}

def main():
    train,a=dataset(("train_pair","train_axis"),126131,55)
    t=time.perf_counter(); coef=fit(train)
    train_ms=(time.perf_counter()-t)*1000
    hold,b=dataset(("scaled_pair","oblique_triple","boundary_feasible"),EVAL_SEED+101,40)
    met=methods(coef); t=time.perf_counter(); results=perf(hold,met,3)
    # Sensitivity controls: permute every row order; exact answer invariant.
    rng=np.random.default_rng(78337)
    drift=[]
    for c in hold:
        r=c["rows"][rng.permutation(M)];prefix=polygon(r[:7])
        assert (not bool(polygon(r[7:],prefix)))==c["truth"]
        drift.append(dict(c,rows=r,prefix=prefix))
    sensitivity=perf(drift,{k:v for k,v in met.items() if k in ("learned-8","geom-8","sparse-exact","lp-rational-warm")},1)
    out={"fit_train_hypotheses":len(train),"test_hypotheses":len(hold),
         "graph":"two-dimensional exact box, 14 integer constraints, D=8, k<=3",
         "generation_attempts":[a,b],"fit_ms":round(train_ms,2),
         "solver_note":"LP uses exact incremental rational 2D half-plane clipping, no floating infeasibility assertion; warm excludes prep before observed frontier",
         "budget_note":"fixed candidate budgets also show measured equal-budget timely-coverage frontier; timing includes proposals/exact checks but excludes generation and warm prefix",
         "summary":summarize(results),"latency_frontier":latency_measured(results),
         "permutation_control":summarize(sensitivity),
         "wall_seconds":round(time.perf_counter()-START,3),"held_raw":results}
    (HERE/"witness_search_report.json").write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ("held_raw","summary","permutation_control","latency_frontier")},indent=2))
    for n,d in out["summary"].items():print(n,d["ALL"],[(f,dd["positive_found"],dd["grid_positive"]) for f,dd in d.items() if f!="ALL"])
    print("frontier 125us",out["latency_frontier"]["125"])
if __name__=="__main__": main()
