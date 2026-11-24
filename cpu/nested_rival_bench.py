"""CPU-only benchmark of exclusion proofs over a synthetic scoped AND/OR DAG.

Leaf atoms are available iff at least one *distinct* current-scope receipt is live.
A true queried root excludes every modeled rival (AND) via available proof routes
(OR). False means DEFER here, not that a real-world rival exists.
"""
import argparse, json, random, time
from collections import deque, defaultdict
from pathlib import Path

def ns(): return time.perf_counter_ns()

class Receipts:
    def __init__(self, init):
        self.byid = {}
        self.removed = set()
        self.generation = 1
        self.count = defaultdict(int)
        for row in init: self.change(row)
    def change(self, row):
        if row[0] == "epoch":
            generation = row[1]
            if generation <= self.generation: return None
            self.generation = generation
            self.count.clear()
            for atom, scope, gen in self.byid.values():
                if scope == 1 and gen == generation: self.count[atom] += 1
            return ("epoch", generation)
        op, rid, atom, scope, generation = row
        if op == "add":
            status = (atom, scope, generation)
            if (rid, *status) in self.removed: return None
            if rid in self.byid:
                if self.byid[rid] != status: raise ValueError("receipt identity reused")
                return None
            self.byid[rid] = status
            if scope != 1 or generation != self.generation: return None
            self.count[atom] += 1
            return (atom, True) if self.count[atom] == 1 else None
        if op != "del": raise ValueError("unsupported receipt event")
        # Events reaching this class are already authenticated. An authentic
        # retract binds the complete identity and may precede its matching add.
        key = (rid, atom, scope, generation)
        self.removed.add(key)
        if self.byid.get(rid) != (atom,scope,generation): return None
        del self.byid[rid]
        if scope != 1 or generation != self.generation: return None
        self.count[atom] -= 1
        assert self.count[atom] >= 0
        return (atom, False) if self.count[atom] == 0 else None

class Indexed:
    def __init__(self, graph, n, init):
        self.graph, self.n, self.receipts = graph, n, Receipts(init)
        self.reverse = [[] for _ in graph]
        self.values = [False]*len(graph)
        self.count = [0]*len(graph)
        self.work = 0
        for i in range(n): self.values[i] = self.receipts.count[i] > 0
        for i,(kind, children) in enumerate(graph[n:],n):
            for c in children: self.reverse[c].append(i)
            self.count[i] = sum(self.values[c] for c in children)
            self.values[i] = self.decide(i)
    def decide(self,i):
        kind,c = self.graph[i]
        return self.count[i] > 0 if kind == "any" else self.count[i] == len(c)
    def update(self,row):
        d = self.receipts.change(row)
        if not d: return
        node,new = d
        if node == "epoch":
            for i in range(self.n): self.values[i] = self.receipts.count[i] > 0
            for i,(_,ch) in enumerate(self.graph[self.n:],self.n):
                self.work += len(ch)
                self.count[i] = sum(self.values[c] for c in ch)
                self.values[i] = self.decide(i)
            return
        self.values[node] = new
        # Each queue item is one exact delta. Propagation order may create
        # transient status internally; no effect queries interleave an update.
        q=deque([(node,1 if new else -1)])
        while q:
            node,delta=q.popleft()
            for j in self.reverse[node]:
                self.work += 1
                self.count[j] += delta
                z=self.decide(j)
                if z != self.values[j]:
                    self.values[j]=z
                    q.append((j,1 if z else -1))
    def query(self,r): return self.values[r]

class Backward:
    def __init__(self, graph, n, init):
        self.graph,self.n,self.receipts=graph,n,Receipts(init)
        self.work=0
    def update(self,row): self.receipts.change(row)
    def eval(self, r, proof=False, wins=None):
        memo={}
        def visit(i):
            if i in memo: return memo[i]
            self.work+=1
            if i < self.n:
                ans = self.receipts.count[i] > 0
                result = (ans,{i} if ans else set()) if proof else ans
            else:
                kind,ch=self.graph[i]
                order=ch
                if wins is not None and kind=="any":
                    order=sorted(ch,key=lambda c:-wins.get((i,c),0))
                if not proof:
                    result=(all if kind=="all" else any)(visit(j) for j in order)
                else:
                    used=set()
                    if kind=="all":
                        good=True
                        for j in order:
                            ok,sub=visit(j)
                            if not ok: good=False; break
                            used.update(sub)
                    else:
                        good=False
                        for j in order:
                            ok,sub=visit(j)
                            if ok:
                                good=True; used=set(sub)
                                if wins is not None: wins[(i,j)]+=1
                                break
                    result=(good,used if good else set())
            memo[i]=result
            return result
        return visit(r)
    def query(self,r): return self.eval(r)

class WitnessBank(Backward):
    # Exact sparse positive shortcut, exact backward fallback.
    # Any false root is deliberately NOT cached, so add cannot stale it.
    def __init__(self, graph,n,init,cap=256):
        super().__init__(graph,n,init)
        self.proofs,self.dep,self.wins={},defaultdict(set),defaultdict(int)
        self.cap=cap; self.hits=0; self.invalidated=0; self.dependency_visits=0
    def remove(self,r):
        for a in self.proofs.pop(r):
            self.dep[a].remove(r)
            if not self.dep[a]: del self.dep[a]
    def update(self,row):
        d=self.receipts.change(row)
        if d and d[0] == "epoch":
            # Public scope/epoch changed; old proof cannot be replayed.
            for r in tuple(self.proofs):
                self.dependency_visits += len(self.proofs[r])
                self.remove(r); self.invalidated += 1
        elif d and not d[1]:
            touched=tuple(self.dep.get(d[0],()))
            for r in touched:
                self.dependency_visits += len(self.proofs[r])
                self.remove(r); self.invalidated+=1
    def query(self,r):
        if r in self.proofs: self.hits+=1; return True
        ok,cert=self.eval(r,proof=True,wins=self.wins)
        if ok and len(cert)<=self.cap:
            if len(self.proofs)>=self.cap: self.remove(next(iter(self.proofs)))
            self.proofs[r]=cert
            for a in cert: self.dep[a].add(r)
        return ok

def fixture(seed,n,width,depth,groups,ops,mode):
    rng=random.Random(seed); graph=[("leaf",()) for _ in range(n)]
    prev=list(range(n))
    for layer in range(depth):
        nxt=[]
        for _ in range(width):
            kind="any" if layer%2==0 else "all"
            ch=tuple(rng.sample(prev,3))
            nxt.append(len(graph)); graph.append((kind,ch))
        prev=nxt
    roots=[]
    for _ in range(groups):
        roots.append(len(graph)); graph.append(("all",tuple(rng.sample(prev,2))))
    init=[("add",a,a,1,1) for a in range(n) if rng.random()<.34]
    present={x[2]:x[1] for x in init}
    events=[]
    for i in range(ops):
        update_p={"updates":.72,"rare_queries":.997}.get(mode,.25)
        if rng.random()<update_p:
            a=rng.randrange(n)
            if a in present: op="del"; rid=present.pop(a)
            else: op="add"; rid=n+i; present[a]=rid
            events.append((op,rid,a,1,1))
        else:
            r=rng.choice(roots[:8] if mode=="hot" else roots)
            events.append(("query",r))
    return graph,init,events

def run(model,g,n,init,events):
    t=ns(); obj=model(g,n,init); build=ns()-t
    change=ask=0; ans=[]
    for e in events:
        t=ns()
        if e[0]=="query":
            q=obj.query(e[1]); ask += ns()-t; ans.append(q)
        else:
            obj.update(e);change+=ns()-t
    return ans,dict(build_ms=build/1e6,update_ms=change/1e6,query_ms=ask/1e6,
                    total_ms=(build+change+ask)/1e6,work=obj.work,
                    hits=getattr(obj,"hits",0),invalidated=getattr(obj,"invalidated",0),
                    dependency_visits=getattr(obj,"dependency_visits",0))

def audit():
    g=[("leaf",()),("leaf",()),("leaf",()),("all",(0,1)),("any",(3,2))]
    init=[]
    x=[
     ("add",9,0,1,1),("add",9,0,1,1),("add",10,0,1,1),("add",20,1,1,1),
     ("query",4),("del",9,0,1,1),("query",4),("del",9,0,1,1),("query",4),
     ("del",10,0,1,1),("query",4),("add",30,2,2,1),("query",4),
     ("add",31,2,1,0),("query",4),("add",32,2,1,1),("query",4),
     ("del",32,2,1,1),("query",4),("add",32,2,1,1),("query",4),
     ("add",33,2,1,1),("query",4),("epoch",2),("query",4),
     ("epoch",1),("query",4),("add",34,2,1,2),("query",4),
     ("del",34,2,1,2),("query",4),("del",35,2,1,2),("query",4),
     ("add",35,2,1,2),("query",4),("add",35,2,1,2),("query",4),
     ("del",36,2,2,2),("add",36,2,1,2),("query",4)]
    want=[True,True,True,False,False,False,True,False,False,True,False,False,True,False,False,False,False,True]
    for m in (Indexed,Backward,WitnessBank): assert run(m,g,3,init,x)[0]==want
    return want

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ops",type=int,default=1800)
    p.add_argument("--seeds",type=int,default=4)
    p.add_argument("--seed-start",type=int,default=910231)
    p.add_argument("--output",default="nested_rival_bench_report.json")
    a=p.parse_args()
    if not 1<=a.ops<=8000 or not 1<=a.seeds<=12: p.error("resource guard")
    scores=[]
    for mode in ("hot","cold","updates","rare_queries"):
        for seed in range(a.seed_start,a.seed_start+a.seeds):
            g,init,events=fixture(seed,1800,1400,6,512,a.ops,mode)
            row={"mode":mode,"seed":seed,"nodes":len(g),"queries":sum(e[0]=="query" for e in events)}
            ans=None
            for m in (Backward,Indexed,WitnessBank):
                z,s=run(m,g,1800,init,events)
                if ans is None: ans=z
                else: assert z==ans,(seed,mode,m)
                row[m.__name__]=s
            row["positive"]=sum(ans)
            scores.append(row)
    report={"audit_expected":audit(),"scores":scores,"scope":"Synthetic current-generation DAG only; timed workload receipt generation fixed at 1; generation rotation and authenticated out-of-order retractions are covered by a focused untimed audit, not the workload timings. Recorded solver ordering is fixed. Synthetic runs start near mixed root truth regimes; focused audit checks definitive positive and negative states."}
    Path(a.output).write_text(json.dumps(report,indent=2))
    for mode in ("hot","cold","updates","rare_queries"):
        rows=[x for x in scores if x["mode"]==mode]
        print(mode, "queries",sum(x["queries"] for x in rows),"positive",sum(x["positive"] for x in rows))
        for key in ("Backward","Indexed","WitnessBank"):
            print(key,{k:round(sum(x[key][k] for x in rows)/len(rows),3) for k in rows[0][key]})
if __name__ == "__main__": main()

