"""Controlled-language, CPU-only sequence-to-address binding. No external actions."""
import os
for v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"): os.environ[v]="1"
import hashlib,json,random,re,time,pathlib,math
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
EVAL_SEED=int(os.environ.get('THEORY_SYNTH_SEED','910231'))
T0=time.perf_counter()
def guard():
    if time.perf_counter()-T0>52:raise TimeoutError("52s isolated CPU budget")
def lex(s):
    if isinstance(s,bytes):
        try:s=s.decode("ascii")
        except UnicodeDecodeError:return None
    return re.findall(r"[a-z]+|[^\s]",s.lower()) if s.isascii() else None
SCOPES=["atrium","depot","harbor","foundry"]
UNITS=["wren","comet","lark","sable"]
OPEN_SCOPES=["terrace","orchard"]
OPEN_UNITS=["heron","otter"]
FIELDS=["ready","cleared"]
TRAIN=[
 "tell me if {u} at {s} is {f}; use {v} at {t} only for comparison.",
 "using {v} at {t} for comparison, is {u} at {s} {f}?",
 "check whether {u} in {s} is {f}, not {v} in {t}.",
 "do not use {v} in {t}; tell me if {u} in {s} is {f}.",
 "compare with {v} in {t}; check whether {u} in {s} is {f}.",
 "is {u} at {s} {f}? keep {v} at {t} only for comparison.",
]
HELD=[
 "using {v} in {t} for comparison; check whether {u} at {s} is {f}.",
 "do not use {v} at {t}; is {u} in {s} {f}?",
]
BADTRAIN=[
 "describe the paint at {s} for {u}.",
 "delete {v} in {t}; use {u} in {s} as a comparison.",
 "which of {u} at {s} and {v} at {t} is {f}?",
 "ignore every request about {u} at {s}.",
]
BADHELD=[
 ('quote','quote this: tell me if {u} at {s} is {f}; use {v} at {t} only for comparison.'),
 ('append','check whether {u} in {s} is {f}, not {v} in {t}. delete {v} in {t}.'),
 ('negated','do not check whether {u} in {s} is {f}, not {v} in {t}.'),
 ('correction','tell me if {u} at {s} is {f}; use {v} at {t} only for comparison. correction: use {v}.'),
 ('unknownverb','publish whether {u} in {s} is {f}, not {v} in {t}.'),
]
def form(tpl,s,u,t,v,f):
    return tpl.format(**locals())
def example(tpl,s,u,t,v,f,*,valid=True):
    raw=form(tpl,s,u,t,v,f)
    toks=lex(raw)
    return (raw,toks,toks.index(s) if valid else 0,toks.index(u) if valid else 0,
            FIELDS.index(f) if valid else 2,(s,u,t,v,f))
def noisy_draw(r,domains=(SCOPES,UNITS)):
    sc,un=domains
    s,t=r.sample(sc,2);u,v=r.sample(un,2);f=r.choice(FIELDS)
    return s,u,t,v,f
def exclusion(s,u,f):
    # Withhold one entire scope/unit/field slice, but every symbol, pair and field
    # appears elsewhere. Independent of state variables and directory indices.
    return (s=="atrium" and u=="wren" and f=="cleared") or (s=="depot" and u=="comet" and f=="ready")
def train_rows(seed=4966,n=2450):
    r=random.Random(seed);xs=[]
    while len(xs)<n:
        s,u,t,v,f=noisy_draw(r)
        if exclusion(s,u,f):continue
        tpl=r.choice(TRAIN)
        xs.append(example(tpl,s,u,t,v,f))
    xs += [example(r.choice(BADTRAIN),*noisy_draw(r),valid=False) for _ in range(525)]
    r.shuffle(xs)
    return xs
def test_rows(seed=EVAL_SEED+131):
    r=random.Random(seed); collections={}
    # Symmetric matched swap pairs. Both members reuse exact non-slot wording.
    for label,styles,domains,draw,acc in [
        ("seen_forms_holdout",TRAIN,(SCOPES,UNITS),210,lambda x:not exclusion(x[0],x[1],x[4])),
        ("withheld_style",HELD,(SCOPES,UNITS),210,lambda x:True),
        ("withheld_scope_entity_field",TRAIN,(SCOPES,UNITS),100,lambda x:exclusion(x[0],x[1],x[4])),
        ("new_scope_and_unit_spellings",HELD,(OPEN_SCOPES,OPEN_UNITS),100,lambda x:True),
    ]:
        xx=[]
        while len(xx)<draw:
            s,u,t,v,f=vals=noisy_draw(r,domains)
            if not acc(vals):continue
            tpl=r.choice(styles)
            k=example(tpl,s,u,t,v,f)
            q=example(tpl,t,v,s,u,f)
            xx.extend((k,q))
        collections[label]=xx
    for label,tpl in BADHELD:
        collections["unsupported_"+label]=[example(tpl,*noisy_draw(r),valid=False) for _ in range(100)]
    return collections

class Lexicon:
    def __init__(self,x):
        self.v=["<PAD>","<UNK>"]+sorted({s for y in x for s in y[1]})
        self.idx={s:i for i,s in enumerate(self.v)}
    def pack(self, x, maxlen=None):
        n=len(x); m=maxlen or max(len(v[1]) for v in x)
        w=np.zeros((m,n),np.int32);y=np.zeros((3,n),np.int32)
        for i,row in enumerate(x):
            for j,k in enumerate(row[1]): w[j,i]=self.idx.get(k,1)
            y[:,i]=row[2:5]
        return w,y

def softmax(z,axis):
    p=np.exp(z-np.max(z,axis=axis,keepdims=True))
    return p/np.sum(p,axis=axis,keepdims=True)

class BiRNN:
    def __init__(self,V,seed,emb=18,h=24):
        self.r=np.random.default_rng(seed); self.h=h
        def mk(*sh):return self.r.normal(0,.18,sh).astype(np.float32)
        self.P={"E":mk(V,emb),"fX":mk(emb,h),"fR":mk(h,h),"bX":mk(emb,h),"bR":mk(h,h),
                "A":mk(2*h,2),"Q":mk(2*h,3)}
        self.P["E"][0]=0
        self.M={k:np.zeros_like(v) for k,v in self.P.items()}
        self.V={k:np.zeros_like(v) for k,v in self.P.items()};self.it=0
    def forward(self, w, dropout=False):
        p=self.P;r=self.r
        if dropout:
            w=np.where((w>1)&(r.random(w.shape)<.08),1,w)
        e=p["E"][w];m=(w!=0)
        T,B=w.shape; d=self.h
        f=np.zeros((T+1,B,d),np.float32); b=np.zeros((T+1,B,d),np.float32)
        ef=e@p["fX"]; eb=e@p["bX"]
        for t in range(T):f[t+1]=np.tanh(ef[t]+f[t]@p["fR"])*m[t,:,None]
        for t in range(T-1,-1,-1):b[t]=np.tanh(eb[t]+b[t+1]@p["bR"])*m[t,:,None]
        z=np.concatenate((f[1:],b[:-1]),axis=-1)
        links=np.where(m[:,:,None],z@p["A"],-1e4)
        count=m.sum(axis=0,dtype=np.float32)[:,None]
        pooled=z.sum(axis=0)/count
        ops=pooled@p["Q"]
        return w,m,e,f,b,z,pooled,count,links,ops
    def batch(self,w,y):
        w,m,e,f,b,z,pool,count,L,O=self.forward(w,True);p=self.P
        n=w.shape[1];ii=np.arange(n)
        ptr=softmax(L,0); op=softmax(O,1)
        g1=ptr.copy();g1[y[0],ii,0]-=1;g1[y[1],ii,1]-=1
        g1*=(y[2]!=2)[None,:,None]/n
        g2=op.copy();g2[ii,y[2]]-=1;g2/=n
        g={"A":z.reshape(-1,z.shape[-1]).T@g1.reshape(-1,2),"Q":pool.T@g2}
        dz=g1@p["A"].T+((g2@p["Q"].T)/count)[None,:,:]*m[:,:,None]
        E=e.reshape(-1,e.shape[-1]); T=len(e); H=self.h
        def back(masked,states,wx,wr,backward):
            da=np.zeros((T,n,H),np.float32);carry=np.zeros((n,H),np.float32)
            order=range(T) if backward else range(T-1,-1,-1)
            for t in order:
                current=states[t] if backward else states[t+1]
                dt=(masked[t]+carry)*(1-current*current)*m[t,:,None]
                da[t]=dt;carry=dt@wr.T
            prev=states[1:] if backward else states[:-1]
            d=da.reshape(-1,H)
            return E.T@d,prev.reshape(-1,H).T@d,da@wx.T
        g["fX"],g["fR"],gx=back(dz[:,:,:H],f,p["fX"],p["fR"],False)
        g["bX"],g["bR"],gb=back(dz[:,:,H:],b,p["bX"],p["bR"],True)
        ge=np.zeros_like(p["E"]);np.add.at(ge,w.reshape(-1),(gx+gb).reshape(-1,E.shape[-1]));ge[0]=0;g["E"]=ge
        self.it+=1;t=self.it
        for k,u in g.items():
            u=np.clip(u,-1,1); self.M[k]=.9*self.M[k]+.1*u;self.V[k]=.999*self.V[k]+.001*u*u
            p[k]-=.008*(self.M[k]/(1-.9**t))/(np.sqrt(self.V[k]/(1-.999**t))+1e-7)
        return float(1e-9 + np.mean(-np.log(op[ii,y[2]]+1e-9))
             +np.mean((-np.log(ptr[y[0],ii,0]+1e-9)-np.log(ptr[y[1],ii,1]+1e-9))*(y[2]!=2)))
    def predict(self,w):
        L,O=self.forward(w)[-2:]
        return np.vstack((L[:,:,0].argmax(0),L[:,:,1].argmax(0),O.argmax(1)))
    def fit(self,w,y,epochs=20):
        loss=[]
        for k in range(epochs):
            guard(); out=[]
            perm=self.r.permutation(w.shape[1])
            for j in range(0,len(perm),80):
                ii=perm[j:j+80]; out.append(self.batch(w[:,ii],y[:,ii]))
            loss.append(round(float(np.mean(out)),5))
        return loss

def grammar_reference(raw):
    ws=lex(raw)
    if ws is None:return None
    # Reference grammar declared before training, including held forms. It never
    # executes in BiRNN forward or in assembling the learned prediction.
    for tpl in TRAIN+HELD:
        marker={"s":"SLOTSC","t":"DECOYSC","u":"SLOTUN","v":"DECOYUN","f":"FIELD"}
        pattern=lex(tpl.format(**marker)); spec=dict()
        if len(pattern)!=len(ws):continue
        okay=True
        for i,(p,q) in enumerate(zip(pattern,ws)):
            if p in map(str.lower,marker.values()):spec[p]=(q,i)
            elif p!=q:okay=False;break
        if okay and spec["field"][0] in FIELDS:
            return (spec["slotsc"][1],spec["slotun"][1],FIELDS.index(spec["field"][0]))
    return None

class Baseline:
    def __init__(self,rs,window):
        self.window=window;self.count={};self.total={};self.oc={};self.oo=np.full(3,1.)
        for r in rs:
            toks=r[1];c=r[4];self.oo[c]+=1
            for w in set(toks):
                self.oc.setdefault(w,np.full(3,1.))[c]+=1
            if c==2:continue
            for k in range(2):
                for i in range(len(toks)):
                    for h in range(-window,window+1):
                        key=(k,h,toks[i+h] if 0<=i+h<len(toks) else "<BOUND>")
                        self.total[key]=self.total.get(key,0)+1
                        if i==r[2+k]:self.count[key]=self.count.get(key,0)+1
    def one(self,ws):
        logits=np.log(self.oo/self.oo.sum())
        for w in set(ws):
            if w in self.oc:logits+=np.log(self.oc[w]/self.oo)
        cls=int(logits.argmax());ptr=[]
        for k in range(2):
            val=np.zeros(len(ws))
            for i in range(len(ws)):
                for h in range(-self.window,self.window+1):
                    key=(k,h,ws[i+h] if 0<=i+h<len(ws) else "<BOUND>")
                    val[i]+=np.log((self.count.get(key,0)+.05)/(self.total.get(key,0)+1.))
            ptr.append(int(val.argmax()))
        return *ptr,cls

def evaluate_fn(fn,rs):
    return [fn(r[1]) for r in rs]
def best_known(report):
    pass

# Independent reference to previously Fitted data. The bank gets only
# authenticated canonical typed events, not any request tokens or gold answer.
BINDER=json.loads((HERE/"minimal_observer_coefficients.json").read_text())["rows_by_token_id"]
def protected_step(x,e,d):
    if e==4:return d
    a,b=BINDER[e];p=a*x+b
    y=(1 if p>=0 else -1)*max(0,abs(p)-5)
    return max(-3,min(3,max(-3,min(3,y))+d))
def bind_fact(q,r):
    x=0
    xs=[1 if q else 3] # authoritative initial write
    xs += [2,0,2,0,2] # net same fact, recurrent state changes twice
    for e in xs:x=protected_step(x,e,r.choice((-1,0,1)))
    return (1 if x>=2 else 0 if x<=-2 else None)
def bank_for(sc,un,seed):
    rng=random.Random(seed)
    keys=[(s,u) for s in sc for u in un]
    rng.shuffle(keys)
    c={}
    for j,key in enumerate(keys):
        f=[rng.randrange(2),rng.randrange(2)]
        c[key]=(j,95100+17*j,[bind_fact(q,rng) for q in f],f)
    return c
def raw_ast(pointer,ws,bank):
    if pointer is None:return None
    si,ui,field=pointer
    if field>1:return None
    s,u=ws[si],ws[ui]; rec=bank.get((s,u))
    if rec is None or isinstance(rec,list):return None
    return ("READ",rec[0],rec[1],field,rec[2][field]) if rec[2][field] is not None else None
def sink(candidate,allow,epoch,issued):
    # Test-only permission step, independent of discovery. Never repairs input.
    return candidate if candidate is not None and allow and epoch==issued else None
def metrics(x,pred):
    psum=0; valid=0; correct=0; raw_n=0; raw_correct=0; unsupported=0; admits=0
    flips=0; auth_test=0; rename_ok=0; erase_block=0; alias_block=0
    example_pairs=[]
    for j,(orig,ws,si,ui,op,v) in enumerate(x):
        s,u,t,w,f=v; p=pred[j]
        sc=SCOPES+OPEN_SCOPES;un=UNITS+OPEN_UNITS
        b=bank_for(sc,un,26000+j)
        a=raw_ast(p,ws,b)
        if op<2:
            valid+=1
            info=b[s,u]; target=("READ",info[0],info[1],op,info[3][op])
            p_good=p== (si,ui,op); good=a==target
            psum+=int(p_good);correct+=int(good);raw_n+=int(a is not None)
            if good:
                alt=dict(b); tf=info[3].copy();tf[op]^=1
                alt[s,u]=(info[0],info[1],[bind_fact(v,random.Random(123+j)) for v in tf],tf)
                flip=raw_ast(p,ws,alt)
                flips+=int(flip==target[:4]+(1-target[4],))
                auth_test+= int(sink(a,True,6,6)==a and sink(a,False,6,6) is None and sink(a,True,6,5) is None and raw_ast(p,ws,b)==a)
                ren=dict(b);ren[s,u]=(314159+j,271828+j,info[2],info[3])
                rename_ok+=int(raw_ast(p,ws,ren)==("READ",314159+j,271828+j,op,target[-1]))
                er=list(info[2]);er[op]=None
                erased=dict(b);erased[s,u]=(info[0],info[1],er,info[3])
                erase_block+=int(raw_ast(p,ws,erased) is None)
                collision=dict(b);collision[s,u]=[info,info]
                alias_block+=int(raw_ast(p,ws,collision) is None)
            example_pairs.append(good)
        else:
            unsupported+=1;raw_n+=int(a is not None);admits+=int(sink(a,True,6,6) is not None)
    return dict(n=len(x),valid=valid,selector_exact=psum,raw_ast_exact=correct,raw_candidates=raw_n,
        both_matched_pairs_correct=sum(a and b for a,b in zip(example_pairs[::2],example_pairs[1::2])),
        same_text_evidence_flip_correct=flips, authority_flip_changes_permission_only=auth_test,
        opaque_directory_ids_permuted_correctly=rename_ok,lawful_numeric_reset_blocks=erase_block,
        ambiguous_directory_alias_blocked=alias_block,unsupported=unsupported,permissive_sink_false_admits=admits)

def main():
    rows=train_rows();lx=Lexicon(rows);w,y=lx.pack(rows)
    tests=test_rows()
    m=BiRNN(len(lx.v),96133)
    t=time.perf_counter();loss=m.fit(w,y,epochs=22); fit_sec=time.perf_counter()-t
    # Float recurrent forward only in this experiment. Not a Lean-checked export.
    np.savez(HERE/"language_binding_checkpoint.npz",**m.P,lexicon=np.array(lx.v))
    checkpoint=(HERE/"language_binding_checkpoint.npz").read_bytes()
    v=m.predict(w)
    systems={"rnn":lambda r:[tuple(map(int,z)) for z in m.predict(lx.pack(r)[0]).T]}
    for key,win in [("bag",0),("local_context",2)]:
        base=Baseline(rows,win);systems[key]=lambda r,b=base:evaluate_fn(b.one,r)
    systems["exact_grammar"]=lambda r:[grammar_reference(r0[0]) for r0 in r]
    stats={}
    for name,fn in systems.items():
        guard()
        stats[name]={}
        for fam,r in tests.items():
            t=time.perf_counter();pr=fn(r)
            psec=time.perf_counter()-t
            s=metrics(r,pr)
            s["parse_us_per_example_batch_average"]=round(psec*1e6/len(r),2)
            stats[name][fam]=s
    out={"seed":96133,"train_count":len(rows),"fit_loss":loss,"train_selector_exact":int(np.count_nonzero(np.all(v[:,y[2]<2]==y[:,y[2]<2],axis=0))),
         "train_supported":int(np.count_nonzero(y[2]<2)),"train_operation_accuracy":float(np.mean(v[2]==y[2])),
         "fit_sec":round(fit_sec,3),"elapsed_sec":round(time.perf_counter()-T0,3),"checkpoint_sha256":hashlib.sha256(checkpoint).hexdigest(),
         "boundaries":"Opaque IDs supplied only by exact dictionary dereference of learned chosen spans. Evidence typed/authenticated fixture. Candidate AST observed before a toy scope/epoch sink. Float RNN not certified. Surfaces not supported by grammar may still be spuriously admitted.",
         "results":stats}
    (HERE/"language_binding_report.json").write_text(json.dumps(out,indent=2))
    print({k:v for k,v in out.items() if k not in ("results","fit_loss")})
    for k,r in stats.items():print(k,[(f,t["raw_ast_exact"],t["n"],t["permissive_sink_false_admits"]) for f,t in r.items()])
if __name__=="__main__":main()
