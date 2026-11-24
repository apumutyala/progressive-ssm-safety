"""Independent no-overlap evaluation; loads the already-frozen first model only."""
import json,hashlib,pathlib,itertools as it,random,time
import numpy as np
import language_binding_cpu as s
HERE=pathlib.Path(__file__).resolve().parent
sha=hashlib.sha256((HERE/"language_binding_checkpoint.npz").read_bytes()).hexdigest()
safe=np.load(HERE/"language_binding_checkpoint.npz",allow_pickle=False)
tr=s.train_rows();ts=set(r[0] for r in tr)
lx=s.Lexicon(tr)
assert lx.v==list(safe["lexicon"])
m=s.BiRNN(len(lx.v),96133)
for key in m.P:m.P[key]=safe[key]
basic=s.test_rows()
audit_original={k:{"n":len(r),"unique":len({x[0] for x in r}),"train_overlap_count":sum(x[0] in ts for x in r),
                   "both_of_pairs_withheld_triple":sum(all(s.exclusion(y[5][0],y[5][1],y[5][4]) for y in p)
                        for p in zip(r[::2],r[1::2]))} for k,r in basic.items() if not k.startswith("unsupported")}
r=random.Random(s.EVAL_SEED+173)
s.OPEN_SCOPES=["terrace","orchard","viaduct","lounge","pier"]
s.OPEN_UNITS=["heron","otter","finch","panda","moose"]
def distinct_pairs(styles,sc,un,n,choose):
    # Both members disjoint from train and other test pairs.
    gathered=[];seen=set()
    for _ in range(30000):
        ss,u,t,v,f=s.noisy_draw(r,(sc,un))
        if not choose(ss,u,t,v,f):continue
        pat=r.choice(styles)
        x=s.example(pat,ss,u,t,v,f);y=s.example(pat,t,v,ss,u,f)
        if any(z[0] in ts or z[0] in seen for z in [x,y]):continue
        gathered.extend([x,y]);seen|={x[0],y[0]}
        if len(gathered)>=n:return gathered
    raise ValueError(("could not sample declarative pairs",n,len(gathered)))
tests={
 "new_full_trajectory_surface_strings":distinct_pairs(s.TRAIN,s.SCOPES,s.UNITS,100,lambda *_:True),
 "held_syntax_no_duplicate":distinct_pairs(s.HELD,s.SCOPES,s.UNITS,200,lambda *_:True),
 "new_names_no_duplicate":distinct_pairs(s.HELD,s.OPEN_SCOPES,s.OPEN_UNITS,200,lambda *_:True)}
# The first primary stratum had two triples excluded on different fields so
# their mirror cannot both be in the excluded slice. Evaluate only actual withheld primary examples.
seen=set();comp=[]
for _ in range(4000):
    ss,u,t,v,f=vals=s.noisy_draw(r)
    if not s.exclusion(ss,u,f):continue
    x=s.example(r.choice(s.TRAIN),*vals)
    assert x[0] not in ts
    if x[0] in seen:continue
    seen.add(x[0]);comp.append(x)
    if len(comp)==90:break
tests["strictly_withheld_primary_triple"]=comp
for label,tpl in s.BADHELD:
    rs={}; 
    while len(rs)<100:
        x=s.example(tpl,*s.noisy_draw(r),valid=False)
        assert x[0] not in ts
        rs[x[0]]=x
    tests["unsupported_"+label]=list(rs.values())

def bind_state(q,r):
    x=0
    for e in [1 if q else 3,2,0,2,0,2]:
        x=s.protected_step(x,e,r.choice((-1,0,1)))
    return x
def decode(x):return 1 if x>=2 else 0 if x<=-2 else None

def bank(n, ss, uu):
    rr=random.Random(n); ls=list(it.product(ss,uu));rr.shuffle(ls); d={}
    for j,k in enumerate(ls):
        fact=[rr.randrange(2),rr.randrange(2)]
        state=[bind_state(q,rr) for q in fact]
        d[k]=(j,9141+13*j,state,fact)
    return d

def propose(pred,ws,d):
    if pred is None:return None
    si,ui,ff=pred
    if ff not in (0,1):return None
    x=d.get((ws[si],ws[ui]))
    if x is None or isinstance(x,list):return None
    f=decode(x[2][ff])
    if f is None:return None
    return ("READ",x[0],x[1],ff,f)

def evals(name,x,ptr):
    rec=dict(total=len(x),raw_exact=0,eligible=0,false_admit=0,raw_pointer_exact=0,auth_changes_only_permission=0,
             live_numeric_evidence_intervention=0,actual_revoke_invalidation=0,opaque_reordering=0,ambiguous_alias=0)
    ex=[]; valid=[]
    for k,(row,p) in enumerate(zip(x,ptr)):
        raw,ws,si,ui,f,b=row;ss,u,t,v,_=b
        d=bank(3526+k,s.SCOPES+s.OPEN_SCOPES,s.UNITS+s.OPEN_UNITS)
        a=propose(p,ws,d)
        if f==2:
            rec["false_admit"]+=int(s.sink(a,True,1,1) is not None)
            continue
        rec["eligible"]+=1
        q=d[ss,u]; target=("READ",q[0],q[1],f,q[3][f])
        yes=(a==target)
        rec["raw_exact"]+=int(yes); valid.append(yes)
        rec["raw_pointer_exact"]+=int(p==(si,ui,f))
        if not yes and len(ex)<3:
            ex.append({"request":raw,"predicted_spans":None if p is None else [ws[p[0]],ws[p[1]],p[2]],
                       "expected_spans":[ss,u,f]})
        if not yes:continue
        rev=[s.protected_step(w,4,1) if j==f else w for j,w in enumerate(q[2])]
        alt=dict(d);alt[ss,u]=(*q[:2],rev,q[3])
        rec["actual_revoke_invalidation"]+=int(propose(p,ws,alt) is None)
        cur=[s.protected_step(w,0,0) if j==f else w for j,w in enumerate(q[2])]
        alt=dict(d);alt[ss,u]=(*q[:2],cur,q[3])
        rec["live_numeric_evidence_intervention"]+=int(propose(p,ws,alt)==(*target[:4],1-target[4]))
        alt=dict(d);alt[ss,u]=(1000+k,9000+k,q[2],q[3])
        rec["opaque_reordering"]+=int(propose(p,ws,alt)==("READ",1000+k,9000+k,f,target[-1]))
        alt=dict(d);alt[ss,u]=[q,q]
        rec["ambiguous_alias"]+=int(propose(p,ws,alt) is None)
        rec["auth_changes_only_permission"]+=int(s.sink(a,False,1,1) is None and s.sink(a,True,1,0) is None and propose(p,ws,d)==a)
    rec["ex_errors"]=ex
    if name in ("new_full_trajectory_surface_strings","held_syntax_no_duplicate","new_names_no_duplicate"):
        rec["matched_both_correct"]=sum(a and b for a,b in zip(valid[::2],valid[1::2]))
        rec["matched_count"]=len(valid)//2
    return rec

local=s.Baseline(tr,2);bag=s.Baseline(tr,0)
def rnn_predict(x):return [tuple(map(int,z)) for z in m.predict(lx.pack(x)[0]).T]
res={}
for name,x in tests.items():
    assert len({r[0] for r in x})==len(x) and not any(r[0] in ts for r in x)
    rp=rnn_predict(x)
    ref=[s.grammar_reference(z[0]) for z in x]
    per={}
    for k,p in [("RNN",rp),("local",s.evaluate_fn(local.one,x)),("bag",s.evaluate_fn(bag.one,x)),("declared_exact_grammar",ref),
                ("grammar_exact_match_no_repair",[a if a==b else None for a,b in zip(rp,ref)])]:
        per[k]=evals(name,x,p)
    res[name]=per
out={"first_run_split_audit":audit_original,"each_test_stratum_is_training_disjoint":True,"checkpoint_sha256":sha,"results":res,
     "note":"All audited sets are unique, raw byte-surface disjoint from training. Primary triple stratum not paired. Raw RNN span binding floated; fitted protected numeric bank read at AST time; exact-match guard only rejects, never repairs."}
(HERE/"language_binding_audit_report.json").write_text(json.dumps(out,indent=2))
print(json.dumps(audit_original,indent=2))
for k,a in res.items():print(k,[(n,r["raw_exact"],r["eligible"],r["false_admit"],r.get("matched_both_correct")) for n,r in a.items()])

