import json, pathlib, itertools, time, random, numpy as np
from witness_search import dataset, polygon, B, EVAL_SEED
HERE=pathlib.Path(__file__).parent
# Independent exact oracle: all intersections of pairs of supporting lines,
# not polygon clipping. Compact nonempty polygons always contain an extreme point.
def exact_vertex_feasible(a):
    m=[tuple(map(int,row)) for row in a]+[(1,0,B),(-1,0,B),(0,1,B),(0,-1,B)]
    for (a,b,c),(d,e,f) in itertools.combinations(m,2):
        det=a*e-b*d
        if det==0:continue
        x=c*e-b*f; y=a*f-c*d
        if det<0:det,x,y=-det,-x,-y
        if all(g*x+h*y<=j*det for g,h,j in m):return True
    return False

def min_rational_support(row):
    for k in (1,2,3):
        for inds in itertools.combinations(range(len(row)),k):
            if not polygon(row[list(inds)]):return k
    return 0
st=time.perf_counter()
hold,_=dataset(("scaled_pair","oblique_triple","boundary_feasible"),EVAL_SEED+101,40)
mismatches=[]
table={}
for h in hold:
    v=not exact_vertex_feasible(h["rows"])
    if v!=h["truth"]:mismatches.append(h["instance"])
    if v:
        k=min_rational_support(h["rows"])
        row=table.setdefault(h["fam"],{})
        key=str((k,h["sparse"]))
        row[key]=row.get(key,0)+1
    assert time.perf_counter()-st<20
print(json.dumps(dict(independent_exact_vertex_mismatches=mismatches,cases=len(hold),
    support_pairs_rational_then_denominator8=table, elapsed=time.perf_counter()-st),indent=2))
