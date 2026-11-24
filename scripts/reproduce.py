"""Reproduce bounded proofs and fresh synthetic exercises; no downloads."""
import argparse, os, pathlib, shutil, subprocess, sys, tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
LEAN_FILES=[
    "FittedObserverConformance.lean",
    "StructuralSSMUnifiedRouted.lean",
    "ExactFramedFittedStrict.lean",
    "IngestionSparseCommit.lean",
    "FaultTolerantDrift.lean",
    "DualWitnessUpdate.lean",
]
CPU_FILES=[
    "fit_observer.py",
    "witness_search.py",
    "independent_geometry_oracle.py",
    "nested_rival_bench.py",
    "language_binding_cpu.py",
    "language_binding_audit.py",
]
def call(cmd,cwd,env,timeout=100,grep=False,quiet=False):
    print("+", " ".join([pathlib.Path(cmd[0]).name,*map(str,cmd[1:])]),flush=True)
    p=subprocess.run(cmd,cwd=cwd,env=env,timeout=timeout,check=False,
                     text=True,encoding="utf-8",errors="replace",
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if (p.returncode or (grep and ("sorryAx" in p.stdout or "uses 'sorry'" in p.stdout))):
        print(p.stdout,end="")
        raise SystemExit(p.returncode or 1)
    if not quiet: print(p.stdout,end="",flush=True)
    return p.stdout
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--lean",default="lean",help="Existing Lean 4.8.0 executable")
    a.add_argument("--seed",type=int,default=910231,help="Fresh synthetic evaluation seed")
    a.add_argument("--only",choices=("lean","cpu","all"),default="all")
    a.add_argument("--summary-only",action="store_true")
    q=a.parse_args()
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","THEORY_SYNTH_SEED":str(q.seed)}
    with tempfile.TemporaryDirectory(prefix="alignment-theory-replay-") as tmp:
        scratch=pathlib.Path(tmp)
        if q.only!="cpu":
            lean=shutil.which(q.lean)
            if not lean: raise SystemExit("Install or provide an existing Lean 4.8.0 toolchain.")
            version=call([lean,"--version"],ROOT,env)
            if "version 4.8.0" not in version: raise SystemExit("Expected Lean version 4.8.0.")
            for f in LEAN_FILES:
                call([lean,str(pathlib.Path("lean")/f)],ROOT,env,grep=True,quiet=q.summary_only)
            mainfile=scratch/"Demo.lean"
            source=(ROOT/"lean"/"ExactFramedFittedStrict.lean").read_text(encoding="utf-8")
            tail=(ROOT/"demo"/"FramedDemoTail.lean").read_text(encoding="utf-8")
            mainfile.write_text(source+"\n"+tail,encoding="utf-8")
            shown=call([lean,"--run",mainfile.name],scratch,env,grep=True,quiet=q.summary_only)
            if "fresh true: fact" not in shown or "same query, fresh false: fact" not in shown:
                raise SystemExit("Positive demo output missing.")
            if "forged request in prose channel: DEFER" not in shown:
                raise SystemExit("Negative demo output missing.")
        if q.only!="lean":
            try: import numpy
            except ImportError: raise SystemExit("CPU checks require NumPy; see requirements.txt.")
            print("NumPy:",numpy.__version__,flush=True)
            for f in [*CPU_FILES,"minimal_observer_coefficients.json"]:
                shutil.copy2(ROOT/"cpu"/f,scratch/f)
            for f in CPU_FILES:
                seed_args=["--seed-start",str(q.seed)] if f=="nested_rival_bench.py" else []
                call([sys.executable,f,*seed_args],scratch,env,timeout=100,quiet=q.summary_only)
    print("Replay finished; generated files were confined to the temporary workspace.")
if __name__=="__main__": main()
