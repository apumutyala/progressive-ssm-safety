"""Watch one Omen research process, using the already installed Torch overlay."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from telemetry import host, process_memory

MIB = 2**20
GIB = 2**30
WORKER_SECONDS = 110
SUPERVISOR_SECONDS = 120


def worker(args):
    sys.path.insert(1, str(args.site))
    import torch
    if (Path(torch.__file__).resolve() != args.site / "torch" / "__init__.py"
            or torch.__version__ != "2.14.1+cu130" or torch.version.cuda != "13.0"
            or sys.version_info[:3] != (3, 12, 10)):
        raise RuntimeError("Unexpected interpreter or Torch overlay for the Omen pilot")
    if os.environ["CUDA_VISIBLE_DEVICES"]:
        if (not torch.cuda.is_available()
                or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
                or torch.cuda.mem_get_info()[0] < 768*MIB):
            raise RuntimeError("Omen GPU identity or available VRAM failed admission")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT), "test*.py")
        if not suite.countTestCases():
            raise RuntimeError("No tests discovered")
        return int(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    if args.mode == "verify-resume":
        sys.argv = ["verify_resume.py", *args.tail]
        runpy.run_path(str(ROOT / "verify_resume.py"), run_name="__main__")
        return 0
    sys.argv = ["pretrain.py", *args.tail]
    runpy.run_path(str(ROOT / "pretrain.py"), run_name="__main__")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True, help="Installed site-packages directory")
    parser.add_argument("--mode", choices=("corpus", "test", "verify-resume"), required=True)
    parser.add_argument("--receipt", type=Path, required=True, help="Fresh path outside checkout")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("tail", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.tail[:1] == ["--"]:
        args.tail = args.tail[1:]
    args.site = args.site.resolve()
    if not (args.site / "torch" / "__init__.py").is_file():
        parser.error("Torch overlay missing")
    if args.child:
        return worker(args)
    receipt = args.receipt.resolve()
    if receipt.is_relative_to(ROOT) or receipt.exists() or not receipt.parent.is_dir():
        parser.error("Provide a fresh receipt path in an existing directory outside the checkout")
    output = None
    if args.mode == "corpus":
        train = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        train.add_argument("--output", type=Path, required=True)
        train.add_argument("--device", choices=("cpu", "cuda"), required=True)
        job, _ = train.parse_known_args(args.tail)
        output = job.output.resolve()
        if output.exists() or output.parent != receipt.parent or output.is_relative_to(ROOT):
            parser.error("Training requires a fresh run directory beside the external receipt")
    baseline = host()
    if baseline["available_physical_bytes"] < 3 * GIB or baseline["global_commit_available_bytes"] < 5.1 * GIB:
        raise RuntimeError(f"Insufficient admission headroom: {baseline}")
    if shutil.disk_usage(receipt.parent).free < 256 * MIB:
        raise RuntimeError("Insufficient disk headroom")
    env = dict(os.environ, CUDA_CACHE_DISABLE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    campaign = (ROOT.parent / "certified-runs").resolve()
    if receipt.parent != campaign:
        parser.error("Use the shared certified-runs directory for the resource ledger")
    retained = sum(p.stat().st_size for p in campaign.rglob("*") if p.is_file())
    remaining = min(4*MIB, 32*MIB - retained) - 512*1024
    if remaining < 256*1024:
        parser.error("Insufficient space within the shared campaign artifact cap")
    env["CORPUS_MAX_ARTIFACT_BYTES"] = str(remaining)
    env["PAIRED_MAX_ARTIFACT_BYTES"] = str(remaining)
    if args.mode in ("test","verify-resume") or job.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.monotonic()
    child = subprocess.Popen([sys.executable, "-I", "-S", "-B", __file__,
                              "--child", "--mode", args.mode, "--site", str(args.site),
                              "--receipt", str(receipt), "--", *args.tail],
                             env=env, creationflags=subprocess.CREATE_NO_WINDOW,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = bytearray()
    overflow = threading.Event()

    def capture():
        with child.stdout as stream:
            while chunk := stream.read(4096):
                remaining = 128 * 1024 - len(log)
                log.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()

    reader = threading.Thread(target=capture, daemon=True)
    reader.start()
    peak = 0
    min_physical = baseline["available_physical_bytes"]
    min_commit = baseline["global_commit_available_bytes"]
    reason = None
    try:
        while child.poll() is None:
            mem = host()
            min_physical = min(min_physical, mem["available_physical_bytes"])
            min_commit = min(min_commit, mem["global_commit_available_bytes"])
            peak = max(peak, process_memory(int(child._handle))["peak_pagefile"])
            if time.monotonic() - start >= WORKER_SECONDS:
                reason = "110 second worker limit"
            elif overflow.is_set():
                reason = "worker log cap"
            elif min(mem["global_commit_available_bytes"], mem["available_physical_bytes"]) < GIB:
                reason = "host memory floor"
            elif peak > 4.5 * GIB:
                reason = "worker private memory cap"
            elif shutil.disk_usage(receipt.parent).free < 256 * MIB:
                reason = "disk floor"
            if reason:
                break
            time.sleep(0.20)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=max(1, SUPERVISOR_SECONDS - (time.monotonic()-start)))
    reader.join(timeout=2)
    if overflow.is_set() or reader.is_alive():
        reason = reason or "incomplete or excessive worker log"
    worker_output = log.decode("utf-8", errors="replace")
    if worker_output:
        print(worker_output, end="")
    result = dict(exit_code=child.returncode, stop_reason=reason, seconds=time.monotonic()-start,
                  peak_worker_private_bytes=peak, mode=args.mode, worker_output=worker_output,
                  admission=baseline, minimum_available_physical_bytes=min_physical,
                  minimum_available_commit_bytes=min_commit)
    artifact_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file()) if output and output.exists() else 0
    result["artifact_bytes"] = artifact_bytes
    if artifact_bytes + len(json.dumps(result).encode()) + 1024 > 4*MIB:
        reason = result["stop_reason"] = "artifacts and receipt exceeded 4 MiB"
    with receipt.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != "worker_output"}))
    return int(bool(child.returncode or reason))


if __name__ == "__main__":
    raise SystemExit(main())
