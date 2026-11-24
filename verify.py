"""Portable stdlib test entry point. No install, model checkpoint or accelerator.

Run python -I -S -B verify.py all. Each suite uses a fresh isolated interpreter
so identically named research modules cannot accidentally share import state.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
PACKAGES = {'quantized_binder': ('test_bridge', 'Qualification', 20),
            'byte_query': ('test_witness', 'Qualification', 19),
            'correspondence': ('test_mapping', 'MappingTests', 7),
            'atomic_reference': ('test_transaction', 'TransactionTests', 13)}


def hashes():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_bytes())
    expected = manifest['files']
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in ROOT.rglob('*') if p.is_file()
              and '.git' not in p.relative_to(ROOT).parts
              and p.relative_to(ROOT).as_posix() != 'MANIFEST.json'}
    if actual != expected:
        raise ValueError('export allowlist/hash mismatch')
    return hashlib.sha256((ROOT / 'MANIFEST.json').read_bytes()).hexdigest()


def child(name):
    sys.path.insert(0, str(ROOT / name))
    module_name, class_name, expected = PACKAGES[name]
    module = __import__(module_name)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, class_name))
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
    ok = result.wasSuccessful() and result.testsRun == expected and 'torch' not in sys.modules
    print(json.dumps(dict(package=name, tests=result.testsRun, failures=len(result.failures),
                          errors=len(result.errors), status='PASS' if ok else 'FAIL',
                          torch_imported='torch' in sys.modules)))
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '--child' and args[1] in PACKAGES:
        return child(args[1])
    if len(args) != 1 or args[0] not in ('all', 'hashes', *PACKAGES):
        raise ValueError('Expected all, hashes, or one package name')
    pin = hashes()
    names = tuple(PACKAGES) if args[0] == 'all' else (() if args[0] == 'hashes' else (args[0],))
    results = []
    for name in names:
        run = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'verify.py'), '--child', name],
                             cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
        if len(run.stdout)+len(run.stderr) > 16384:
            raise ValueError('unexpected test log size')
        sys.stderr.write(run.stderr)
        if run.returncode:
            raise RuntimeError('Failed suite '+name+': '+run.stdout)
        report = json.loads(run.stdout)
        if report['status'] != 'PASS':
            raise RuntimeError('Non-PASS suite '+name)
        results.append(report)
    print(json.dumps(dict(status='PASS', manifest_sha256=pin, packages=results,
                          total_tests=sum(r['tests'] for r in results)), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
