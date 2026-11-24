"""Small bounded artifact transport. Stdlib JSON/hash are explicit TCB."""
import hashlib
import json

MAX_BYTES = 65536


class Reject(ValueError):
    pass


def need(ok, reason):
    if not ok:
        raise Reject(reason)


def encode(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                     allow_nan=False).encode('ascii')
    need(len(raw) <= MAX_BYTES, 'transport-size')
    return raw


def digest(raw):
    need(type(raw) is bytes and len(raw) <= MAX_BYTES, 'transport-bytes')
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    need(type(raw) is bytes and len(raw) <= MAX_BYTES, 'transport-size')

    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, 'duplicate-key')
            out[k] = v
        return out

    def integer(s):
        need(len(s) <= 5, 'integer-size')
        return int(s)

    def no_float(_):
        raise Reject('noninteger')

    try:
        obj = json.loads(raw.decode('ascii'), object_pairs_hook=pairs,
                         parse_int=integer, parse_float=no_float, parse_constant=no_float)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise Reject('transport-parse') from exc
    need(encode(obj) == raw, 'noncanonical-bytes')
    return obj
