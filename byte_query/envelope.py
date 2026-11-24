"""Combined typed envelope. Binder's exact copied wire.py is UNCHANGED.

Larger integer token allowance supports the reference signed32-bit epoch field;
all fields are separately validated. Standard-library JSON/hash remain TCB.
"""
import json
from wire import encode, digest, need, Reject, MAX_BYTES


def decode(raw):
    need(type(raw) is bytes and len(raw) <= MAX_BYTES, 'envelope-size')

    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate-key')
            result[key] = value
        return result

    def integer(text):
        need(len(text) <= 12, 'envelope-integer-size')
        return int(text)

    def no_float(_):
        raise Reject('noninteger-envelope')

    try:
        obj = json.loads(raw.decode('ascii'), object_pairs_hook=pairs, parse_int=integer,
                         parse_float=no_float, parse_constant=no_float)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise Reject('envelope-parse') from exc
    need(encode(obj) == raw, 'noncanonical-envelope')
    return obj
