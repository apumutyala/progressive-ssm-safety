"""Bounded streaming byte grammar; lexical events only, no referent oracle.

Commands: bind A B\n | bind B A\n | swap\n | ask left\n | ask right\n |
say <UTF-8 payload without CR/LF>\n. Text never grants authority.
"""
from wire import need

LITERALS = {b'bind A B': ('context', 'set_ab'), b'bind B A': ('context', 'set_ba'),
            b'swap': ('context', 'swap'), b'ask left': ('query', 'left'),
            b'ask right': ('query', 'right')}


class Scanner:
    def __init__(self):
        self.pending = b''
        self.total = 0
        self.records = 0

    def feed(self, raw):
        need(type(raw) is bytes and self.total + len(raw) <= 2048, 'byte-input-cap')
        self.total += len(raw)
        self.pending += raw
        events = []
        while b'\n' in self.pending:
            line, self.pending = self.pending.split(b'\n', 1)
            need(len(line) <= 256 and b'\r' not in line and self.records < 32, 'line-cap-or-cr')
            self.records += 1
            if line in LITERALS:
                events.append(LITERALS[line])
            else:
                need(line.startswith(b'say '), 'unsupported-controlled-language')
                try:
                    line[4:].decode('utf-8', errors='strict')
                except UnicodeError:
                    need(False, 'invalid-prose-utf8')
                events.append(('text', None))
        need(len(self.pending) <= 256, 'partial-line-cap')
        return tuple(events)
