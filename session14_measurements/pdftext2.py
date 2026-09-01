"""
PDF text extraction via ToUnicode CMaps.

The datasheets embed subset fonts with custom encodings, so the bytes inside
the TJ/Tj operators are glyph ids, not characters. Each font carries a
ToUnicode CMap mapping glyph id -> unicode. We build one merged mapping
(good enough to read numbers and English words out of a datasheet) and
apply it to every hex string in the content streams.
"""
import re, zlib, sys


def streams(data):
    for m in re.finditer(rb'stream\r?\n(.*?)endstream', data, re.S):
        raw = m.group(1)
        try:
            yield zlib.decompress(raw)
        except Exception:
            yield raw


def build_cmap(chunks):
    """Merge every 'beginbfchar'/'beginbfrange' block we can find."""
    cmap = {}
    for c in chunks:
        s = c.decode('latin-1', 'replace')
        if 'beginbfchar' not in s and 'beginbfrange' not in s:
            continue
        for block in re.findall(r'beginbfchar(.*?)endbfchar', s, re.S):
            for src, dst in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', block):
                try:
                    cmap[int(src, 16)] = bytes.fromhex(dst).decode('utf-16-be', 'replace')
                except Exception:
                    pass
        for block in re.findall(r'beginbfrange(.*?)endbfrange', s, re.S):
            for lo, hi, dst in re.findall(
                    r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', block):
                try:
                    lo_i, hi_i, d = int(lo, 16), int(hi, 16), int(dst, 16)
                    for k in range(lo_i, min(hi_i, lo_i + 512) + 1):
                        cmap[k] = chr(d + (k - lo_i))
                except Exception:
                    pass
    return cmap


def decode_content(chunks, cmap):
    out = []
    for c in chunks:
        s = c.decode('latin-1', 'replace')
        if 'TJ' not in s and 'Tj' not in s:
            continue
        for hx in re.findall(r'<([0-9A-Fa-f]{2,})>', s):
            if len(hx) % 4:
                continue
            word = ''
            for i in range(0, len(hx), 4):
                word += cmap.get(int(hx[i:i + 4], 16), '')
            out.append(word)
    return ' '.join(out)


if __name__ == '__main__':
    path = sys.argv[1]
    keys = sys.argv[2:]
    data = open(path, 'rb').read()
    chunks = list(streams(data))
    cmap = build_cmap(chunks)
    txt = decode_content(chunks, cmap)
    txt = re.sub(r'\s+', ' ', txt)
    print('cmap entries:', len(cmap), '| decoded chars:', len(txt))
    print('SAMPLE:', txt[:200])
    for k in keys:
        hits = list(re.finditer(re.escape(k), txt, re.I))
        print('\n##### %r -> %d hits' % (k, len(hits)))
        for m in hits[:6]:
            a = max(0, m.start() - 300)
            b = min(len(txt), m.end() + 400)
            print('---', txt[a:b])
