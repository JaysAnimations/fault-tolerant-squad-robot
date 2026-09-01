"""
RPLIDAR A1 datasheet: combine both text encodings.

The file mixes WinAnsiEncoding simple fonts (literal text in parentheses)
with Identity-H CID fonts (glyph ids in hex strings, decoded via ToUnicode).
Take both, but only from real page-content streams -- the embedded font
programs also contain parenthesised byte runs and are pure noise.
"""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdftext2 import streams, build_cmap

PDF = (r'C:\Users\User\Desktop\Final Year Project\Project\Final Year Project'
       r'\LD108_SLAMTEC_rplidar_datasheet_A1M8_v1.0_en-1626872.pdf')

data = open(PDF, 'rb').read()
chunks = list(streams(data))
cmap = build_cmap(chunks)
print('cmap entries:', len(cmap))

TOKEN = re.compile(r'\(((?:[^()\\]|\\.)*)\)|<([0-9A-Fa-f\s]{4,})>')

out = []
for c in chunks:
    if b'BT' not in c or (b'Tj' not in c and b'TJ' not in c):
        continue
    s = c.decode('latin-1')
    for m in TOKEN.finditer(s):
        lit, hx = m.group(1), m.group(2)
        if lit is not None:
            out.append(re.sub(r'\\([()\\])', r'\1', lit))
        else:
            hx = re.sub(r'\s', '', hx)
            if len(hx) % 4:
                continue
            word = ''.join(cmap.get(int(hx[i:i + 4], 16), '')
                           for i in range(0, len(hx), 4))
            out.append(word)

txt = ' '.join(out)
txt = ''.join(ch if 32 <= ord(ch) < 127 else ' ' for ch in txt)
txt = re.sub(r'\s+', ' ', txt)
flat = txt.replace(' ', '')
print('chars:', len(txt), '| flat:', len(flat))

for key in sys.argv[1:]:
    k = key.replace(' ', '')
    hits = list(re.finditer(re.escape(k), flat, re.I))
    print('\n########## %r -> %d hits' % (key, len(hits)))
    for m in hits[:4]:
        a = max(0, m.start() - 450)
        b = min(len(flat), m.end() + 800)
        print('-----')
        print(flat[a:b])
