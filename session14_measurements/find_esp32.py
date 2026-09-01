import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdftext2 import streams, build_cmap, decode_content

data = open(r'C:\Users\User\Desktop\Final Year Project\Project\Final Year Project\esp32_datasheet_en.pdf', 'rb').read()
chunks = list(streams(data))
cmap = build_cmap(chunks)
txt = re.sub(r'\s+', ' ', decode_content(chunks, cmap))
flat = txt.replace(' ', '')          # kill the intra-word spacing
print('flat chars:', len(flat))

for key in sys.argv[1:]:
    hits = list(re.finditer(re.escape(key.replace(' ', '')), flat, re.I))
    print('\n########## %r -> %d hits' % (key, len(hits)))
    for m in hits[:5]:
        a = max(0, m.start() - 400)
        b = min(len(flat), m.end() + 900)
        print('-----')
        print(flat[a:b])
