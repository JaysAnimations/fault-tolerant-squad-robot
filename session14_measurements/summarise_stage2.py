"""Summarise the Stage 2 run from its captured console output."""
import re, collections

SRC = (r'C:\Users\User\AppData\Local\Temp\claude'
       r'\C--Users-User-Desktop-Final-Year-Project-Project-Final-Year-Project'
       r'\9eaef393-eb6f-4494-bac6-ec1703df4456\tasks\bqe264hv3.output')

LINE = re.compile(
    r'^\s{2}(?P<arm>.+?)\s+seed\s+(?P<seed>\d+)\s+'
    r'(?P<energy>[\d.]+) J\s+(?P<perpt>[\d.]+) J/pt\s+'
    r'truly (?P<truly>\d+)\s+cov\s+(?P<cov>[\d.]+)\s+'
    r'realloc (?P<realloc>\d+)')

rows = []
for line in open(SRC):
    m = LINE.match(line.rstrip('\n'))
    if m:
        d = m.groupdict()
        d['arm'] = d['arm'].strip()
        rows.append(d)

print('parsed %d runs' % len(rows))
arms = []
for r in rows:
    if r['arm'] not in arms:
        arms.append(r['arm'])

def mean(arm, key):
    v = [float(r[key]) for r in rows if r['arm'] == arm]
    return sum(v) / len(v)

print()
print('MEANS OVER THE FIVE comms_loss SEEDS (new coefficients throughout)')
print('%-36s %10s %9s %8s %8s %8s' %
      ('arm', 'energy J', 'J/point', 'truly', 'cov %', 'realloc'))
print('-' * 84)
for a in arms:
    print('%-36s %10.1f %9.2f %8.2f %8.2f %8.2f' %
          (a, mean(a, 'energy'), mean(a, 'perpt'), mean(a, 'truly'),
           mean(a, 'cov'), mean(a, 'realloc')))

c2 = mean(arms[0], 'perpt')
c5 = mean(arms[1], 'perpt')
old = mean(arms[2], 'perpt')
new = mean(arms[3], 'perpt')
print()
print('C3 old vs C2 : %+.2f J/point (%+.1f %%)' % (old - c2, 100 * (old - c2) / c2))
print('C3 new vs C2 : %+.2f J/point (%+.1f %%)' % (new - c2, 100 * (new - c2) / c2))
print('C3 new vs C5 : %+.2f J/point' % (new - c5))
print('EFFECT OF THE STAGE 2 FLAG ALONE : %+.2f J/point' % (new - old))

# per-seed identity check
print()
print('PER-SEED, C3 old vs C3 new:')
byseed = collections.defaultdict(dict)
for r in rows:
    byseed[r['seed']][r['arm']] = r
identical = 0
for s in sorted(byseed, key=int):
    a, b = byseed[s][arms[2]], byseed[s][arms[3]]
    same = (a['energy'] == b['energy'] and a['truly'] == b['truly']
            and a['cov'] == b['cov'])
    identical += same
    print('  seed %-4s energy %9s vs %9s  truly %s vs %s  realloc %s vs %s   %s'
          % (s, a['energy'], b['energy'], a['truly'], b['truly'],
             a['realloc'], b['realloc'],
             'IDENTICAL' if same else 'DIFFERS'))
print('  -> %d of %d seeds identical on energy, truly and coverage'
      % (identical, len(byseed)))

print()
print('PER-SEED, C5 vs C3 new:')
for s in sorted(byseed, key=int):
    a, b = byseed[s][arms[1]], byseed[s][arms[3]]
    same = a['energy'] == b['energy'] and a['truly'] == b['truly']
    print('  seed %-4s %9s vs %9s   %s'
          % (s, a['energy'], b['energy'], 'IDENTICAL' if same else 'DIFFERS'))
