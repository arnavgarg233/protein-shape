"""Beta-sheet topology strings, enumeration, connections and descriptors.

Entry k of `order` is the sequence index (1 = N-terminal) of the strand at sheet position k+1, and
`signs[k]` is its orientation: Koga code 1324+-+-, Minami string 1|3|2|4:+-+-. enumerate_koga and
minami_rule_counts follow generate_beta_topo.py and TOPOLcode2frustrations.pl of the Koga laboratory
(github.com/kogalab21/novel_ab-fold_design).
"""
import itertools as it


def parse(s):
    """Return (order, signs) from a Minami string, a Koga code or a Murata digit string."""
    if ':' in s:
        o, sg = s.split(':')
        return tuple(int(x) for x in o.split('|')), tuple(sg)
    digits = ''.join(ch for ch in s if ch.isdigit())
    signs = ''.join(ch for ch in s if ch in '+-')
    if not signs:
        signs = '+' * len(digits)
    # Murata strings such as '123+++' repeat the sign once per strand: keep n signs
    return tuple(int(x) for x in digits), tuple(signs)


def minami(order, signs):
    return '|'.join(str(x) for x in order) + ':' + ''.join(signs)


def koga(order, signs):
    return ''.join(str(x) for x in order) + ''.join(signs)


def enumerate_koga(n):
    """All n!/2 * 2^(n-1) topologies of n strands (generate_beta_topo.py, same order)."""
    half = n // 2 + n % 2
    orders = []
    for p in it.permutations(range(n)):
        if 0 not in p[:half]:
            continue
        if n % 2 == 1 and p[half - 1] == 0 and 1 not in p[:half - 1]:
            continue
        orders.append(tuple(i + 1 for i in p))
    sign_sets = list(it.product(*([('+',)] + [('+', '-')] * (n - 1))))
    return [(o, s) for o in orders for s in sign_sets]


def canonical_order(order):
    """Pick the left-right flip of a parallel sheet that the Koga generator keeps."""
    r = order[::-1]
    return min((order, r), key=lambda t: (t.index(1), t.index(2)))


def reverse_parallel(order):
    """Reverse N- to C-terminal chain direction of a pure parallel topology."""
    n = len(order)
    return canonical_order(tuple(n + 1 - x for x in order))


def _bseq(order, signs):
    """Per strand (sequence order): sheet position (1-based) and sign (+1/-1)."""
    b = [None] * len(order)
    for k, (s, sg) in enumerate(zip(order, signs)):
        b[s - 1] = {'pos': k + 1, 'sign': 1 if sg == '+' else -1}
    return b


def connections(order, signs):
    """One dict per sequence-adjacent strand pair (strand i to i+1)."""
    b = _bseq(order, signs)
    out = []
    for i in range(len(b) - 1):
        p1, p2, s1, s2 = b[i]['pos'], b[i + 1]['pos'], b[i]['sign'], b[i + 1]['sign']
        c = {'pos1': p1, 'pos2': p2, 'sign1': s1, 'sign2': s2, 'i1': i, 'i2': i + 1,
             'jump': abs(p2 - p1) - 1, 'type': 'over' if s1 * s2 > 0 else 'turn',
             'posmin': min(p1, p2), 'posmax': max(p1, p2), 'check': 0, 'geom': 0}
        face = 1 if (p2 - p1) * s1 > 0 else -1          # +1 front of sheet, -1 back
        if (c['type'] == 'turn' and c['jump'] == 1) or (c['type'] == 'over' and c['jump'] <= 100):
            c['check'], c['geom'] = 1, face
        out.append(c)
    return b, out


def _greek_key(b, base):
    in2 = si2 = in3 = si3 = 0
    for i, st in enumerate(b):
        if st['pos'] == base['posmin'] + 1:
            in2, si2 = i, st['sign']
        if st['pos'] == base['posmax'] - 1:
            in3, si3 = i, st['sign']
    in1, si1, in4, si4 = base['i1'], base['sign1'], base['i2'], base['sign2']
    if base['pos1'] > base['pos2']:
        in1, in4, si1, si4 = in4, in1, si4, si1
    if not (si1 == si3 and si2 == si4 and si1 != si2):
        return 0
    if in1 > in4:
        in1, in4 = in4, in1
        in2, in3 = in3, in2
        si1, si4 = si4, si1
        si2, si3 = si3, si2
    return int((in1 < in4 < in3 < in2) or (in2 < in1 < in4 < in3) or (in3 < in2 < in1 < in4))


def _ov_cross_cross(a, c):
    if a['type'] == 'over' and c['type'] == 'over':
        if (a['posmin'] < c['posmin'] and c['posmax'] < a['posmax']) or \
           (c['posmin'] < a['posmin'] and a['posmax'] < c['posmax']):
            return a['geom'] * c['geom']
    return 0


def _ov_turn_turn(a, c):
    if a['type'] == 'turn' and c['type'] == 'turn' and a['sign1'] == c['sign1']:
        if (a['posmin'] < c['posmin'] and a['posmax'] < c['posmax']) or \
           (c['posmin'] < a['posmin'] and c['posmax'] < a['posmax']):
            return a['geom'] * c['geom']
    return 0


def _ov_cross_turn(a, c):
    if {a['type'], c['type']} == {'turn', 'over'}:
        if a['posmax'] <= c['posmin'] or c['posmax'] <= a['posmin']:
            return 0
        return a['geom'] * c['geom']
    return 0


def _pairing2(a, c):
    if a['sign1'] != c['sign1']:
        return 0
    if a['type'] == 'over' and c['type'] == 'over' and abs(a['pos2'] - c['pos2']) == 1:
        if (a['pos2'] - a['pos1']) * (c['pos2'] - c['pos1']) > 0:
            if (a['pos1'] - c['pos1']) * (a['pos2'] - c['pos2']) > 0:
                return 2
            return -1
        return 1
    return 0


def minami_rule_counts(order, signs):
    """Counts of rule violations as printed by TOPOLcode2frustrations.pl.

    jump_para / jump_anti: connection jump-distance rule (parallel > 3, antiparallel > 1,
    Greek key exempt). ov_pp / ov_aa / ov_pa: connection overlap rule. ending: connection
    ending rule. Greek-key exemptions are returned as `greek_key`.
    """
    b, bxb = connections(order, signs)
    jp_para = jp_anti = greek = 0
    for c in bxb:
        if c['type'] == 'over' and c['jump'] > 3:
            jp_para += 1
        if c['type'] == 'turn' and c['jump'] > 1:
            if c['jump'] == 2 and _greek_key(b, c) == 1:
                greek += 1
            else:
                jp_anti += 1
    pp = aa = pa = end = 0
    for i in range(len(bxb) - 1):
        if not bxb[i]['check']:
            continue
        for j in range(i + 1, len(bxb)):
            if not bxb[j]['check']:
                continue
            pp += _ov_cross_cross(bxb[i], bxb[j]) == 1
            aa += _ov_turn_turn(bxb[i], bxb[j]) == 1
            pa += _ov_cross_turn(bxb[i], bxb[j]) == 1
            end += _pairing2(bxb[i], bxb[j]) == 1
    return {'jump_para': jp_para, 'jump_anti': jp_anti, 'ov_pp': pp, 'ov_aa': aa,
            'ov_pa': pa, 'ending': end, 'greek_key': greek}


def descriptors(order, signs):
    """Geometric descriptors computed from the string alone.

    Sheet neighbours are consecutive sheet positions; connections join strand i to i+1.
    rco_sheet is the mean sequence separation of sheet-neighbour strands divided by the
    number of strands (Murata's betaRCO). A crossover connection joins two strands of the
    same orientation (a parallel beta-X-beta unit).
    """
    n = len(order)
    b, bxb = connections(order, signs)
    sheet_pairs = [(order[k], order[k + 1], signs[k], signs[k + 1]) for k in range(n - 1)]
    n_par_pairs = sum(1 for p in sheet_pairs if p[2] == p[3])
    jumps = [c['jump'] for c in bxb]
    cross = [c for c in bxb if c['type'] == 'over']
    return {
        'n_strands': n,
        'n_parallel_pairs': n_par_pairs,
        'n_antiparallel_pairs': (n - 1) - n_par_pairs,
        'n_crossover': len(cross),
        'n_crossover_front': sum(1 for c in cross if c['geom'] == 1),
        'n_crossover_back': sum(1 for c in cross if c['geom'] == -1),
        'n_jumps': sum(1 for j in jumps if j > 0),
        'sum_jump': sum(jumps),
        'max_jump': max(jumps),
        'rco_sheet': sum(abs(p[1] - p[0]) for p in sheet_pairs) / ((n - 1) * n),
        'nterm_pos': b[0]['pos'],
        'cterm_pos': b[-1]['pos'],
        'nterm_edge': int(b[0]['pos'] in (1, n)),
        'cterm_edge': int(b[-1]['pos'] in (1, n)),
    }


def clash_free_parallel(order):
    """Reconstruction of Murata's clash-free test for a pure parallel sheet.

    With every beta-alpha-beta unit right-handed, the helix of strand i -> i+1 lies on the
    front face when the chain moves right across the sheet and on the back face when it
    moves left. Two connections on the same face whose sheet spans are strictly nested
    clash. This reproduces the 3/10/34/120 topologies of the Murata repo exactly.
    """
    pos = {s: k for k, s in enumerate(order)}
    spans = []
    for i in range(1, len(order)):
        a, c = pos[i], pos[i + 1]
        spans.append((min(a, c), max(a, c), 1 if c > a else -1))
    for (a, b, fa), (c, d, fc) in it.combinations(spans, 2):
        if fa == fc and ((a < c and d < b) or (c < a and b < d)):
            return False
    return True
