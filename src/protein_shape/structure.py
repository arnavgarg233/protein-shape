"""Beta-sheet topology of a protein structure from DSSP, with the three strand definitions (strict,
lenient, lenient_noterm) used by the census.

    python -m protein_shape.structure FILE [--chain A] [--range 10-95] [--model 1]

Strands are DSSP E-runs; two strands are neighbours when they share at least two backbone H-bonds;
open, unbranched sheets are written as topology strings in the Koga-laboratory convention.
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
import numpy as np

from .paths import DATA

_BIN = Path(sys.executable).parent
MKDSSP = os.environ.get('MKDSSP', str(_BIN / 'mkdssp'))
CIFPP_DIR = _BIN.parent / 'share' / 'libcifpp'
# Optional trimmed libcifpp data dir (standard-residue CCD subset): same DSSP output, much faster
# mkdssp start-up. Falls back to the full dir.
CIFPP_MIN_DIR = Path(os.environ.get('LIBCIFPP_MIN_DIR', DATA / 'external' / 'libcifpp_min'))

MIN_STRAND = 3        # "lengths of constituent beta-strands are more than two residues"
MIN_HBONDS = 2        # "two neighbouring strands have at least two main chain hydrogen bonds"
HB_CUTOFF = -0.5      # kcal/mol, Kabsch-Sander
MIN_HELIX = 4         # DSSP 'H' residues needed to call a helix between strands
CUT_JUMP_PARA = 3     # Koga: parallel jump above this violates the jump rule
CUT_JUMP_ANTI = 1     # Koga: antiparallel jump above this violates the jump rule


# ---------------------------------------------------------------------------
# input: structure, chain, residue ranges
# ---------------------------------------------------------------------------

def parse_range(text):
    """'10-95' or 'A:1-50,A:70-120' -> list of (chain or None, (num, icode), (num, icode))."""
    if not text:
        return []
    out = []
    for part in text.split(','):
        part = part.strip()
        chain = None
        if ':' in part:
            chain, part = part.split(':', 1)
        m = re.fullmatch(r'(-?\d+)([A-Za-z]?)-(-?\d+)([A-Za-z]?)', part)
        if not m:
            raise ValueError(f'cannot parse residue range {part!r}')
        out.append((chain, (int(m[1]), m[2]), (int(m[3]), m[4])))
    return out


def _in_ranges(res, chain_name, ranges):
    if not ranges:
        return True
    key = (res.seqid.num, res.seqid.icode.strip())
    return any((c is None or c == chain_name) and lo <= key <= hi for c, lo, hi in ranges)


def select_residues(path, chain=None, ranges=None, model=1):
    """Return (gemmi.Structure, labels) for the selected amino-acid residues.

    The structure has one chain 'A' numbered 1..N (so DSSP rows map to list positions);
    labels[i] is the author residue number (with insertion code) of residue i. Residues
    without N, CA, C or O are dropped. Several chains are concatenated only if the ranges
    name them explicitly.
    """
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    if not 1 <= model <= len(st):
        raise ValueError(f'model {model} not in file ({len(st)} models)')
    src = st[model - 1]
    ranges = ranges or []
    range_chains = {c for c, _, _ in ranges if c}
    if chain is None and not range_chains:
        chain = src[0].name
    out = gemmi.Chain('A')
    labels = []
    for ch in src:
        wanted = ch.name in range_chains if range_chains else ch.name == chain
        if not wanted:
            continue
        for res in ch:
            if res.het_flag == 'H' and res.name != 'MSE':
                continue
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            if not _in_ranges(res, ch.name, ranges):
                continue
            if not all(res.find_atom(a, '*') for a in ('N', 'CA', 'C', 'O')):
                continue
            r = res.clone()
            r.seqid = gemmi.SeqId(len(labels) + 1, ' ')
            out.add_residue(r)
            labels.append(f'{res.seqid.num}{res.seqid.icode.strip()}')
    if not labels:
        raise ValueError('no residues selected')
    new = gemmi.Structure()
    model_out = gemmi.Model('1')
    model_out.add_chain(out)
    new.add_model(model_out)
    return new, labels


# ---------------------------------------------------------------------------
# DSSP
# ---------------------------------------------------------------------------

def run_dssp(structure, workdir=None):
    """Write the structure to a PDB file, run mkdssp, return the classic DSSP text."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        pdb = os.path.join(tmp, 'in.pdb')
        out = os.path.join(tmp, 'out.dssp')
        structure.write_pdb(pdb)
        for data_dir in ([CIFPP_MIN_DIR] if CIFPP_MIN_DIR.exists() else []) + [CIFPP_DIR]:
            env = dict(os.environ, LIBCIFPP_DATA_DIR=str(data_dir))
            p = subprocess.run([MKDSSP, '--output-format', 'dssp', pdb, out], env=env,
                               capture_output=True, text=True)
            if os.path.exists(out):
                break
        if not os.path.exists(out):
            raise RuntimeError(f'mkdssp failed: {p.stderr[-300:]}')
        with open(out) as f:
            return f.read()


_PAIR = re.compile(r'(-?\d+),\s*(-?\d+\.\d)')


def parse_dssp(text):
    """Parse classic DSSP output.

    Returns dict with per-residue lists ss and brk (True if a chain break precedes the
    residue) and hb, the set of (donor, acceptor) residue indices (0-based) for backbone
    N-H...O bonds with Kabsch-Sander energy below HB_CUTOFF.
    """
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('  #  RESIDUE')) + 1
    rows = [(int(l[0:5]), l) for l in lines[start:] if len(l) >= 38 and l[13] != '!']
    row2idx = {row: idx for idx, (row, _) in enumerate(rows)}
    ss, brk, hb = [], [], set()
    prev = None
    for idx, (row, line) in enumerate(rows):
        ss.append(line[16] if line[16] != ' ' else '-')
        brk.append(prev is not None and row != prev + 1)
        prev = row
        for k, (off, en) in enumerate(_PAIR.findall(line[38:83])):
            off, en = int(off), float(en)
            other = row2idx.get(row + off)
            if off == 0 or en > HB_CUTOFF or other is None:
                continue
            # columns alternate: N-H(this)-->O(other), O(this)-->H-N(other)
            hb.add((idx, other) if k % 2 == 0 else (other, idx))
    return dict(ss=ss, brk=brk, hb=hb)


# ---------------------------------------------------------------------------
# strands, bridges, sheets
# ---------------------------------------------------------------------------

def _runs(ss, brk, ch, min_len):
    runs, i, n = [], 0, len(ss)
    while i < n:
        if ss[i] == ch:
            j = i
            while j + 1 < n and ss[j + 1] == ch and not brk[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def bridge_type(hb, i, j):
    """Kabsch-Sander bridge test for residues i, j. Returns (type, bonds) or (None, ())."""
    # Kabsch-Sander Hbond(a, b) is the bond CO(a)...NH(b); hb holds (NH residue, CO residue)
    h = lambda a, b: (b, a) in hb
    for bonds in (((i - 1, j), (j, i + 1)), ((j - 1, i), (i, j + 1))):
        if all(h(*b) for b in bonds):
            return 'P', tuple((b, a) for a, b in bonds)
    for bonds in (((i, j), (j, i)), ((i - 1, j + 1), (j - 1, i + 1))):
        if all(h(*b) for b in bonds):
            return 'A', tuple((b, a) for a, b in bonds)
    return None, ()


def strand_pairings(strands, hb):
    """For each strand pair with a bridge: {'P': n bridges, 'A': n bridges, 'hbonds': n H-bonds}."""
    pairs = {}
    for k1, (a1, b1) in enumerate(strands):
        for k2 in range(k1 + 1, len(strands)):
            a2, b2 = strands[k2]
            n = {'P': 0, 'A': 0}
            bonds = set()
            for i in range(a1, b1 + 1):
                for j in range(a2, b2 + 1):
                    if abs(i - j) < 3:
                        continue
                    t, b = bridge_type(hb, i, j)
                    if t:
                        n[t] += 1
                        bonds.update(b)
            if n['P'] or n['A']:
                pairs[(k1, k2)] = dict(P=n['P'], A=n['A'], hbonds=len(bonds))
    return pairs


def _merge_bulges(runs, hb, max_gap=2):
    """Join neighbouring runs split by a beta-bulge or a missing bridge.

    Two runs separated by 1..max_gap residues are one strand if some third run is bridged to
    both of them with the same bridge type (they continue one ladder).
    """
    changed = True
    while changed:
        changed = False
        pr = strand_pairings(runs, hb)
        for k in range(len(runs) - 1):
            (a1, b1), (a2, b2) = runs[k], runs[k + 1]
            if not 1 <= a2 - b1 - 1 <= max_gap:
                continue
            for m in range(len(runs)):
                if m in (k, k + 1):
                    continue
                p1 = pr.get((min(k, m), max(k, m)))
                p2 = pr.get((min(k + 1, m), max(k + 1, m)))
                if p1 and p2 and (p1['P'] > p1['A']) == (p2['P'] > p2['A']):
                    runs[k:k + 2] = [(a1, b2)]
                    changed = True
                    break
            if changed:
                break
    return runs


def find_strands(ss, brk, weak=True, hb=None, merge=True, use_b=False):
    """Strands as (start, end, is_weak) sorted by position.

    Ladder residues are DSSP 'E' (and, if use_b, isolated bridges 'B'). Strong strands are runs
    of at least MIN_STRAND ladder residues; runs split by a bulge are merged first (needs hb).
    Weak strands (only if `weak`) are runs of 2 ladder residues that overlap no strong strand;
    they stand in for the STRIDE strands that refine_topol_2019.pl adds when they make >= 2
    H-bonds to a sheet. A weak strand never joins two other weak strands.
    """
    mask = ['E' if (x == 'E' or (use_b and x == 'B')) else '-' for x in ss]
    runs = _runs(mask, brk, 'E', 2)
    if merge and hb is not None:
        runs = _merge_bulges(runs, hb)
    return [(a, b, b - a + 1 < MIN_STRAND) for a, b in runs if weak or b - a + 1 >= MIN_STRAND]


def build_sheets(n_strands, pairs, is_weak=None):
    """Connected components of the neighbour graph, with closed / bifurcated flags."""
    is_weak = is_weak or [False] * n_strands
    adj = {k: [] for k in range(n_strands)}
    for (a, b), p in pairs.items():
        if p['hbonds'] >= MIN_HBONDS and not (is_weak[a] and is_weak[b]):
            adj[a].append(b)
            adj[b].append(a)
    seen, sheets = set(), []
    for s in range(n_strands):
        if s in seen:
            continue
        comp, stack = [], [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(y for y in adj[x] if y not in seen)
        comp.sort()
        n_edges = sum(len(adj[x]) for x in comp) // 2
        sheets.append(dict(members=comp, adj=adj,
                           closed=n_edges >= len(comp) and len(comp) >= 3,
                           bifurcated=any(len(adj[x]) > 2 for x in comp)))
    return sheets


def spatial_order(sheet, pairs):
    """Order strands along an open, unbranched sheet; sign (+1/-1) relative to the first.

    Returns (order, signs) with strand indices, or (None, None) if the sheet is closed or
    bifurcated.
    """
    if sheet['closed'] or sheet['bifurcated']:
        return None, None
    adj, comp = sheet['adj'], sheet['members']
    ends = [x for x in comp if len(adj[x]) == 1]
    if len(ends) != 2:
        return None, None
    order, prev = [ends[0]], None
    while len(order) < len(comp):
        nxt = [y for y in adj[order[-1]] if y != prev][0]
        prev = order[-1]
        order.append(nxt)
    signs = [1]
    for a, b in zip(order, order[1:]):
        p = pairs[(min(a, b), max(a, b))]
        signs.append(signs[-1] * (1 if p['P'] > p['A'] else -1))
    return order, signs


def canonical(local_order, signs):
    """tpl2line.pl: reverse so strand 1 is in the left half, then make the leftmost sign '+'.

    local_order: sheet-local strand numbers (1..n, N to C) in spatial order.
    Returns (order, signs, reversed).
    """
    vals = list(zip(signs, local_order))
    n = len(vals)
    left_min = min(num for _, num in vals[:int(np.ceil((n - 1) / 2))])
    if n % 2 == 1 and vals[(n - 1) // 2][1] == 1:
        reverse = left_min > 2
    else:
        reverse = left_min > 1
    if reverse:
        vals.reverse()
    if vals[0][0] < 0:
        vals = [(-s, num) for s, num in vals]
    return [num for _, num in vals], [s for s, _ in vals], reverse


def code_strings(order, signs):
    """('1324+-+-', '1|3|2|4:+-+-')."""
    sg = ''.join('+' if s > 0 else '-' for s in signs)
    return ''.join(map(str, order)) + sg, '|'.join(map(str, order)) + ':' + sg


# ---------------------------------------------------------------------------
# frustration flags (port of TOPOLcode2frustrations.pl)
# ---------------------------------------------------------------------------

def _connections_of_code(pos_sign):
    """pos_sign[i] = (position, sign) of strand i+1. One dict per sequence-consecutive pair."""
    out = []
    for i in range(len(pos_sign) - 1):
        p1, s1 = pos_sign[i]
        p2, s2 = pos_sign[i + 1]
        c = dict(pos1=p1, pos2=p2, sign1=s1, sign2=s2, i1=i, i2=i + 1, jump=abs(p2 - p1) - 1,
                 type='over' if s1 * s2 > 0 else 'turn', check=0, geom=0,
                 posmax=max(p1, p2), posmin=min(p1, p2))
        if (c['type'] == 'turn' and c['jump'] == 1) or (c['type'] == 'over' and c['jump'] <= 100):
            c['check'] = 1
            c['geom'] = 1 if (p2 - p1) * s1 > 0 else -1     # +1 front of the sheet, -1 back
        out.append(c)
    return out


def _greek_key(pos_sign, b):
    in2 = si2 = in3 = si3 = 0
    for i, (p, s) in enumerate(pos_sign):
        if p == b['posmin'] + 1:
            in2, si2 = i, s
        if p == b['posmax'] - 1:
            in3, si3 = i, s
    in1, si1, in4, si4 = b['i1'], b['sign1'], b['i2'], b['sign2']
    if b['pos1'] > b['pos2']:
        in1, in4, si1, si4 = in4, in1, si4, si1
    if not (si1 == si3 and si2 == si4 and si1 != si2):
        return 0
    if in1 > in4:
        in1, in4, in2, in3, si1, si4, si2, si3 = in4, in1, in3, in2, si4, si1, si3, si2
    return int((in1 < in4 < in3 < in2) or (in2 < in1 < in4 < in3) or (in3 < in2 < in1 < in4))


def _overlap_cross_cross(a, b):
    if a['type'] == b['type'] == 'over':
        if (a['posmin'] < b['posmin'] and b['posmax'] < a['posmax']) or \
           (b['posmin'] < a['posmin'] and a['posmax'] < b['posmax']):
            return a['geom'] * b['geom']
    return 0


def _overlap_turn_turn(a, b):
    if a['type'] == b['type'] == 'turn' and a['sign1'] == b['sign1']:
        if (a['posmin'] < b['posmin'] and a['posmax'] < b['posmax']) or \
           (b['posmin'] < a['posmin'] and b['posmax'] < a['posmax']):
            return a['geom'] * b['geom']
    return 0


def _overlap_cross_turn(a, b):
    if {a['type'], b['type']} == {'turn', 'over'}:
        if a['posmax'] <= b['posmin'] or b['posmax'] <= a['posmin']:
            return 0
        return a['geom'] * b['geom']
    return 0


def _ending(a, b):
    """check_pairing2; the <ending> column counts only return value 1."""
    if a['sign1'] != b['sign1']:
        return 0
    if a['type'] == b['type'] == 'over' and abs(a['pos2'] - b['pos2']) == 1:
        if (a['pos2'] - a['pos1']) * (b['pos2'] - b['pos1']) > 0:
            return 2 if (a['pos1'] - b['pos1']) * (a['pos2'] - b['pos2']) > 0 else -1
        return 1
    return 0


def frustration_flags(code):
    """Jump / overlap / ending counts and 0/1 flags for '1324+-+-' or '1|3|2|4:+-+-'."""
    if ':' in code:
        v1, v2 = code.split(':')
        order = [int(x) for x in v1.split('|')]
    else:
        order = [int(x) for x in code[:len(code) // 2]]
        v2 = code[len(code) // 2:]
    signs = [1 if c == '+' else -1 for c in v2]
    if len(order) != len(signs):
        raise ValueError(f'invalid code {code}')
    pos_sign = [None] * len(order)
    for i, (strand, sg) in enumerate(zip(order, signs)):
        pos_sign[strand - 1] = (i + 1, sg)
    cons = _connections_of_code(pos_sign)
    jp_para = jp_anti = greek = 0
    for c in cons:
        if c['type'] == 'over' and c['jump'] > CUT_JUMP_PARA:
            jp_para += 1
        if c['type'] == 'turn' and c['jump'] > CUT_JUMP_ANTI:
            if c['jump'] == 2 and _greek_key(pos_sign, c):
                greek += 1
            else:
                jp_anti += 1
    pp = aa = pa = ending = 0
    for a, b in itertools.combinations([c for c in cons if c['check']], 2):
        pp += _overlap_cross_cross(a, b) == 1
        aa += _overlap_turn_turn(a, b) == 1
        pa += _overlap_cross_turn(a, b) == 1
        ending += _ending(a, b) == 1
    return dict(jump_para=jp_para, jump_anti=jp_anti, greek_key=greek, overlap_pp=int(pp),
                overlap_aa=int(aa), overlap_pa=int(pa), ending=int(ending),
                jp=int(jp_para + jp_anti > 0), ov=int(pp + aa + pa > 0), en=int(ending > 0))


# ---------------------------------------------------------------------------
# geometry: face and handedness of a connection
# ---------------------------------------------------------------------------

def _unit(v):
    return v / np.linalg.norm(v)


def _strand_dir(ca):
    """Direction N to C of a strand from its first two and last two C-alpha atoms."""
    if len(ca) < 4:
        return _unit(ca[-1] - ca[0])
    return _unit(ca[-2:].mean(0) - ca[:2].mean(0))


def connection_geometry(ca, strands, sheet_members, frame_order, k1, k2, conn_residues):
    """Side of the sheet that the connection k1 -> k2 lies on, and its handedness.

    Sheet plane: least-squares plane through the C-alpha atoms of all strand residues of the
    sheet. The connection side is the sign of the median height of its C-alpha atoms.
    Face: 'front' when that side is +z of the frame x = leftmost to rightmost strand,
    y = direction of the leftmost strand, z = x cross y (frame_order = canonical spatial
    order, so this matches the string convention). Handedness: the path strand k1, connection,
    strand k2 is a right-handed turn about the axis d (k1 to k2) when (u x n) . d > 0, with u
    the direction of strand k1 and n the unit normal pointing to the connection side.
    """
    def strand_ca(k):
        return ca[strands[k][0]:strands[k][1] + 1]

    pts = np.vstack([strand_ca(k) for k in sheet_members])
    centre = pts.mean(0)
    normal = np.linalg.svd(pts - centre)[2][2]
    height = float(np.median((ca[list(conn_residues)] - centre) @ normal))
    n_side = normal if height >= 0 else -normal
    u = _strand_dir(strand_ca(k1))
    d = strand_ca(k2).mean(0) - strand_ca(k1).mean(0)
    d = d - (d @ normal) * normal
    x = _unit(strand_ca(frame_order[-1]).mean(0) - strand_ca(frame_order[0]).mean(0))
    y = _strand_dir(strand_ca(frame_order[0]))
    y = _unit(y - (y @ x) * x)
    z = np.cross(x, y)
    return dict(face='front' if n_side @ z > 0 else 'back',
                handedness='right' if np.cross(u, n_side) @ d > 0 else 'left',
                height=round(height, 2))


# ---------------------------------------------------------------------------
# putting it together
# ---------------------------------------------------------------------------

STRAND_MODES = ('strict', 'lenient', 'lenient_noterm')


def _terminal_weak(found, hb):
    """Indices of weak strands that are the end of their sheet (at most one neighbour)."""
    strands = [(a, b) for a, b, _ in found]
    pairs = strand_pairings(strands, hb)
    deg = [0] * len(found)
    for (a, b), p in pairs.items():
        if p['hbonds'] >= MIN_HBONDS and not (found[a][2] and found[b][2]):
            deg[a] += 1
            deg[b] += 1
    return {k for k, (_, _, w) in enumerate(found) if w and deg[k] <= 1}


def strands_for_mode(dssp, strand_mode='lenient', opts=None):
    """Strands (start, end, is_weak) for a strand mode.

    strict          strong DSSP E-runs only (3 or more residues)
    lenient         strong runs plus weak 2-residue strands (the default)
    lenient_noterm  lenient, then weak strands that end a sheet are removed
    """
    if strand_mode not in STRAND_MODES:
        raise ValueError(f'strand_mode must be one of {STRAND_MODES}')
    found = find_strands(dssp['ss'], dssp['brk'], strand_mode != 'strict', dssp['hb'], **(opts or {}))
    if strand_mode == 'lenient_noterm':
        drop = _terminal_weak(found, dssp['hb'])
        found = [f for k, f in enumerate(found) if k not in drop]
    return found


def topology_from_dssp(dssp, labels, ca=None, weak=True, opts=None, strand_mode=None, _found=None):
    """Result dict from parsed DSSP, author-number labels and (optional) C-alpha coordinates.

    strand_mode ('strict', 'lenient', 'lenient_noterm') overrides `weak`; with None, weak=True is
    'lenient' and weak=False is 'strict'.
    """
    ss = dssp['ss']
    if _found is not None:
        found = _found
    elif strand_mode is not None:
        found = strands_for_mode(dssp, strand_mode, opts)
    else:
        found = find_strands(ss, dssp['brk'], weak, dssp['hb'], **(opts or {}))
    strands = [(a, b) for a, b, _ in found]
    helices = _runs(ss, dssp['brk'], 'H', MIN_HELIX)
    pairs = strand_pairings(strands, dssp['hb'])
    sheets = [s for s in build_sheets(len(strands), pairs, [w for _, _, w in found])
              if len(s['members']) >= 3]
    # global numbering of strands in sheets of >= 3 strands (Koga's 'independent' test)
    glob = {k: i + 1 for i, k in enumerate(sorted(k for s in sheets for k in s['members']))}
    out = []
    for si, sh in enumerate(sorted(sheets, key=lambda s: s['members'][0]), 1):
        mem = sh['members']                                   # strand indices, N to C
        local = {k: r + 1 for r, k in enumerate(mem)}
        g = [glob[k] for k in mem]
        rec = dict(sheet=si, n_strands=len(mem),
                   strands=[dict(strand=local[k], start=labels[strands[k][0]],
                                 end=labels[strands[k][1]]) for k in mem],
                   closed=sh['closed'], barrel=sh['closed'], bifurcated=sh['bifurcated'],
                   independent=all(b - a == 1 for a, b in zip(g, g[1:])),
                   topology=None, code=None, frustration=None)
        order, signs = spatial_order(sh, pairs)
        frame_order, sign_of = None, None
        if order is not None:
            c_order, c_signs, rev = canonical([local[k] for k in order], signs)
            rec['code'], rec['topology'] = code_strings(c_order, c_signs)
            rec['frustration'] = frustration_flags(rec['topology'])
            frame_order = order[::-1] if rev else order
            sign_of = dict(zip(frame_order, c_signs))
        rec['connections'] = [_connection(strands, helices, mem, local, k1, k2, frame_order,
                                          sign_of, ca) for k1, k2 in zip(mem, mem[1:])]
        out.append(rec)
    return dict(n_residues=len(ss), n_strands=len(strands), n_sheets=len(out),
                strands_per_sheet=[s['n_strands'] for s in out],
                n_open_sheets=sum(1 for s in out if s['topology']),
                sheets=out,
                helices=[dict(start=labels[a], end=labels[b]) for a, b in helices])


def _connection(strands, helices, mem, local, k1, k2, frame_order, sign_of, ca):
    lo, hi = strands[k1][1] + 1, strands[k2][0] - 1
    c = dict(from_strand=local[k1], to_strand=local[k2], n_residues=hi - lo + 1,
             helices_between=sum(1 for a, b in helices if a >= lo and b <= hi))
    if sign_of is None:
        return c
    pos = {k: p + 1 for p, k in enumerate(frame_order)}
    c['type'] = 'parallel' if sign_of[k1] == sign_of[k2] else 'antiparallel'
    c['jump'] = abs(pos[k2] - pos[k1]) - 1
    c['face_if_right_handed'] = 'front' if (pos[k2] - pos[k1]) * sign_of[k1] > 0 else 'back'
    if ca is not None and hi >= lo:
        g = connection_geometry(ca, strands, mem, frame_order, k1, k2, range(lo, hi + 1))
        c['face'] = g['face']
        c['connection_height'] = g['height']
        if c['type'] == 'parallel':
            c['handedness'] = g['handedness']
    return c


def topology_set(dssp, labels, ca=None, sizes=(4, 5), opts=None, extra_found=None):
    """Distinct topology strings of open sheets with `sizes` strands over the strand modes.

    Returns dict(per_mode={mode: sorted codes}, topology_set=sorted union, ambiguous=len(set)>1,
    modes_disagree=per-mode sets differ). `extra_found` maps an additional mode name (for
    example 'stride') to a ready strand list, added to the union.
    """
    per_mode = {}
    modes = list(STRAND_MODES) + list(extra_found or {})
    for m in modes:
        found = (extra_found or {}).get(m)
        res = topology_from_dssp(dssp, labels, ca, opts=opts, strand_mode=None if found else m, _found=found)
        per_mode[m] = sorted({s['code'] for s in res['sheets'] if s['code'] and s['n_strands'] in sizes})
    union = sorted(set().union(*per_mode.values()))
    return dict(per_mode=per_mode, topology_set=union, ambiguous=len(union) > 1,
                modes_disagree=len({tuple(v) for v in per_mode.values()}) > 1)


def topology(path, chain=None, residue_range=None, model=1, strand_mode='lenient'):
    """Main entry point. residue_range uses author numbering: '10-95' or 'A:1-50,A:70-120'.

    The result has the calls for `strand_mode` plus `topology_set` over all strand modes.
    """
    st, labels = select_residues(path, chain, parse_range(residue_range), model)
    dssp = parse_dssp(run_dssp(st))
    if len(dssp['ss']) != len(labels):
        raise RuntimeError(f"DSSP returned {len(dssp['ss'])} residues, expected {len(labels)}")
    ca = np.array([r.find_atom('CA', '*').pos.tolist() for r in st[0][0]])
    res = topology_from_dssp(dssp, labels, ca, strand_mode=strand_mode)
    res.update(strand_mode=strand_mode, topology_set=topology_set(dssp, labels, ca),
               source=str(path), chain=chain, range=residue_range, model=model)
    return res


# ---------------------------------------------------------------------------
# census call record for one domain
# ---------------------------------------------------------------------------

CENSUS_SIZES = (3, 4, 5, 6, 7)  # open-sheet sizes kept in per_mode / topology_set


def chopping_to_range(chop):
    """TED-style chopping 'a-b_c-d' to the range syntax 'a-b,c-d'."""
    return ','.join(seg for seg in chop.split('_'))


def call_domain(row, pdb_text):
    """Census call record for one domain: cut the chain to row['chopping'], run DSSP and return the
    open sheets, the per-mode topology codes and their union, with ok=False and the error on failure."""
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=True) as tmp:
            tmp.write(pdb_text)
            tmp.flush()
            st, labels = select_residues(tmp.name, None, parse_range(chopping_to_range(row['chopping'])), 1)
            dssp = parse_dssp(run_dssp(st))
            ca = np.array([r.find_atom('CA', '*').pos.tolist() for r in st[0][0]])
            res = topology_from_dssp(dssp, labels, ca, strand_mode='lenient')
            ts = topology_set(dssp, labels, ca, sizes=CENSUS_SIZES)
        sheets = [dict(n=s['n_strands'], code=s['code'], closed=s['closed'], bifurcated=s['bifurcated'])
                  for s in res['sheets']]
        return dict(row, ok=True, sheets=sheets, topology_set=ts['topology_set'],
                    per_mode=ts['per_mode'], ambiguous=ts['ambiguous'], modes_disagree=ts['modes_disagree'])
    except Exception as e:
        return dict(row, ok=False, error=repr(e)[:200])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('file')
    ap.add_argument('--chain')
    ap.add_argument('--range', dest='rng', help='author numbering, e.g. 10-95 or A:1-50,A:70-120')
    ap.add_argument('--model', type=int, default=1)
    ap.add_argument('--strand-mode', default='lenient', choices=STRAND_MODES)
    a = ap.parse_args()
    print(json.dumps(topology(a.file, a.chain, a.rng, a.model, a.strand_mode), indent=1))


if __name__ == '__main__':
    main()
