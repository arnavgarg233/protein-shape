"""Replication census: call the sheet topology of AFESM domains whose clusters have no AlphaFold DB member,
using ESMFold models range-read from the Foldcomp esmatlas database.

Stages:
  python scripts/census/build_replication_census.py candidates --f2 AFESM_FILE_2.tsv.gz --f5 AFESM_FILE_5.tsv.gz \\
      --out OUT/candidates.tsv.gz
  python scripts/census/build_replication_census.py resolve --cand OUT/candidates.tsv.gz --out OUT/esm_locator.tsv.gz
  python scripts/census/build_replication_census.py call --cand OUT/candidates.tsv.gz --locator OUT/esm_locator.tsv.gz \\
      --out OUT/calls.jsonl
Shards run in parallel are combined with merge_calls.py. Call records have the layout of build_census.py.
"""
import argparse
import csv
import gc
import gzip
import io
import json
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from protein_shape.structure import call_domain

FOLDCOMP_URL = 'https://steineggerlab.s3.amazonaws.com/foldcomp/esmatlas'  # lookup, index and the data file
MIN_PLDDT, MIN_LEN = 70.0, 50
CAND_COLS = ['ted_id', 'chain', 'chopping', 'length', 'plddt', 'cath_label', 'cath_level', 'cath_method',
             'cath_class', 'tier', 'chain_len']
LOC_COLS = ['chain', 'key', 'offset', 'length']
UA = {'User-Agent': 'protein-shape-census/1 (research; CPU only)'}


def det_gzip_text(path):
    """Text writer for a byte-reproducible .gz: no file name and mtime 0 in the gzip header."""
    fh = open(path, 'wb')
    raw = gzip.GzipFile(filename='', mode='wb', fileobj=fh, mtime=0)
    raw.myfileobj = fh  # so that close() also closes the file
    return io.TextIOWrapper(raw, newline='\n')


def tier_of(label, level):
    """primary = H-level label in CATH class 2 or 3; the rest are kept, flagged, for the secondaries."""
    if label in ('', '-'):
        return 'unlabelled'
    cls = label.split('.')[0]
    if level == 'H' and label.count('.') == 3:
        return 'primary' if cls in ('2', '3') else 'H_other_class'
    if level == 'T':
        return 'T_level'
    return 'other'


def cmd_candidates(a):
    chain_len, n_clusters = {}, Counter()
    with gzip.open(a.f2, 'rt') as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            n_clusters['clusters'] += 1
            if p[1] == '1':
                n_clusters['esm_only'] += 1
                n_clusters['esm_only_non_MGYP'] += not p[0].startswith('MGYP')
                chain_len[p[0]] = p[7]
    c = Counter()
    seen = set()
    with gzip.open(a.f5, 'rt') as fh, det_gzip_text(a.out) as out:
        out.write('\t'.join(CAND_COLS) + '\n')
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) != 9:
                c['bad_field_count'] += 1
                continue
            did, rep, bnd, length, pl, label, level, method, glob = p
            c['domains_total'] += 1
            if rep not in chain_len:
                continue
            c['domains_esm_only'] += 1
            if did in seen:
                c['duplicate_domain_id'] += 1
            seen.add(did)
            if float(pl) < MIN_PLDDT:
                continue
            c['plddt70'] += 1
            if glob != '1':
                continue
            c['plddt70_globular'] += 1
            if int(length) < MIN_LEN:
                continue
            c['plddt70_globular_len50'] += 1
            tier = tier_of(label, level)
            c['tier_' + tier] += 1
            if tier == 'primary':
                c['primary_class' + label.split('.')[0]] += 1
            out.write('\t'.join([did, rep, bnd, length, pl, label, level, method,
                                 label.split('.')[0] if label != '-' else '-', tier, chain_len[rep]]) + '\n')
    prim_chains = set()
    with gzip.open(a.out, 'rt') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['tier'] == 'primary':
                prim_chains.add(r['chain'])
    c['primary_chains'] = len(prim_chains)
    print(json.dumps(dict(n_clusters), sort_keys=True))
    print(json.dumps(dict(c), sort_keys=True))


def read_candidates(path, tiers=None):
    with gzip.open(path, 'rt') as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    return [r for r in rows if tiers is None or r['tier'] in tiers]


def stream_lines(url, start, end, slack=4096, block=8 << 20):
    """Yield (line_start_offset, line_bytes) for every line that begins in [start, end) of a remote file.

    The request starts one byte early so that a line beginning exactly at `start` is recognised; the
    partial line before the first newline belongs to the previous segment and is dropped.
    """
    lo = max(start - 1, 0)
    req = urllib.request.Request(url, headers=dict(UA, Range=f'bytes={lo}-{end + slack - 1}'))
    carry, carry_off, skipping = b'', lo, start > 0
    with urllib.request.urlopen(req, timeout=120) as r:
        while True:
            blk = r.read(block)
            if not blk:
                break
            buf = carry + blk
            base = carry_off
            cut = buf.rfind(b'\n')
            if cut < 0:
                carry = buf
                continue
            body, carry = buf[:cut + 1], buf[cut + 1:]
            carry_off = base + cut + 1
            off = base
            for ln in body.split(b'\n')[:-1]:
                n = len(ln) + 1
                if skipping:
                    skipping = False
                    off += n
                    continue
                if off >= end:
                    return
                yield off, ln
                off += n
    if carry and not skipping and carry_off < end:
        yield carry_off, carry


def remote_size(url):
    req = urllib.request.Request(url, method='HEAD', headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length']), r.headers.get('ETag'), r.headers.get('Last-Modified')


def _scan_segment(args, tries=6):
    url, start, end, mode, wanted = args
    for attempt in range(tries):  # a dropped connection restarts the segment, never the whole scan
        hits, n = {}, 0
        try:
            for _, ln in stream_lines(url, start, end):
                n += 1
                f = ln.split(b'\t')
                if mode == 'lookup':
                    if f[1] in wanted:
                        hits[f[1]] = f[0]
                else:
                    if f[0] in wanted:
                        hits[f[0]] = (f[1], f[2])
            return n, hits
        except (OSError, IOError) as e:  # TimeoutError, URLError, ConnectionError are OSError subclasses
            print(f'segment {start} retry {attempt + 1}: {e!r}', file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'segment {start}-{end} of {url} failed {tries} times')


def scan_remote(url, mode, wanted, seg_mb, conns, limit_mb=0):
    size, etag, lm = remote_size(url)
    if limit_mb:
        size = min(size, limit_mb << 20)
    segs = [(url, s, min(s + (seg_mb << 20), size), mode, wanted) for s in range(0, size, seg_mb << 20)]
    hits, lines, done, t0 = {}, 0, 0, time.time()
    with ThreadPoolExecutor(conns) as ex:
        for n, h in ex.map(_scan_segment, segs):
            hits.update(h)
            lines += n
            done += 1
            if done % 10 == 0 or done == len(segs):
                el = time.time() - t0
                print(f'{mode}: {done}/{len(segs)} segments, {lines} lines, {len(hits)} hits, {el:.0f}s',
                      file=sys.stderr, flush=True)
    return hits, lines, dict(url=url, bytes=size, etag=etag, last_modified=lm, lines=lines,
                             seconds=round(time.time() - t0, 1))


def cmd_resolve(a):
    chains = sorted({r['chain'] for r in read_candidates(a.cand, set(a.tiers.split(',')) if a.tiers else None)})
    if a.max_chains:
        chains = chains[:a.max_chains]
    wanted = {c.encode() for c in chains}
    print(f'{len(chains)} chains to resolve', file=sys.stderr)
    cache = Path(a.out + '.lookup_cache.json')  # chain -> key; lets a failed index scan resume without re-reading the lookup
    if cache.exists():
        c = json.loads(cache.read_text())
        name2key, info_l = {k.encode(): v.encode() for k, v in c['name2key'].items()}, c['info']
    else:
        name2key, _, info_l = scan_remote(FOLDCOMP_URL + '.lookup', 'lookup', wanted, a.seg_mb, a.conns, a.limit_mb)
        cache.write_text(json.dumps(dict(info=info_l, name2key={k.decode(): v.decode() for k, v in name2key.items()})))
    keys = {k: n for n, k in name2key.items()}
    key2loc, _, info_i = scan_remote(FOLDCOMP_URL + '.index', 'index', set(keys), a.seg_mb, a.conns, a.limit_mb)
    found = 0
    with det_gzip_text(a.out) as out:
        out.write('\t'.join(LOC_COLS) + '\n')
        for c in chains:
            k = name2key.get(c.encode())
            if k is None or k not in key2loc:
                continue
            off, ln = key2loc[k]
            out.write(f'{c}\t{k.decode()}\t{off.decode()}\t{ln.decode()}\n')
            found += 1
    meta = dict(requested=len(chains), in_lookup=len(name2key), located=found, lookup=info_l, index=info_i)
    Path(a.out + '.meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def fetch_range(offset, length, url=FOLDCOMP_URL, tries=4):
    req = urllib.request.Request(url, headers=dict(UA, Range=f'bytes={offset}-{offset + length - 1}'))
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) != length:
                raise IOError(f'short read {len(data)}/{length}')
            return data
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + 2 * i)


def fetch_foldcomp_pdb(offset, length):
    """PDB text of one esmatlas entry range-read from the Foldcomp data file."""
    import foldcomp
    _name, text = foldcomp.decompress(fetch_range(offset, length))  # (TITLE string, PDB text)
    return text


def pdb_ca(pdb_text):
    """(residue count, mean pLDDT over residues, CA coordinate array) from PDB text; B-factor is pLDDT."""
    import numpy as np
    xs, bs = [], []
    for l in pdb_text.splitlines():
        if l.startswith('ATOM') and l[12:16].strip() == 'CA':
            xs.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            bs.append(float(l[60:66]))
    return len(xs), (float(np.mean(bs)) if bs else float('nan')), np.array(xs)


# Failures of the run rather than of a domain: re-raised, never written as per-domain records,
# since the resume logic skips domains that already have a record.
WHOLE_RUN_FAILURES = (ImportError, ModuleNotFoundError, MemoryError, SystemError)


def preflight():
    """Fail loudly, before any record is written, if the interpreter cannot do the job."""
    missing = []
    for mod in ('foldcomp', 'gemmi'):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(f"preflight: missing {', '.join(missing)} in {sys.executable}")


def call_chain_task(args):
    rows, loc = args
    try:
        text = fetch_foldcomp_pdb(int(loc['offset']), int(loc['length']))
        n, _, _ = pdb_ca(text)
        if n != int(rows[0]['chain_len']):
            return [dict(r, ok=False, error=f'length mismatch {n} vs {rows[0]["chain_len"]}') for r in rows]
    except WHOLE_RUN_FAILURES:
        raise
    except Exception as e:
        return [dict(r, ok=False, error='fetch ' + repr(e)[:180]) for r in rows]
    return [call_domain(r, text) for r in rows]


def cmd_call(a):
    preflight()
    from multiprocessing import Pool
    loc = {}
    with gzip.open(a.locator, 'rt') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            loc[r['chain']] = r
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)['ted_id'] for l in open(a.out)}
    rows = [r for r in read_candidates(a.cand, set(a.tiers.split(','))) if r['ted_id'] not in done]
    if a.ids:
        keep = {l.strip() for l in open(a.ids)}
        rows = [r for r in rows if r['ted_id'] in keep]
    by_chain = {}
    for r in rows:
        by_chain.setdefault(r['chain'], []).append(r)
    n_rows = len(rows)
    del rows, done
    tasks = [(v, loc[k]) for k, v in sorted(by_chain.items()) if k in loc]
    # tasks carries everything needed; free the other copies of the candidate records
    by_chain.clear()
    loc.clear()
    del by_chain, loc
    gc.collect()
    print(f'{n_rows} domains on {len(tasks)} located chains', file=sys.stderr)
    n = 0
    # foldcomp.decompress grows the worker's memory; recycle workers every 500 chains
    with open(a.out, 'a') as out, Pool(min(a.workers, 2), maxtasksperchild=500) as p:
        for res in p.imap_unordered(call_chain_task, tasks):
            for r in res:
                out.write(json.dumps(r) + '\n')
            n += len(res)
            out.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('candidates'); c.add_argument('--f2', required=True); c.add_argument('--f5', required=True)
    c.add_argument('--out', required=True)
    r = sub.add_parser('resolve'); r.add_argument('--cand', required=True); r.add_argument('--out', required=True)
    r.add_argument('--tiers', default='', help='comma list of tiers to resolve (default all)')
    r.add_argument('--seg-mb', type=int, default=256); r.add_argument('--conns', type=int, default=4)
    r.add_argument('--limit-mb', type=int, default=0, help='scan only the first N MB of lookup/index (tests)')
    r.add_argument('--max-chains', type=int, default=0)
    k = sub.add_parser('call'); k.add_argument('--cand', required=True); k.add_argument('--locator', required=True)
    k.add_argument('--out', required=True); k.add_argument('--tiers', default='primary')
    k.add_argument('--workers', type=int, default=2); k.add_argument('--ids', default='')
    a = ap.parse_args()
    {'candidates': cmd_candidates, 'resolve': cmd_resolve, 'call': cmd_call}[a.cmd](a)


if __name__ == '__main__':
    main()
