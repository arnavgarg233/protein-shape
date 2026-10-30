"""Reduce a census call file to per-topology, per-superfamily chain counts under each credit rule.

The output (a few MB) is all that the matched comparisons and the composition model need from the
call files, which are hundreds of MB.

    python scripts/census/extract_credits.py CALLS.jsonl[.gz] OUT.json.gz [--kinds agree lenient union]
"""
import argparse
import gzip
import json

from protein_shape.support import KINDS, credits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('calls')
    ap.add_argument('out')
    ap.add_argument('--kinds', nargs='+', default=list(KINDS))
    a = ap.parse_args()
    cnt, size = credits(a.calls, None, tuple(a.kinds))
    out = dict(source=a.calls.split('/')[-1], sf_size=dict(sorted(size.items())),
               chains={k: {c: dict(sorted(v.items())) for c, v in sorted(cnt[k].items())} for k in a.kinds})
    with gzip.GzipFile(a.out, 'wb', mtime=0) as fh:
        fh.write(json.dumps(out, separators=(',', ':')).encode())
    print(f'{len(size)} superfamilies; codes per credit: ' + ', '.join(f'{k} {len(cnt[k])}' for k in a.kinds))


if __name__ == '__main__':
    main()
