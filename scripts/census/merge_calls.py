"""Merge census call shards into one call file, keeping the first record of each ted_id.

    python scripts/census/merge_calls.py calls.jsonl calls_shardB.jsonl calls_shardC.jsonl --out calls_merged.jsonl

Shards written by concurrent processes can overlap, and the support rule counts chains, so a duplicated
domain would be counted twice. A report is written beside the output as <out>.merge.json.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def read_records(path):
    """Yield (lineno, record) for every line, streaming. Only a trailing partial line is tolerated."""
    prev = prev_no = None
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if prev is not None:
                try:
                    yield prev_no, json.loads(prev)
                except json.JSONDecodeError:
                    raise SystemExit(
                        f'{path}: unparsable line {prev_no}, which is not the last line. '
                        f'Refusing to merge a damaged file.')
            prev, prev_no = line.strip(), lineno
    if prev:
        try:
            yield prev_no, json.loads(prev)
        except json.JSONDecodeError:
            print(f'{path}: last line unparsable (line {prev_no}), excluded as a partial write',
                  file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    if Path(a.out).exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite')

    seen = {}                      # ted_id -> (file index, lineno)
    per_file = []
    dup_within = Counter()
    dup_across = Counter()         # (first file, this file) -> count
    n_ok = n_written = 0
    label_types = Counter()

    # write to a temporary path and rename only on success
    tmp = Path(a.out + '.partial')
    try:
        with open(tmp, 'w') as out:
            for fi, path in enumerate(a.inputs):
                kept = total = 0
                for lineno, r in read_records(path):
                    total += 1
                    tid = r['ted_id']
                    # cath_label must stay a string: a float parse merges labels such as 3.30.70.1 and 3.30.70.10
                    label_types[type(r.get('cath_label')).__name__] += 1
                    if tid in seen:
                        prev_fi, _ = seen[tid]
                        if prev_fi == fi:
                            dup_within[path] += 1
                        else:
                            dup_across[(a.inputs[prev_fi], path)] += 1
                        continue
                    seen[tid] = (fi, lineno)
                    out.write(json.dumps(r) + '\n')
                    kept += 1
                    n_written += 1
                    n_ok += bool(r.get('ok'))
                per_file.append(dict(path=path, records=total, written=kept))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(a.out)

    report = dict(
        inputs=per_file,
        out=a.out,
        distinct_ted_ids=len(seen),
        written=n_written,
        ok=n_ok,
        not_ok=n_written - n_ok,
        duplicates_within_file=dict(dup_within),
        duplicates_across_files={f'{x} -> {y}': n for (x, y), n in dup_across.items()},
        cath_label_types=dict(label_types),
    )
    assert n_written == len(seen), (n_written, len(seen))
    non_str = {k: v for k, v in label_types.items() if k != 'str'}
    if non_str:
        report['WARNING_cath_label_not_string'] = non_str

    Path(a.out + '.merge.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
