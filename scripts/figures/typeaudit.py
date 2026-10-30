"""Re-open saved figure PDFs with PyMuPDF and check that every rendered text span is at least the placed-size
floor and is set in Arial regular or bold, including embedded fonts.

Usage: python typeaudit.py file.pdf [file.pdf ...]
"""
import sys

import fitz

# Figures are authored at 7.0 in and placed at 6.27 in; the floor applies to the placed size.
PLACED_SCALE = 6.27 / 7.0
MIN_PT_PLACED = 6.0
MIN_PT = MIN_PT_PLACED / PLACED_SCALE
ALLOWED_FONTS = {'ArialMT', 'Arial-BoldMT'}


def _base(name):
    return name.split('+', 1)[-1]   # strip the 'ABCDEF+' subset prefix


def audit(path):
    """Return (n_spans, min_size_pt, fonts_used, errors)."""
    errs, sizes, fonts = [], [], set()
    doc = fitz.open(path)
    for pno, page in enumerate(doc):
        for b in page.get_text('dict')['blocks']:
            if b['type'] != 0:
                continue
            for line in b['lines']:
                for s in line['spans']:
                    if not s['text'].strip():
                        continue
                    where = f'p{pno + 1} ({s["bbox"][0]:.0f}, {s["bbox"][1]:.0f}) {s["text"]!r}'
                    sizes.append(s['size'])
                    fonts.add(_base(s['font']))
                    if s['size'] < MIN_PT - 1e-6:
                        errs.append(f'placed size {s["size"] * PLACED_SCALE:.2f} pt '
                                    f'< {MIN_PT_PLACED} (authored {s["size"]:.2f}): {where}')
                    if _base(s['font']) not in ALLOWED_FONTS:
                        errs.append(f'font {s["font"]} not in {sorted(ALLOWED_FONTS)}: {where}')
        for f in page.get_fonts():
            if _base(f[3]) not in ALLOWED_FONTS:
                errs.append(f'embedded font {f[3]} not in {sorted(ALLOWED_FONTS)}')
    return len(sizes), (min(sizes) if sizes else float('nan')), sorted(fonts), list(dict.fromkeys(errs))


def main(paths):
    bad = False
    for p in paths:
        n, mn, fonts, errs = audit(p)
        print(f'type audit {"FAIL" if errs else "PASS"} {p}: {n} spans, min rendered {mn:.2f} pt, fonts {fonts}')
        for e in errs:
            print('  ' + e)
        bad |= bool(errs)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
