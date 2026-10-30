"""Build every paper figure into results/figures/, each script in a fresh process with network access disabled,
then print the SHA256 of each output file."""
import hashlib
import subprocess
import sys
from pathlib import Path

from protein_shape.paths import FIGURES

HERE = Path(__file__).resolve().parent / 'figures'
SCRIPTS = ['fig1_concept.py', 'fig2_prediction.py', 'fig3_rule.py', 'fig4_strand_order.py',
           'supp_fig1_interleaving.py', 'supp_fig2_specificity.py']
WRAP = """
import runpy, socket, sys
sys.dont_write_bytecode = True
def _blocked(*a, **k):
    raise RuntimeError('network access attempted during figure build')
socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
sys.path.insert(0, {here!r})
runpy.run_path({script!r}, run_name='__main__')
"""


def main():
    for p in FIGURES.glob('*'):
        if p.suffix in ('.pdf', '.png'):
            p.unlink()
    for s in SCRIPTS:
        r = subprocess.run([sys.executable, '-c', WRAP.format(here=str(HERE), script=str(HERE / s))],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'{s} failed:\n{r.stdout}\n{r.stderr}')
        print(s, 'ok')
    for p in sorted(FIGURES.glob('*')):
        if p.suffix in ('.pdf', '.png'):
            print(hashlib.sha256(p.read_bytes()).hexdigest(), p.name)


if __name__ == '__main__':
    main()
