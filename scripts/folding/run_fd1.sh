#!/usr/bin/env bash
# Folding test on the selected ferredoxin-type domains: per domain, replica exchange for Tf, then 20 trajectories at
# 0.92 Tf capped at 10 M steps. Run fd1_select.py first.
# Usage: bash scripts/folding/run_fd1.sh [DIR] [NWORKERS]    (DIR defaults to results/folding/fd1)
set -euo pipefail
cd "$(dirname "$0")/../.."
FD=${1:-results/folding/fd1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$FD/logs"
one() {
  nm=$1
  [ -f "$FD/remd_$nm.json" ] || python scripts/folding/remd.py "$nm" --summary "$FD/summary.json" \
    --structures "$FD/structures" --out "$FD" --steps 600000 --nrep 12 --tlo 0.60 --thi 1.50 > "$FD/logs/remd_$nm.log" 2>&1
  python scripts/folding/fold_traj.py "$nm" 0.92 20 10000000 --summary "$FD/summary.json" \
    --structures "$FD/structures" --out "$FD" > "$FD/logs/fold_$nm.log" 2>&1
}
export -f one
export FD
# largest domains first
python -c "import json,sys; d=json.load(open(sys.argv[1])); [print(k) for k in sorted(d, key=lambda k: -d[k]['nres'])]" \
  "$FD/summary.json" | xargs -P "${2:-6}" -I{} bash -c 'one {}'
python scripts/folding/fd1_analyse.py --dir "$FD"
