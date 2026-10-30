# An unwritten rule of β-sheet architecture revealed by a million predicted protein structures

Code for the paper "An unwritten rule of β-sheet architecture revealed by a million predicted protein structures" (Nature Communications submission).

**Authors**: Anonymous authors

## Abstract

The β-sheets of natural proteins use only a small fraction of the strand orders that are geometrically possible, and a handful of rules about how strands are connected explain much of this choice. However, these rules were derived from the limited set of experimental structures, so it has remained unclear whether natural proteins obey constraints that the rules do not describe. Here, we measure how more than a million predicted protein domains use every sheet topology of five to seven strands and uncover a rule of sheet architecture that none of the published connection rules covers. Sheets built around a ferredoxin-type core, one of the most common arrangements in nature, are markedly less likely than comparable sheets to add a second pair of parallel crossovers that run the same way with interleaved strand intervals. The avoidance follows the order of the strands rather than whether the connections cross, holds against both matched references and a model of sheet composition, and reappears in an independent census of metagenomic proteins. A few small families, such as the PAN/apple and snake toxin-like domains, nonetheless build the arrangement as their standard topology. Overall, natural β-sheets follow constraints beyond the known rules, which bears on how protein folds evolve and on which topologies to choose for design.

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
```

## Reproducing the results

```bash
bash reproduce.sh
```

This runs the tests, recomputes Table 1 and Supplementary Table 2 from the tables in `data/` and checks every value against `results/tables/expected.json`, rebuilds the figures in `results/figures/`, and verifies `results/CHECKSUMS.txt`. It runs offline on CPU in a few minutes.

Steps that need external downloads or long runs are not part of `reproduce.sh`: building the census from AFDB and TED (`scripts/census/`), the topology frames and four-strand score (`scripts/topology/`, `scripts/composition/`), the robustness analyses that read structures (`scripts/robustness/`) and the Gō-model simulations (`scripts/folding/`). Each script documents its inputs with `--help`.

## Code structure

- `src/protein_shape/`: topology strings and enumeration, crossover-pair relations, the DSSP strand caller, superfamily support, matched-comparison statistics, the composition model and the Gō model.
- `scripts/`: `tables.py`, `make_figures.py` and the steps listed above.
- `data/`: derived per-topology tables; `data/manifest.json` lists each file and each external source with its version and licence.

## License

MIT. External data remain subject to their original terms. The topology enumeration and rule counting in `src/protein_shape/topology.py` follow the published scripts of the Koga laboratory (github.com/kogalab21/novel_ab-fold_design).
