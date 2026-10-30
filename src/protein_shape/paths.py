"""Repository-relative locations of the shipped data and the regenerated results."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
FIGURE_INPUTS = DATA / 'figure_inputs'
RESULTS = REPO / 'results'
TABLES = RESULTS / 'tables'
FIGURES = RESULTS / 'figures'

OCC_MAIN = DATA / 'occupancy_main.csv.gz'
OCC_REPLICATION = DATA / 'occupancy_replication.csv.gz'
CREDITS_MAIN = DATA / 'credits_main.json.gz'
CREDITS_REPLICATION = DATA / 'credits_replication.json.gz'
CLASS_RECALL_CENSUS = DATA / 'class_recall_census.json.gz'
COMPOSITION_COVARIATES = DATA / 'composition_covariates.csv.gz'
