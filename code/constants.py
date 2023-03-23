""" Some useful constants used throughout codebase """
import json

DATASETS = json.load(open("/home/renzo/workspace/epistatic_prior/nn4dms/data/datasets.json", "r"))

# list of chars that can be encountered in any sequence
CHARS = ["*", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# number of chars
NUM_CHARS = 21  # len(CHARS)

# dictionary mapping chars->int
C2I_MAPPING = {c: i for i, c in enumerate(CHARS)}
