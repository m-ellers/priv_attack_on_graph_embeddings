import multiprocessing
import os
from pathlib import Path

NUM_CORES = multiprocessing.cpu_count()  # may be too much for large machines

# path for installed embeddings and default data target directory
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

DIR_PATH = str(Path(PROJECT_PATH).parent) + "/"

# embedding paths
LINE_DIR = DIR_PATH + 'LINE/linux/'
GEM_EMBEDDING_DIR = DIR_PATH + "GEM"
NODE2VEC_SNAP_DIR = DIR_PATH + 'snap/examples/node2vec/'
