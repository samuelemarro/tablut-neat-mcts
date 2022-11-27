
import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},)

from agent import Agent
from tablut import *

b = Board()
print(len(b.get_legal_moves(-1)))
print(len(b.get_legal_moves(1)))
print(np.sum(np.abs(np.nonzero(b.pieces))))
import heuristics
print(heuristics.get_actions(b, 1, b.get_legal_moves(1), np.ones((19,))))