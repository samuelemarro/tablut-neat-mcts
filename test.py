import os
import numpy as np
os.environ["C_INCLUDE_PATH"] = np.get_include()
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},)


from agent import Agent
from tablut import *

import pstats, cProfile


from time import time

class FakeNetwork:
    def activate(self, x):
        return np.ones((len(x),))

def test():
    for i in range(20):
        a = Agent([lambda board, turn: 1], [lambda board, turn, moves: np.ones((len(moves),))], FakeNetwork())
        b = Board()
        a.monte_carlo_tree_search(b, -1)


cProfile.runctx("test()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()