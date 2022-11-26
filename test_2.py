import pyximport; pyximport.install()

from agent import Agent
from TablutLogic import *
import numpy as np

b = Board()
print(b.get_legal_moves(-1))
print(b.get_legal_moves(1))