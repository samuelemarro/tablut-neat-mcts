# cython: language_level=3
from agent import Agent

import cython
import numpy as np
cimport numpy as np

from tablut import *
import heuristics

@cython.boundscheck(False) 
@cython.wraparound(False)
def play_game(agent_black : Agent, agent_white : Agent):
    # agent_1 plays black, agent_2 plays white
    board = Board()
    turn = 1 # Black begins
    players = [agent_black, None, agent_white]
    previous_move = None
    previous_score = 0

    # agent_1 always goes first
    for i in range(50):
        move = players[turn + 1].play(board.copy(), turn, depth=4, count=24, previous_move=previous_move, previous_score=previous_score)
        
        if move is None:
            # Out-of-moves, instant loss
            return -turn
        previous_score = board.get_score_diff()
        board.execute_move(*move)
        turn = -turn
        previous_move = move

        winner = board.check_winner(check_no_moves=False)
        if winner != 0:
            return winner
        
    return board.get_score_diff()

def battle(agent_1, agent_2, num_battles=1):
    total_score = 0

    for _ in range(num_battles):
        for swapped in [False, True]:
            if swapped:
                agent_black = agent_2
                agent_white = agent_1
            else:
                agent_black = agent_1
                agent_white = agent_2
            
            score = play_game(agent_black, agent_white)
            if swapped:
                score *= -1
            # print('Inner score:', score)
            total_score += score

    return total_score / (num_battles * 2)
