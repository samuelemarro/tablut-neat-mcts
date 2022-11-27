# cython: language_level=3

import numpy as np
cimport numpy as np

from tablut import *
import heuristics

class Agent:
    def play(self, initial_board : Board, initial_turn, depth=50, count=100):
        raise NotImplementedError

class GeneticAgent(Agent):
    def __init__(self, network):
        self.network = network
    
    def get_moves_and_probs(self, board : Board, turn, previous_move, previous_score):
        valid_moves = board.get_legal_moves(turn)
        if len(valid_moves) == 0:
            return [], None
        else:
            in_values = heuristics.get_features(board, turn, valid_moves, previous_move, previous_score)
            parameters = self.network.activate(in_values)[:-1] # Last one is quality
            move_probability = heuristics.get_actions(board, turn, valid_moves, parameters, previous_move, previous_score)

            return valid_moves, move_probability

    def get_value(self, board : Board, turn, previous_move, previous_score):
        valid_moves = board.get_legal_moves(turn)
        in_values = heuristics.get_features(board, turn, valid_moves, previous_move, previous_score)
        return self.network.activate(in_values)[-1] * 0.75 + board.get_score_diff() * 0.25 # Last one is quality
        pass#in_values = np.array([feature(board, turn) for feature in self.in_features])
        #return self.network.activate(in_values)[-1]

    def choose_move(self, board : Board, turn, previous_move, previous_score):
        moves, move_probability = self.get_moves_and_probs(board, turn, previous_move, previous_score)

        if len(moves) == 0:
            return None
        
        chosen_move_index = np.random.choice(np.arange(len(moves)), p=move_probability)

        return moves[chosen_move_index]
    
    
    def tree_search(self, initial_board : Board, initial_turn, previous_move, depth=50):
        board, turn = initial_board, initial_turn
        previous_score = 0
        for i in range(depth):
            move = self.choose_move(board, turn, previous_move, previous_score)
            if move is None:
                # No available moves: automatic loss
                return -turn
            
            previous_move = move
            previous_score = board.get_score_diff()

            board.execute_move(*move)
            turn = -turn
            winner = board.check_winner(check_no_moves=False)
            if winner != 0:
                return winner
        
        # Didn't result in a true score, use a heuristic

        return self.get_value(board, turn, previous_move, previous_score)# * 0.75 + board.get_score() * 0.25
        # return board.get_score()#self.get_value(board, turn)
        #simple_score = board.get_score()
        #advanced_score = 

        #return board.get_score()

    def monte_carlo_tree_search(self, initial_board : Board, initial_turn, depth=50, count=100):
        moves, move_probability = self.get_moves_and_probs(initial_board, initial_turn, None, 0)

        current_scores = np.zeros((len(moves),))
        run_count = np.zeros((len(moves,)))

        actual_depths = np.random.randint(0, depth + 1, count)
        for i in range(count):
            chosen_move_index = np.random.choice(np.arange(len(moves)), p=move_probability)
            chosen_move = moves[chosen_move_index]

            new_board = initial_board.copy()
            new_board.execute_move(*chosen_move)
            new_turn = -initial_turn

            # If we're black, we flip the score
            score = self.tree_search(new_board, new_turn, chosen_move, depth=actual_depths[i]) * initial_turn

            current_scores[chosen_move_index] += score
            run_count[chosen_move_index] += 1
        
        return moves[np.argmax(current_scores)]
    
    def play(self, initial_board : Board, initial_turn, depth=50, count=100):
        return self.monte_carlo_tree_search(initial_board, initial_turn, depth=depth, count=count)

class RandomAgent(Agent):
    def play(self, initial_board : Board, initial_turn, depth=50, count=100):
        moves = initial_board.get_legal_moves(initial_turn)
        index = np.random.choice(np.arange((len(moves))))
        return moves[index]