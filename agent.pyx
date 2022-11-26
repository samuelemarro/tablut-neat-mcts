import numpy as np
import torch
import torch.nn as nn

from TablutLogic import *

class RescaleScore(nn.Module):
    def forward(self, x):
        return x * 2 - 1

class Agent:
    def __init__(self, in_features, action_features, network):
        self.in_features = in_features
        self.action_features = action_features

        self.network = network
    
    def get_moves_and_probs(self, board : Board, turn):
        in_values = np.array([feature(board, turn) for feature in self.in_features])
        out_weights = self.network.activate(in_values)#[:-1] # Last one is quality

        moves = board.get_legal_moves(turn)
        move_probability = np.zeros((len(moves,)))

        for i in range(len(self.action_features)):
            move_probability += out_weights[i] * self.action_features[i](board, turn, moves)
        
        move_probability /= move_probability.sum()
        
        return moves, move_probability

    
    def get_value(self, board : Board, turn):
        pass#in_values = np.array([feature(board, turn) for feature in self.in_features])
        #return self.network.activate(in_values)[-1]

    def choose_move(self, board : Board, turn):
        moves, move_probability = self.get_moves_and_probs(board, turn)
        
        chosen_move_index = np.random.choice(np.arange(len(moves)), p=move_probability)

        return moves[chosen_move_index]
    
    
    def tree_search(self, initial_board : Board, initial_turn, depth=50):
        board, turn = initial_board, initial_turn
        for i in range(depth):
            move = self.choose_move(board, turn)
            board.execute_move(*move)
            turn = -turn
            winner = board.check_winner(check_no_moves=False)
            if winner != 0:
                return winner
        
        # Didn't result in a true score, use a heuristic

        return board.get_score()#self.get_value(board, turn)
        #simple_score = board.get_score()
        #advanced_score = 

        #return board.get_score()

    def monte_carlo_tree_search(self, initial_board : Board, initial_turn, depth=50, count=100, exploration_factor=1):
        moves, move_probability = self.get_moves_and_probs(initial_board, initial_turn)

        current_scores = np.zeros((len(moves),))
        run_count = np.zeros((len(moves,)))

        for i in range(count):
            average_score = np.zeros((len(moves),))
            at_least_one = ~np.equal(run_count, 0)
            # Normalize score to [0, 1]
            average_score[at_least_one] = (current_scores[at_least_one] + 1) / (2 * run_count[at_least_one])
            final_score = move_probability * exploration_factor + average_score * (1 - exploration_factor)

            chosen_move_index = np.argmax(final_score)
            chosen_move = moves[chosen_move_index]

            new_board = Board(initial_board.pieces.copy())
            new_board.execute_move(*chosen_move)
            new_turn = -initial_turn

            # If we're black, we flip the score
            score = self.tree_search(new_board, new_turn, depth=depth) * initial_turn

            current_scores[chosen_move_index] += score
            run_count[chosen_move_index] += 1
        
        return moves[np.argmax(current_scores)]