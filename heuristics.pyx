# cython: language_level=3

# TODO: Controllare che le mosse valide vengano calcolate correttamente
import cython
import numpy as np
cimport numpy as np
from tablut import *
from scipy.special import softmax

@cython.boundscheck(False) 
@cython.wraparound(False)
def get_features(board, turn, valid_moves, previous_move, previous_score):
    king_position = np.argwhere(np.equal(board.pieces, P_KING))
    if len(king_position) == 0:
        king_moves = 0
        king_radius = 0
    else:
        king_position = king_position[0]
        king_moves = len(board.get_moves_for_square(tuple(king_position)))
        king_radius = np.abs(king_position - 4).sum()

    if previous_move is None:
        previous_move_diff = 0
    else:
        previous_move_radius_from = np.abs(previous_move[0][0] - 4) + np.abs(previous_move[0][1] - 4)
        previous_move_radius_to = np.abs(previous_move[1][0] - 4) + np.abs(previous_move[1][1] - 4)
        previous_move_diff = previous_move_radius_to - previous_move_radius_from

    # print('King position:', king_position)

    global_radius = np.mean(np.abs(np.argwhere(board.pieces)).sum(axis=1))
    current_score = board.get_score()
    return [
        turn, # Turn
        len(valid_moves) / 68 * 2 - 1, # Number of valid moves
        current_score, # Score
        current_score - previous_score,
        np.count_nonzero(board.pieces) / 25 * 2 - 1, # Surviving pieces
        king_radius / 8 * 2 - 1, # King radius
        king_moves / 16 * 2 - 1, # King moves
        global_radius / 8 * 2 - 1, # Global radius
        previous_move_diff / 8 * 2 - 1
    ]

@cython.boundscheck(False) 
@cython.wraparound(False)
def border(points):
    return np.logical_or(
        np.logical_or(np.equal(points[:, 0], 0), np.equal(points[:, 0], 8)),
        np.logical_or(np.equal(points[:, 1], 0), np.equal(points[:, 1], 8))
    )

@cython.boundscheck(False) 
@cython.wraparound(False)
def move_score(board : Board, move):
    board = board.copy()
    board.execute_move((move[0], move[1]), (move[2], move[3]))
    return board.get_score_diff()

@cython.boundscheck(False) 
@cython.wraparound(False)
def get_neighbors(board, points):
    counts = [np.zeros((len(points),)) for _ in range(3)]
    #counts_black = np.zeros((len(points),))
    #counts_king = np.zeros((len(points),))

    left_index = points[:, 1] > 0
    right_index = points[:, 1] < 8
    up_index = points[:, 0] > 0
    down_index = points[:, 0] < 8


    left = points[left_index] + np.array([[0, -1]])
    right = points[right_index] + np.array([[0, 1]])
    up = points[up_index] + np.array([[-1, 0]])
    down = points[down_index] + np.array([[1, 0]])

    for axis, index in [(left, left_index), (right, right_index), (up, up_index), (down, down_index)]:
        for color_index, color in enumerate([P_WHITE, P_KING, P_BLACK]):
            counts[color_index][index] += np.sum(np.equal(board.pieces[axis[:, 0], axis[:, 1]], color))

    return counts

@cython.boundscheck(False) 
@cython.wraparound(False)
def get_cross(board, points):
    row_counts = [np.zeros(9) for _ in range(3)]
    column_counts = [np.zeros(9) for _ in range(3)]

    for color_index, color in enumerate([P_WHITE, P_KING, P_BLACK]):
        eq = np.equal(board.pieces, color)
        row_counts[color_index] += np.sum(eq, axis=1)
        column_counts[color_index] += np.sum(eq, axis=0)
    

    final_counts = [np.zeros((len(points),)) for _ in range(3)]
    
    for color_index, color in enumerate([P_WHITE, P_KING, P_BLACK]):
        final_counts[color_index] += row_counts[color_index][points[:, 1]]
        final_counts[color_index] += column_counts[color_index][points[:, 0]]
        # Don't count the element itself (which appears both in the row and the column)
        final_counts[color_index] -= np.equal(board.pieces[points[:, 0], points[:, 1]], color).astype(np.float32) * 2

    return final_counts

@cython.boundscheck(False) 
@cython.wraparound(False)
def count_pieces(pieces, turn):
    if turn == P_WHITE:
        allied = pieces[0] + pieces[1]
        enemy = pieces[2]
    else:
        allied = pieces[2]
        enemy = pieces[0] + pieces[1]
    
    return allied, enemy, pieces[1]

@cython.boundscheck(False) 
@cython.wraparound(False)
def same_axis(points, reference_point):
    return np.logical_or(np.equal(points[:, 0], reference_point[0]), np.equal(points[:, 1], reference_point[1]))

@cython.boundscheck(False) 
@cython.wraparound(False)
def neighbor(points, reference_point):
    is_neighbor = np.zeros((len(points),), dtype=bool)
    for diff_y, diff_x in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        is_neighbor |= np.logical_and(
            np.equal(points[:, 0], reference_point[0] + diff_y),
            np.equal(points[:, 1], reference_point[1] + diff_x),
        )
    return is_neighbor

# vicinanze del punto di arrivo della mossa nemica
# vicino al re
# aggressive
# defensive
# avoidant
# start_radius
@cython.boundscheck(False) 
@cython.wraparound(False)
def get_actions(board, turn, valid_moves, parameters, previous_move, previous_score):
    values = np.ones((len(valid_moves), )) * parameters[0] # Baseline
    valid_moves = np.array([[from_y, from_x, to_y, to_x] for (from_y, from_x), (to_y, to_x) in valid_moves])
    
    values += turn * parameters[1]

    if turn == P_WHITE:
        # King move
        king_move = np.equal(board.pieces[valid_moves[:, 0], valid_moves[:, 1]], P_KING).astype(np.float32)
        values += king_move * parameters[2] * 2
    #else:
    #    leave_camp = np.logical_and(in_camp(valid_moves[:, :2]), ~in_camp(valid_moves[:, 2:4])).astype(np.float32)
    #    values += leave_camp * parameters[2] * 2


    original_radius = np.abs(valid_moves[:, :2] - 4).sum(axis=1)
    new_radius = np.abs(valid_moves[:, 2:4] - 4).sum(axis=1)
    values += original_radius / 8 * parameters[3]
    centrifuge = (new_radius - original_radius) / 4
    values += centrifuge * parameters[4]

    move_size = np.abs(valid_moves[:, :2] - valid_moves[:, 2:4]).sum(axis=1)
    values += move_size * parameters[5]

    original_scores = np.array([move_score(board, move) for move in valid_moves])
    score_diff = (original_scores - board.get_score_diff()) * turn
    values += score_diff * parameters[6]

    original_neighbors = count_pieces(get_neighbors(board, valid_moves[:, :2]), turn)
    new_neighbors = count_pieces(get_neighbors(board, valid_moves[:, 2:]), turn)

    original_cross = count_pieces(get_cross(board, valid_moves[:, :2]), turn)
    new_cross = count_pieces(get_cross(board, valid_moves[:, 2:]), turn)

    values += (original_neighbors[0] * parameters[7] + original_neighbors[1] * parameters[8] + original_neighbors[2] * parameters[9]) / 4
    values += (new_neighbors[0] * parameters[10] + new_neighbors[1] * parameters[11] + new_neighbors[2] * parameters[12]) / 4
    values += (original_cross[0] * parameters[13] + original_cross[1] * parameters[14] + original_cross[2] * parameters[15]) / 16
    values += (new_cross[0] * parameters[16] + new_cross[1] * parameters[17] + new_cross[2] * parameters[18]) / 16

    values += (border(valid_moves[:, :2]).astype(np.float32) - border(valid_moves[:, 2:]).astype(np.float32)) * parameters[19]


    if previous_move is not None:
        i = 20
        # previous_move non-Markovian magic
        for enemy_start in [True, False]:
            for axis in [True, False]:
                for allied_start in [True, False]:
                    reference_point = previous_move[0] if enemy_start else previous_move[1]
                    possible_points = valid_moves[:, :2] if allied_start else valid_moves[:, 2:]
                    if axis:
                        previous_move_scores = same_axis(possible_points, reference_point).astype(np.float32) * parameters[i]
                    else:
                        previous_move_scores = neighbor(possible_points, reference_point).astype(np.float32) * parameters[i]
                    i += 1

    return softmax(values)