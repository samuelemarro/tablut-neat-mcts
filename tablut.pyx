# cython: language_level=3

'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

# 0 = empty, 1 = white, 2 = king, -1 = black
# 0 = regular board, 1 = castle, 2 = safety, -1 = camp

# SAFETY is non-blocking
# You can capture a black piece in a camp, as long as you surround it properly (not camps)
# Capture needs to be fresh

from enum import Enum

import cython
import numpy as np
cimport numpy as np

P_EMPTY = 0
P_WHITE = 1
P_KING = 2
P_BLACK = -1

T_NORMAL = 0
T_CASTLE = 1
T_SAFETY = 2
T_CAMP = -1

CASTLE_POSITIONS = [(4, 4)]

POSITIONS_NEXT_TO_CASTLE = [(3, 4), (5, 4), (4, 3), (4, 5)]

SAFETY_POSITIONS = [
    (0, 1), (0, 2), (0, 6), (0, 7),
    (1, 0), (1, 8),
    (2, 0), (2, 8),
    (6, 0), (6, 8),
    (7, 0), (7, 8),
    (8, 1), (8, 2), (8, 6), (8, 7)
]
CAMP_POSITIONS = [
    (0, 3), (0, 4), (0, 5),
    (1, 4),
    (3, 0), (3, 8),
    (4, 0), (4, 1), (4, 7), (4, 8),
    (5, 0), (5, 8),
    (7, 4),
    (8, 3), (8, 4), (8, 5),
]

WHITE_POSITIONS = [
    (2, 4),
    (3, 4),
    (4, 2), (4, 3), (4, 5), (4, 6),
    (5, 4),
    (6, 4)
]

KING_POSITIONS = CASTLE_POSITIONS
BLACK_POSITIONS = CAMP_POSITIONS


class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, pieces=None):
        "Set up initial board configuration."

        self.n = 9

        self.board = np.ones((9, 9), dtype=np.int8) * T_NORMAL
        self.board[tuple(np.array(CASTLE_POSITIONS).T)] = T_CASTLE
        self.board[tuple(np.array(SAFETY_POSITIONS).T)] = T_SAFETY
        self.board[tuple(np.array(CAMP_POSITIONS).T)] = T_CAMP

        if pieces is None:
            self.pieces = np.ones((9, 9), dtype=np.int8) * P_EMPTY
            self.pieces[tuple(np.array(WHITE_POSITIONS).T)] = P_WHITE
            self.pieces[tuple(np.array(KING_POSITIONS).T)] = P_KING
            self.pieces[tuple(np.array(BLACK_POSITIONS).T)] = P_BLACK
        else:
            self.pieces = pieces

    def copy(self):
        return Board(pieces=self.pieces.copy())


    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.board[index], self.pieces[index]

    def countDiff(self, color : int): # TODO: Rewrite
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
                if self[x][y]==-color:
                    count -= 1
        return count
    
    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def get_legal_moves_old(self, color : int):
        moves = []
        blocked_by_camps = True
        for horizontal in [True, False]:
            for forward in [True, False]:
                for i in range(9):
                    current_piece = P_EMPTY
                    current_start_position = None
                    for j in range(9) if forward else range(8, -1, -1):
                        pos = (i, j) if horizontal else (j, i)
                        if current_piece != P_EMPTY:
                            if self.pieces[pos] == P_EMPTY and (self.board[pos] == T_NORMAL or self.board[pos] == T_SAFETY or (self.board[pos] == T_CAMP and not blocked_by_camps)):
                                moves.append((current_start_position, pos))
                        if self.pieces[pos] != P_EMPTY:
                            if self.get_allegiance(self.pieces[pos]) == color:
                                # New piece
                                current_piece = self.pieces[pos]
                                current_start_position = pos
                                blocked_by_camps = not(self.pieces[pos] == P_BLACK and self.board[pos] == T_CAMP)
                            else:
                                # Irrelevant
                                current_piece = P_EMPTY
                                current_start_position = None
        return moves


    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def get_legal_moves(self, color : int):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = []  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self.get_allegiance(self.pieces[y][x])==color:
                    newmoves = self.get_moves_for_square((y,x))
                    moves.extend([((y, x), (to_y, to_x)) for to_y, to_x in newmoves])
        # moves = list(moves)
        return moves

    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def has_legal_moves(self, color : int):
        for y in range(self.n):
            for x in range(self.n):
                if self.get_allegiance(self.pieces[y][x])==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    _move_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def rook_moves(self, square : tuple):
        moves = []
        for dy, dx in self._move_directions:
            y, x = square
            y += dy
            x += dx

            while self.in_board((y, x)) and self.pieces[y, x] == P_EMPTY and (self.board[y, x] == T_NORMAL or self.board[y, x] == T_SAFETY):
                moves.append((y, x))
                y += dy
                x += dx

        return moves

    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def rook_moves_black_in_camp(self, square : tuple):

        moves = []
        for dy, dx in self._move_directions:
            y, x = square
            y += dy
            x += dx
            left_camp = False

            while self.in_board((y, x)) and self.pieces[y, x] == P_EMPTY and self.board[y, x] != T_CASTLE and \
                (self.board[y, x] != T_CAMP or not left_camp):
                moves.append((y, x))
                y += dy
                x += dx
                if self.in_board((y, x)) and self.board[y, x] != T_CAMP:
                    left_camp = True

        return moves


    @cython.boundscheck(False) 
    @cython.wraparound(False)
    def get_moves_for_square(self, square):
        tile, piece = self[square]

        if piece == P_BLACK and tile == T_CAMP:
            # Black in camp, needs special treatment
            return self.rook_moves_black_in_camp(square)

        return self.rook_moves(square)

    def get_allegiance(self, piece):
        if piece == P_KING:
            return P_WHITE
        else:
            return piece

    def is_barrier(self, position, black_in_camp=False):
        # For a black piece in a camp, other pieces
        return self.board[position] in ([T_CASTLE] if black_in_camp else [T_CASTLE, T_CAMP])
    
    def in_board(self, position):
        y, x = position
        return y < 9 and x < 9 and y >= 0 and x >= 0

    def check_capture_king_near_castle(self, position):
        num_surrounders = 0

        start_y, start_x = position

        position_set = [
            (start_y, start_x - 1),
            (start_y, start_x + 1),
            (start_y - 1, start_x),
            (start_y + 1, start_x)
        ]

        for y, x in position_set:
            if self.in_board((y, x)):
                if self.get_allegiance(self.pieces[y, x]) == P_BLACK or self.is_barrier((y, x)):
                    # Counts as a surrounder
                    num_surrounders += 1
                else:
                    pass
                    # Note: end-of-board doesn't count as a barrier
        
        if num_surrounders == 4:
            # Note: a capture needs to be fresh, however it's impossible
            # to capture the king in the castle without a fresh capture
            return True
        
        return False

    # position: position of the enemy (e.g. black)
    # attacking_allegiance: who's doing the capturing (e.g. white)
    def check_capture(self, position, attacking_allegiance, attacking_position):
        piece = self.pieces[position]
        captured_allegiance = self.get_allegiance(piece)
        if captured_allegiance in [P_EMPTY, attacking_allegiance]:
            # Trying to capture either ally or empty
            return False
        #if self.get_allegiance(piece) == P_BLACK and self.board[position] == T_CAMP:
            # Black cannot be captured in a camp
            # That's false!

        if piece == P_KING and (self.board[position] == T_CASTLE or position in POSITIONS_NEXT_TO_CASTLE):
            # King near (or in) castle, follows special rules
            # print('King is following special rules')
            return self.check_capture_king_near_castle(position)

        start_y, start_x = position

        possible_surrounding_positions = [
            [(start_y, start_x - 1), (start_y, start_x + 1)],
            [(start_y - 1, start_x), (start_y + 1, start_x)]
        ]

        for position_set in possible_surrounding_positions:
            num_surrounders = 0
            active_capture = False
            for y, x in position_set:
                if self.in_board((y, x)):
                    black_in_camp = captured_allegiance == P_BLACK and self.board[start_y, start_x] == T_CAMP
                    if self.get_allegiance(self.pieces[y, x]) == attacking_allegiance or self.is_barrier((y, x), black_in_camp=black_in_camp):
                        num_surrounders += 1
                else:
                    pass
                    # Note: End-of-board doesn't count as a barrier
                if attacking_position == (y, x):
                    # The attacker is doing the actual capturing
                    active_capture = True

            if num_surrounders == 2 and active_capture: # TODO: Capture needs to be fresh
                # Surrounded!
                return True
        return False

    def execute_move(self, from_, to_):
        piece = self.pieces[from_]
        self.pieces[from_] = P_EMPTY
        self.pieces[to_] = piece

        to_y, to_x = to_
        possible_enemy_positions = [
            (to_y, to_x - 1),
            (to_y, to_x + 1),
            (to_y - 1, to_x),
            (to_y + 1, to_x)
        ]

        piece_allegiance = self.get_allegiance(piece)
        enemy_allegiance = P_BLACK if piece_allegiance == P_WHITE else P_WHITE

        for y, x in possible_enemy_positions:
            if self.in_board((y, x)) and self.get_allegiance(self.pieces[y, x]) == enemy_allegiance:
                # It's an enemy! Check if we captured it
                if self.check_capture((y, x), piece_allegiance, (to_y, to_x)):
                    self.pieces[y, x] = P_EMPTY

    def check_move(self, from_, to_, enforce_turn=None):
        if enforce_turn is not None and self.get_allegiance(self.pieces[from_]) != enforce_turn:
            return False
        return to_ in self.get_moves_for_square(from_)
    
    def check_winner(self, check_no_moves=True):
        king_indices = np.where(np.equal(self.pieces, P_KING))
        if len(king_indices) == 0 or len(king_indices[0]) == 0:
            # King was captured, black wins
            return P_BLACK
        king_position = (king_indices[0], king_indices[1])
        # assert len(king_positions) == 1
        if self.board[king_position] == T_SAFETY:
            # King has escaped, white wins
            return P_WHITE
        
        if check_no_moves:
            if not self.has_legal_moves(P_WHITE):
                return P_BLACK
            if not self.has_legal_moves(P_BLACK):
                return P_WHITE

        # Game continues
        return P_EMPTY

    def get_score_diff(self):
        return np.count_nonzero(np.equal(self.pieces, P_WHITE)) / 8 - np.count_nonzero(np.equal(self.pieces, P_BLACK)) / 16

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
            #print(x,y)
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])

    def tile_representation(self, position, character_override=None, color_override=None, pretty=True):
        tile, piece = self[position]

        from termcolor import colored

        if character_override is None:
            if piece == P_EMPTY:
                character = '.'
            elif piece == P_WHITE:
                character = 'W'
            elif piece == P_BLACK:
                character = 'B'
            elif piece == P_KING:
                character = 'K'
        else:
            character = character_override
   
        if pretty:     
            if color_override is None:
                if tile == T_CASTLE:
                    color = 'yellow'
                elif tile == T_CAMP:
                    color = 'red'
                elif tile == T_SAFETY:
                    color = 'blue'
                elif tile == T_NORMAL:
                    color = 'white'
            else:
                color = color_override

            return colored(character, color)
        else:
            return character

    def string_representation(self, pretty=True):
        s = ''
        
        if pretty:
            s += ' ' + ''.join([str(i) for i in range(9)]) + '\n'

        for y in range(9):
            if pretty:
                s += str(y)
            for x in range(9):
                s += self.tile_representation((y, x), pretty=pretty)
            s += '\n'
        return s
    
    def moves_representation(self, position, pretty=True):
        moves = self.get_moves_for_square(position)
        s = ''
        
        if pretty:
            s += ' ' + ''.join([str(i) for i in range(9)]) + '\n'

        for y in range(9):
            if pretty:
                s += str(y)
            for x in range(9):
                if (y, x) in moves:
                    if y == position[0]:
                        # Same y, it's a horizontal move
                        character_override = '-'
                    else:
                        character_override = '|'
                else:
                    character_override = None
                s += self.tile_representation((y, x), character_override=character_override, pretty=pretty)
            s += '\n'
        return s
    
    def get_score(self):
        return np.count_nonzero(np.equal(self.pieces, P_WHITE)) / 8 - np.count_nonzero(np.equal(self.pieces, P_BLACK)) / 16