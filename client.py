import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},)

import pickle
import socket
import json
from requests.exceptions import ConnectionError

import numpy as np

import click
import neat
import pickle

from agent import RandomAgent, GeneticAgent
from tablut import Board

def read_bytes(sock, number):
    # read <number> bytes from the socket <sock>
    data = b''
    emptyNum = 0
    while len(data) < number:
        b = sock.recv(1)
        if b != b'':
            data += b
        else:
            emptyNum += 1
            if emptyNum > 10:
                raise ConnectionError

    return data


def read_message_length(sock):
    # read server message length
    return int.from_bytes(read_bytes(sock, 4), byteorder='big')


def read_board_info(sock):
    # read board info and return them

    # read 4 bytes corresponding to the length of the sent data
    boardDataLength = read_message_length(sock)

    # reading the data sent by server
    boardByte = read_bytes(sock, boardDataLength)

    # byte to string
    boardString = boardByte.decode("utf-8")

    # JSON to dict
    return json.loads(boardString)


def write_player_name(sock, name):
    # write player name to the server

    # get the name bytes (UTF-8 is used on the Java server too!)
    nameBytes = bytes(name, 'UTF-8')

    if len(nameBytes) > 255:
        print("[ERROR] playerName is too long")
        exit(1)

    # server waits 4 bytes. They specify how many bytes will be written
    nameLen = bytearray()
    nameLen.append(0)
    nameLen.append(0)
    nameLen.append(0)
    nameLen.append(len(nameBytes))

    # first send the name length
    sock.sendall(nameLen)

    # then send the name
    sock.sendall(nameBytes)


def write_move(sock, move):
    # move dict to json (string)
    jsonMove = json.dumps(move)

    # json to bytes
    bytesMove = bytearray()
    bytesMove.extend(jsonMove.encode())

    moveLen = bytearray()
    moveLen.append(0)
    moveLen.append(0)
    moveLen.append(0)
    moveLen.append(len(bytesMove))

    # first send the move length
    sock.sendall(moveLen)

    # then send the bytes move
    sock.sendall(bytesMove)


def is_game_finished(boardInfo):
    turn = boardInfo["turn"]
    return turn == "WHITEWIN" or turn == "BLACKWIN" or turn == "DRAW" or turn == "FAILBLACK" or turn == "FAILWHITE"


def parse_board_info(board_info):
    if board_info['turn'] == 'WHITE':
        turn = 1
    else:
        turn = -1
    
    board = np.zeros((9, 9))

    for y, row in enumerate(board_info['board']):
        for x, element in enumerate(row):
            if element == 'WHITE':
                element_code = 1
            elif element == 'KING':
                element_code = 2
            elif element == 'BLACK':
                element_code = -1
            else:
                element_code = 0
            board[y, x] = element_code
    
    return Board(board), turn

def to_letter_number(point):
    letters = 'abcdefghi'
    return letters[point[1]] + str(point[0] + 1)

def encode_move(move, turn):
    from_, to_ = move
    return {
        'from': to_letter_number(from_),
        'to' : to_letter_number(to_),
        'turn' : 'WHITE' if turn == 1 else 'BLACK'
    }

def find_previous_move(previous_board, new_board, color):
    try:
        if previous_board is None:
            return None
        converted_previous_board = np.zeros((9, 9))# previous_board.copy()
        converted_new_board = np.zeros((9, 9)) #new_board.copy()
        for y in range(9):
            for x in range(9):
                previous = previous_board.get_allegiance(previous_board.pieces[y, x])
                new = new_board.get_allegiance(new_board.pieces[y, x])
                if previous != color and previous != 0:
                    converted_previous_board[y, x] = 1
                if new != color and new != 0:
                    converted_new_board[y, x] = 1
        
        # print('Converted new:', converted_new_board)
        diff = converted_new_board - converted_previous_board
        # print('Diff:', diff)
        start_positions = np.argwhere(diff < 0)
        # print('Start positions:', start_positions)
        assert len(start_positions) == 1
        end_positions = np.argwhere(diff > 0)
        assert len(end_positions) == 1

        return tuple(start_positions[0]), tuple(end_positions[0])
    except:
        # Sometimes the server executes the command incorrectly
        # As a failsafe, we return None
        return None

def load_agent(random, genome_path, config):
    if random:
        return RandomAgent()
    
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    network = neat.nn.FeedForwardNetwork.create(genome, config)

    return GeneticAgent(network)

@click.command()
@click.argument('color', type=str)
@click.argument('timeout', type=int)
@click.argument('ip', type=str)
@click.option('--genome-path', type=click.Path(exists=True, dir_okay=False), default='best_genome', help='Path to genome file.', show_default=True)
@click.option('--config-path', type=click.Path(exists=True, dir_okay=False), default='config-feedforward', help='Path to configuration file.', show_default=True)
@click.option('--random', is_flag=True, help='If used, replaces the player with a random move agent.')
@click.option('--depth', type=int, help='Monte Carlo Tree Search depth.', default=8, show_default=True)
@click.option('--parallel', type=int, help='Number of parallel threads.', default=4, show_default=True)
def main(color : str, timeout : int, ip : str, genome_path, config_path, random, depth, parallel):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    original_color = color
    color = color.lower()[0]

    if color == 'w':
        color = 1
        player_name = '"Ta-Marro WHITE"'
        PORT = 5800
    elif color == 'b':
        color = -1
        player_name = '"Ta-Marro BLACK"'
        PORT = 5801
    else:
        print(f'Unrecognized color "{original_color}". Valid options: WHITE/W, BLACK/B (case-insensitive).')
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, PORT))
        print("[INFO] Connected")

        write_player_name(sock, player_name)
        print("[INFO] Sent player name: " + player_name)

        agent = load_agent(random, genome_path, config)

        # initial board state
        
        previous_board = Board() if color == -1 else None
        previous_score = 0
        #i = 0
        while True:
            board_info = read_board_info(sock)

            if is_game_finished(board_info):
                break

            # print(board_info)
            board, turn = parse_board_info(board_info)

            if turn != color:
                print('[INFO] Not my turn.')
                continue

            #if previous_board is not None:
            #    print((~np.equal(board.pieces, 0)).astype(float) - (~np.equal(previous_board.pieces, 0)).astype(float))
            #    i += 1
            #    if i == 2:
            #        assert False
            previous_move = find_previous_move(previous_board, board, color)
            print('Previous move:', previous_move)

            #print('Parsed:', parse_board_info(board_info))
            try:
                # .925 is our failsafe margin
                move = agent.play(board, turn, parallel=parallel, depth=depth, timeout=(timeout*.925), previous_move=previous_move, previous_score=previous_score)
            except:
                # If for some unknown reason the agent failed, we send back something
                move = RandomAgent().play(board, turn)
            print('Playing:', move)
            encoded = encode_move(move, turn)
            #print('Encoded:', encoded)
            write_move(sock, encoded)

            board.execute_move(*move)
            previous_score = board.get_score_diff()
            previous_board = board




if __name__ == '__main__':
    main()