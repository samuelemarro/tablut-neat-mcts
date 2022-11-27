import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},)

# import the necessary packages
import random
import sys
import threading

import argparse
import pickle
import numpy
import socket
import json
from requests.exceptions import ConnectionError

import numpy as np

from time import sleep

import click
import neat
import pickle

from agent_2 import RandomAgent, GeneticAgent
from tablut import Board

def readBytes(sock, number):
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
    return int.from_bytes(readBytes(sock, 4), byteorder='big')


def read_board_info(sock):
    # read board info and return them

    # read 4 bytes corresponding to the length of the sent data
    boardDataLength = read_message_length(sock)

    # reading the data sent by server
    boardByte = readBytes(sock, boardDataLength)

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


def encode_board(rawBoard):
    # transform a 2D array of the board to a numpy array readable by the neural network

    board = []

    for row in rawBoard:
        for square in row:
            if square == "EMPTY":
                # O
                board += [False, False, False, False, True]
            elif square == "WHITE":
                # W
                board += [False, False, False, True, False]
            elif square == "BLACK":
                # B
                board += [False, False, True, False, False]
            elif square == "KING":
                # K
                board += [False, True, False, False, False]
            elif square == "THRONE":
                # T
                board += [True, False, False, False, False]
            else:
                print("[ERROR] Not recognized board square value. Terminating...")
                exit(1)

    return board


def print_other_predictions(predFrom, predTo, labelFrom, labelTo):
    dictPredFrom = {}
    for i in range(len(predFrom[0])):
        dictPredFrom[labelFrom.classes_[i]] = predFrom[0][i]

    sortedPredFrom = sorted(dictPredFrom.items(), key=lambda x: x[1], reverse=True)

    dictPredTo = {}
    for i in range(len(predTo[0])):
        dictPredTo[labelTo.classes_[i]] = predTo[0][i]

    sortedPredTo = sorted(dictPredTo.items(), key=lambda x: x[1], reverse=True)

    print("From: ", end='')
    for p in sortedPredFrom[:3]:
        print(p[0] + " (" + "{0:.3f}".format(p[1]) + "); ", end='')
    print()
    print("To: ", end='')
    for p in sortedPredTo[:3]:
        print(p[0] + " (" + "{0:.3f}".format(p[1]) + "); ", end='')
    print()

    print("***************************************")
    input("Premi per continuare")


def generate_move(modelFrom, modelTo, labelFrom, labelTo, boardInfo, dictFromMoves, turn, lock):
    # generate the two parts of the move using the two neural networks provided

    # enable debug print
    debug = False

    board = encode_board(boardInfo["board"])
    # [(legal_from=e4, legal_to=e5), (e1, e0), (b2, b5) ... ], letters are rows and numbers are columns
    legal_moves = ck.get_legal_moves(boardInfo["board"], turn)

    '''
    # CANNOT USE THE CODE BELOW BECAUSE SOCKETS THEN IN THE END EXPLODE, DEBUG NEEDED
    # check instant win or instant lose
    to_remove = []
    for m in legal_moves:
        # black
        if turn:
            if evaluate_single_move(m, boardInfo["board"], 1, False) == 100:
                return {"from": m[0], "to": m[1], "turn": boardInfo["turn"]}
            elif evaluate_single_move(m, boardInfo["board"], 1, False) == -25:
                to_remove.append(m)
        # white
        else:
            if evaluate_single_move(m, boardInfo["board"], 0, False) == 100:
                return {"from": m[0], "to": m[1], "turn": boardInfo["turn"]}
            elif evaluate_single_move(m, boardInfo["board"], 0, False) == -25:
                to_remove.append(m)

    # remove stupid moves
    legal_moves = [mov for mov in legal_moves if mov not in to_remove]
    '''

    # black turn
    if turn:
        label_from_classes = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'b1', 'b2', 'b3', 'b4', 'b5',
                              'b6', 'b7', 'b8', 'b9', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'd1',
                              'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'e1', 'e2', 'e3', 'e4', 'e6', 'e7',
                              'e8', 'e9', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'g1', 'g2', 'g3',
                              'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                              'h9', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9']
        output_move_link = list(zip(label_from_classes, list(range(80))))
    # white turn
    else:
        label_from_classes = ['a1', 'a2', 'a3', 'a7', 'a8', 'a9', 'b1', 'b2', 'b3', 'b4', 'b6', 'b7', 'b8', 'b9',
                              'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'd2', 'd3', 'd4', 'd5', 'd6',
                              'd7', 'd8', 'e3', 'e4', 'e5', 'e6', 'e7', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8',
                              'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'h1', 'h2', 'h3', 'h4', 'h6',
                              'h7', 'h8', 'h9', 'i1', 'i2', 'i3', 'i7', 'i8', 'i9']
        output_move_link = list(zip(label_from_classes, list(range(65))))
    legal_index_from = ck.get_legal_index_from(legal_moves, output_move_link)

    if debug:
        print("\n\n------------------------------------\nNow we get a legal move:")
        print("\nlegal_moves = \n" + str(legal_moves))
        print("\n(indice dei neuroni di output della rete legali) legal_index_from = " + str(legal_index_from))
        print("\n(mosse collegate a tali indici) legal_moves_from = ")
        for j in range(len(legal_index_from)):
            print(label_from_classes[legal_index_from[j]])

    # make prediction about the From part of the move
    with lock:
        predFrom = modelFrom.predict(numpy.array([board]))
    if debug:
        print("\n\nNow we decide: FROM")

    # find the class label index with the largest corresponding probability
    i = predFrom.argmax(axis=1)[0]
    if debug:
        print("\ni = " + str(i))

    # while it is not choose a legal move, search for it.
    # 'predFrom[i] = -9999' is useful to discard illegal moves
    # check on list's emptiness it is needed in case there are no legal moves available
    if legal_index_from:
        while i not in legal_index_from:
            predFrom[0][i] = -9999
            i = predFrom.argmax(axis=1)[0]
            if debug:
                print("\ni-while = " + str(i))
                sleep(0.2)
    else:
        print("WARNING! No legal moves recognized (FL will occur)! That is boardInfo[board], you will probably notice" +
              " that all pawns of a color are missing or that none pawns of a color can move:\n" +
              str(boardInfo["board"]) + "\n")

    moveFrom = labelFrom.classes_[i]
    if debug:
        print("\nmoveFrom = " + str(moveFrom))

    # White: 65 possible from parts of the move (impossible: a4, a5, a6, b5, d1, e1, f1, e2, i4, i5, i6, h5, d9, e9, f9, e8).
    # Black: 80 possible from parts of the move (impossible: e5)
    # Only the choosen one is 1, the rest are 0
    moveFromList = [False] * len(dictFromMoves)
    indexOn = dictFromMoves[moveFrom]
    moveFromList[indexOn] = True

    # make prediction about the To part of the move
    with lock:
        predTo = modelTo.predict(numpy.array([board + moveFromList]))
    if debug:
        print("\n\nNow we decide: TO")

    # find the class label index with the largest corresponding probability
    i = predTo.argmax(axis=1)[0]
    moveTo = labelTo.classes_[i]
    if debug:
        print("\ni = " + str(i))
        print("\nmoveTo = " + str(moveTo))

    # while it is not choose a legal move, search for it.
    # 'predTo[i] = -9999' is useful to discard illegal moves
    # check on list's emptiness it is needed in case there are no legal moves available
    # in case at least one legal "from" has been found, it guaranteed that one legal "to" exists
    if legal_moves:
        while (moveFrom, moveTo) not in legal_moves:
            predTo[0][i] = -9999
            i = predTo.argmax(axis=1)[0]
            moveTo = labelTo.classes_[i]
            if debug:
                print("\ni-while = " + str(i))
                print("\nmoveTo-while = " + str(moveTo))
                sleep(0.2)

    move = {"from": moveFrom, "to": moveTo, "turn": boardInfo["turn"]}

    if debug:
        print(move["from"] + " -> " + move["to"])
    # printOtherPredictions(predFrom, predTo, labelFrom, labelTo)

    return move


def get_points(finalState, playerColor):
    if finalState == "DRAW":
        return 1
    elif finalState == "WHITEWIN":
        if playerColor == "W":
            return 3
        else:
            return 0
    elif finalState == "BLACKWIN":
        if playerColor == "W":
            return 0
        else:
            return 3
    elif finalState == "FAILWHITE":
        if playerColor == "W":
            return -1
        else:
            return 2
    elif finalState == "FAILBLACK":
        if playerColor == "W":
            return 2
        else:
            return -1


def connect_and_play(modelFrom, modelTo, labelFrom, labelTo, playerName, playerColor, port, championship, lock,
                   baseline_player, folder, enemy_name):
    # playerColor: B or W

    if playerColor == "W":
        # indicates the index that has to be 1 in the FromMove input array
        # white not possible moves removed: a4, a5, a6, b5, d1, e1, f1, e2, i4, i5, i6, h5, d9, e9, f9, e8
        dictFromMoves = {'a1': 0, 'b1': 1, 'c1': 2, 'g1': 3, 'h1': 4, 'i1': 5, 'a2': 6, 'b2': 7, 'c2': 8, 'd2': 9,
                         'f2': 10, 'g2': 11, 'h2': 12, 'i2': 13, 'a3': 14, 'b3': 15, 'c3': 16, 'd3': 17, 'e3': 18,
                         'f3': 19, 'g3': 20, 'h3': 21, 'i3': 22, 'b4': 23, 'c4': 24, 'd4': 25, 'e4': 26, 'f4': 27,
                         'g4': 28, 'h4': 29, 'c5': 30, 'd5': 31, 'e5': 32, 'f5': 33, 'g5': 34, 'b6': 35, 'c6': 36,
                         'd6': 37, 'e6': 38, 'f6': 39, 'g6': 40, 'h6': 41, 'a7': 42, 'b7': 43, 'c7': 44, 'd7': 45,
                         'e7': 46, 'f7': 47, 'g7': 48, 'h7': 49, 'i7': 50, 'a8': 51, 'b8': 52, 'c8': 53, 'd8': 54,
                         'f8': 55, 'g8': 56, 'h8': 57, 'i8': 58, 'a9': 59, 'b9': 60, 'c9': 61, 'g9': 62, 'h9': 63,
                         'i9': 64}
    else:
        # indicates the index that has to be 1 in the FromMove input array
        # black not possible moves removed: e5
        dictFromMoves = {'a1': 0, 'b1': 1, 'c1': 2, 'd1': 3, 'e1': 4, 'f1': 5, 'g1': 6, 'h1': 7, 'i1': 8, 'a2': 9,
                         'b2': 10, 'c2': 11, 'd2': 12, 'e2': 13, 'f2': 14, 'g2': 15, 'h2': 16, 'i2': 17, 'a3': 18,
                         'b3': 19, 'c3': 20, 'd3': 21, 'e3': 22, 'f3': 23, 'g3': 24, 'h3': 25, 'i3': 26, 'a4': 27,
                         'b4': 28, 'c4': 29, 'd4': 30, 'e4': 31, 'f4': 32, 'g4': 33, 'h4': 34, 'i4': 35, 'a5': 36,
                         'b5': 37, 'c5': 38, 'd5': 39, 'f5': 40, 'g5': 41, 'h5': 42, 'i5': 43, 'a6': 44, 'b6': 45,
                         'c6': 46, 'd6': 47, 'e6': 48, 'f6': 49, 'g6': 50, 'h6': 51, 'i6': 52, 'a7': 53, 'b7': 54,
                         'c7': 55, 'd7': 56, 'e7': 57, 'f7': 58, 'g7': 59, 'h7': 60, 'i7': 61, 'a8': 62, 'b8': 63,
                         'c8': 64, 'd8': 65, 'e8': 66, 'f8': 67, 'g8': 68, 'h8': 69, 'i8': 70, 'a9': 71, 'b9': 72,
                         'c9': 73, 'd9': 74, 'e9': 75, 'f9': 76, 'g9': 77, 'h9': 78, 'i9': 79}

    # connecting to the Tablut server on localhost
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(("localhost", port))
        # print("[INFO] Connected")

        write_player_name(sock, playerName)
        # print("[INFO] Sent player name: " + playerName)

        # initial board state
        boardInfo = read_board_info(sock)

        total_move = 0
        all_move = []

        while True:

            if (boardInfo["turn"] == "WHITE" and playerColor == "W") or (
                    boardInfo["turn"] == "BLACK" and playerColor == "B"):
                # player choose move
                # added 'turn' parameter: false white, true black
                turn = (boardInfo["turn"] == "BLACK")

                if not baseline_player:
                    move = generate_move(modelFrom, modelTo, labelFrom, labelTo, boardInfo, dictFromMoves, turn, lock)
                else:
                    print_debug = False
                    move = generateBaselineMove(boardInfo["board"], turn, boardInfo["turn"], print_debug)

                total_move += 1
                if baseline_player:
                    all_move.append(move)

                write_move(sock, move)

                # player read own move
                boardInfo = read_board_info(sock)

                if is_game_finished(boardInfo):
                    break

            # player read enemy move
            boardInfo = read_board_info(sock)

            if is_game_finished(boardInfo):
                break

        if baseline_player and get_points(boardInfo["turn"], playerColor) in [0, -1]:
            if playerColor == "W":
                name = folder + "BLACK__" + str(numpy.random.randint(0, sys.maxsize)) + "__" + str(enemy_name) + ".txt"
            else:
                name = folder + "WHITE__" + str(numpy.random.randint(0, sys.maxsize)) + "__" + str(enemy_name) + ".txt"
            with open(name, "w") as reportFile:
                reportFile.write("Baseline has lost with:\n" + str(all_move))

        championship.calculate(playerName, playerColor == "W", get_points(boardInfo["turn"], playerColor), total_move)

    # print("[INFO] Game finished: " + boardInfo["turn"])

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
@click.option('--genome-path', type=click.Path(exists=True, dir_okay=False), default='best_genome')
@click.option('--config-path', type=click.Path(exists=True, dir_okay=False), default='config-feedforward')
@click.option('--random', is_flag=True)
def main(color : str, timeout : int, ip : str, genome_path, config_path, random):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    color = color.lower()[0]

    if color == 'w':
        color = 1
        player_name = '"Ta-Marro WHITE"'
        PORT = 5800
    else:
        color = -1
        player_name = '"Ta-Marro BLACK"'
        PORT = 5801
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, PORT))
        print("[INFO] Connected")

        write_player_name(sock, player_name)
        print("[INFO] Sent player name: " + player_name)

        agent = load_agent(random, genome_path, config)

        # initial board state
        
        previous_board = Board() if color == -1 else None
        previous_score = None
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
            move = agent.play(board, turn, parallel=1, depth=5, timeout=timeout)
            print('Playing:', move)
            encoded = encode_move(move, turn)
            #print('Encoded:', encoded)
            write_move(sock, encoded)

            board.execute_move(*move)
            previous_score = board.get_score_diff()
            previous_board = board




if __name__ == '__main__':
    main()
    exit(1)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    #ap.add_argument("-mf", "--model-from", required=True, help="path to trained Keras model From")
    #ap.add_argument("-mt", "--model-to", required=True, help="path to trained Keras model To")
    #ap.add_argument("-lf", "--label-bin-from", required=True, help="path to label binarizer From")
    #ap.add_argument("-lt", "--label-bin-to", required=True, help="path to label binarizer To")
    ap.add_argument("-p", "--player-color", required=True, help="player color: 'W' for white, 'B' for black")
    #ap.add_argument("-b", "--baseline", required=True, help="is a baseline: '1'. Else: '0'")
    ap.add_argument("-ip", "--ipaddr", required=True, help="ip address of the server")
    ap.add_argument("-t", "--timeout", required=False, help="timeout")
    args = vars(ap.parse_args())

    # load the model and label binarizer
    print("[INFO] loading networks and label binarizers...")

    modelFrom = load_model(args["model_from"])
    modelTo = load_model(args["model_to"])

    labelFrom = pickle.loads(open(args["label_bin_from"], "rb").read())
    labelTo = pickle.loads(open(args["label_bin_to"], "rb").read())

    baseline_player = int(args["baseline"])

    ip = args["ipaddr"]

    player = args["player_color"]
    if player == "W":
        player_name = "NNWhite"
        PORT = 5800
        # indicates the index that has to be 1 in the FromMove input array
        # white not possible moves removed: a4, a5, a6, b5, d1, e1, f1, e2, i4, i5, i6, h5, d9, e9, f9, e8
        dictFromMoves = {'a1': 0, 'b1': 1, 'c1': 2, 'g1': 3, 'h1': 4, 'i1': 5, 'a2': 6, 'b2': 7, 'c2': 8, 'd2': 9,
                         'f2': 10, 'g2': 11, 'h2': 12, 'i2': 13, 'a3': 14, 'b3': 15, 'c3': 16, 'd3': 17, 'e3': 18,
                         'f3': 19, 'g3': 20, 'h3': 21, 'i3': 22, 'b4': 23, 'c4': 24, 'd4': 25, 'e4': 26, 'f4': 27,
                         'g4': 28, 'h4': 29, 'c5': 30, 'd5': 31, 'e5': 32, 'f5': 33, 'g5': 34, 'b6': 35, 'c6': 36,
                         'd6': 37, 'e6': 38, 'f6': 39, 'g6': 40, 'h6': 41, 'a7': 42, 'b7': 43, 'c7': 44, 'd7': 45,
                         'e7': 46, 'f7': 47, 'g7': 48, 'h7': 49, 'i7': 50, 'a8': 51, 'b8': 52, 'c8': 53, 'd8': 54,
                         'f8': 55, 'g8': 56, 'h8': 57, 'i8': 58, 'a9': 59, 'b9': 60, 'c9': 61, 'g9': 62, 'h9': 63,
                         'i9': 64}
    elif player == "B":
        player_name = "NNBlack"
        PORT = 5801
        # indicates the index that has to be 1 in the FromMove input array
        # black not possible moves removed: e5
        dictFromMoves = {'a1': 0, 'b1': 1, 'c1': 2, 'd1': 3, 'e1': 4, 'f1': 5, 'g1': 6, 'h1': 7, 'i1': 8, 'a2': 9,
                         'b2': 10, 'c2': 11, 'd2': 12, 'e2': 13, 'f2': 14, 'g2': 15, 'h2': 16, 'i2': 17, 'a3': 18,
                         'b3': 19, 'c3': 20, 'd3': 21, 'e3': 22, 'f3': 23, 'g3': 24, 'h3': 25, 'i3': 26, 'a4': 27,
                         'b4': 28, 'c4': 29, 'd4': 30, 'e4': 31, 'f4': 32, 'g4': 33, 'h4': 34, 'i4': 35, 'a5': 36,
                         'b5': 37, 'c5': 38, 'd5': 39, 'f5': 40, 'g5': 41, 'h5': 42, 'i5': 43, 'a6': 44, 'b6': 45,
                         'c6': 46, 'd6': 47, 'e6': 48, 'f6': 49, 'g6': 50, 'h6': 51, 'i6': 52, 'a7': 53, 'b7': 54,
                         'c7': 55, 'd7': 56, 'e7': 57, 'f7': 58, 'g7': 59, 'h7': 60, 'i7': 61, 'a8': 62, 'b8': 63,
                         'c8': 64, 'd8': 65, 'e8': 66, 'f8': 67, 'g8': 68, 'h8': 69, 'i8': 70, 'a9': 71, 'b9': 72,
                         'c9': 73, 'd9': 74, 'e9': 75, 'f9': 76, 'g9': 77, 'h9': 78, 'i9': 79}
    else:
        print("[ERROR] Player must be W (white) or B (black)")
        exit(1)

    # connecting to the Tablut server on localhost
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, PORT))
        print("[INFO] Connected")

        write_player_name(sock, player_name)
        print("[INFO] Sent player name: " + player_name)

        # initial board state
        boardInfo = read_board_info(sock)
        print("PRESS ENTER IF 'input()' IS IN THE LOOP !")

        while True:

            if (boardInfo["turn"] == "WHITE" and player == "W") or (boardInfo["turn"] == "BLACK" and player == "B"):
                # player choose move
                if not baseline_player:
                    move = generate_move(modelFrom, modelTo, labelFrom, labelTo, boardInfo, dictFromMoves,
                                        boardInfo["turn"] == "BLACK", threading.Lock())
                else:
                    print_debug = True
                    move = generateBaselineMove(boardInfo["board"], boardInfo["turn"] == "BLACK", boardInfo["turn"], print_debug)

                write_move(sock, move)

                # player read own move
                boardInfo = read_board_info(sock)
                input("PRESS ENTER FOR A MOVE !")
                if is_game_finished(boardInfo):
                    break

            # player read enemy move
            boardInfo = read_board_info(sock)

            if is_game_finished(boardInfo):
                break

        print("[INFO] Game finished: " + boardInfo["turn"])