# cython: language_level=3
from agent import Agent

import cython
import numpy as np
cimport numpy as np
import neat

from tablut import *
from agent import GeneticAgent, RandomAgent
import heuristics

import multiprocessing as mp

@cython.boundscheck(False) 
@cython.wraparound(False)
def play_game(agent_black : Agent, agent_white : Agent, depth, count):
    # agent_1 plays black, agent_2 plays white
    board = Board()
    turn = 1 # Black begins
    players = [agent_black, None, agent_white]
    previous_move = None
    previous_score = 0

    # agent_1 always goes first
    for i in range(50): # No more than 50 moves
        move = players[turn + 1].play(board.copy(), turn, depth=depth, count=count, previous_move=previous_move, previous_score=previous_score)
        
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

def battle(agent_1, agent_2, depth, count, num_battles=1):
    total_score = 0

    for _ in range(num_battles):
        for swapped in [False, True]:
            if swapped:
                agent_black = agent_2
                agent_white = agent_1
            else:
                agent_black = agent_1
                agent_white = agent_2
            
            score = play_game(agent_black, agent_white, depth, count)
            if swapped:
                score *= -1
            # print('Inner score:', score)
            total_score += score

    return total_score / (num_battles * 2)



def create_agent(genome, config):
    if genome is None:
        return RandomAgent()
    else:
        network_1 = neat.nn.FeedForwardNetwork.create(genome, config)
        return GeneticAgent(network_1)
def genetic_battle(pairing):
    genome_1, genome_2, config, depth, count, num_battles = pairing
    agent_1 = create_agent(genome_1, config)
    agent_2 = create_agent(genome_2, config)

    return battle(agent_1, agent_2, depth, count, num_battles=num_battles)


def fancy_tournament(genomes, config, max_winners, depth_schedule, count_schedule, num_battles=1, processes=10):
    print(f'Tournament with {len(genomes)} players')
    pairings = []
    winning_genomes = []

    np.random.shuffle(genomes)
    for i in range(0, len(genomes), 2):
        if i + 1 == len(genomes):
            # The last genome wins by default
            last_genome = genomes[i]
            if last_genome is not None:
                last_genome.fitness += 100
            winning_genomes.append(last_genome)
        else:
            genome_1, genome_2 = genomes[i], genomes[i + 1]
            pairings.append((genome_1, genome_2, config, depth_schedule[0], count_schedule[0], num_battles))
    

    results = mp.Pool(processes=processes).map(genetic_battle, pairings)

    for pairing, score in zip(pairings, results):
        genome_1, genome_2, _, _, _, _ = pairing
        if genome_1 is not None:
            genome_1.fitness += -score
        if genome_2 is not None:
            genome_2.fitness += score
        if score > 0:
            # Genome 2 won
            if genome_2 is not None:
                genome_2.fitness += 100
            winning_genomes.append(genome_2)
        elif score < 0:
            # Genome 1 won
            if genome_1 is not None:
                genome_1.fitness += 100
            winning_genomes.append(genome_1)
        else:
            # Tie
            if genome_1 is not None:
                genome_1.fitness += 50
            if genome_2 is not None:
                genome_2.fitness += 50
            winning_genomes.append(np.random.choice([genome_1, genome_2]))

    if len(winning_genomes) > max_winners:
        return fancy_tournament(winning_genomes, config, max_winners, depth_schedule[1:], count_schedule[1:])

    return winning_genomes

def tournament(genomes, config, max_winners, depth, count, num_battles=1, processes=10):
    return fancy_tournament(genomes, config, max_winners, [depth] * 100, [count] * 100, num_battles=num_battles, processes=processes)