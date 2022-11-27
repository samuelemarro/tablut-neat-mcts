# cython: language_level=3
import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},)


import os

import neat
import visualize

from agent import Agent, GeneticAgent, RandomAgent
from tablut import Board
import arena


import multiprocessing as mp

import pstats, cProfile

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]



"""class FakeNetwork:
    def activate(self, x):
        return np.ones((len(x),))

def test():
    for i in range(10):
        a1 = Agent([lambda board, turn: 1], [lambda board, turn, moves: np.ones((len(moves),))], FakeNetwork())
        a2 = Agent([lambda board, turn: 1], [lambda board, turn, moves: np.ones((len(moves),))], FakeNetwork())
        play_game(a1, a2)

cProfile.runctx("test()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
assert False"""


def create_agent(genome, config):
    if genome is None:
        return RandomAgent()
    else:
        network_1 = neat.nn.FeedForwardNetwork.create(genome, config)
        return GeneticAgent(network_1)
def genetic_battle(pairing):
    genome_1, genome_2, config, num_battles = pairing
    agent_1 = create_agent(genome_1, config)
    agent_2 = create_agent(genome_2, config)

    return arena.battle(agent_1, agent_2, num_battles=num_battles)

def tournament(genomes, config, max_winners, num_battles=1):
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
            pairings.append((genome_1, genome_2, config, num_battles))
    

    results = mp.Pool(processes=10).map(genetic_battle, pairings)

    for pairing, score in zip(pairings, results):
        genome_1, genome_2, _, _ = pairing
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
        tournament(winning_genomes, config, max_winners)

random_players = 10
max_winners = 10

def eval_genomes(genomes, config):
    genomes = [genome[1] for genome in genomes]
    for genome in genomes:
        genome.fitness = 0
    genomes += [None] * random_players
    tournament(genomes, config, 10)
    """for i in range(0, len(genomes), 2):
        genome_1, genome_2 = genomes[i][1], genomes[i + 1][1]
        score = battle(genome_1, genome_2, config)
        print('Final score:', score)
        genome_1.fitness = score
        genome_2.fitness = -score"""

start_checkpoint = 118

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if start_checkpoint is None:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(f'neat-checkpoint-{start_checkpoint}')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    print('Done!')
    return
    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)