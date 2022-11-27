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

random_players = 10
max_winners = 10
depth = 8
count = 60
num_battles = 1

def eval_genomes(genomes, config):
    genomes = [genome[1] for genome in genomes]
    for genome in genomes:
        genome.fitness = 0
    genomes += [None] * random_players
    arena.tournament(genomes, config, max_winners, depth, count, num_battles=num_battles)
    """for i in range(0, len(genomes), 2):
        genome_1, genome_2 = genomes[i][1], genomes[i + 1][1]
        score = battle(genome_1, genome_2, config)
        print('Final score:', score)
        genome_1.fitness = score
        genome_2.fitness = -score"""

start_checkpoint = 181

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