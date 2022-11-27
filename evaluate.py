# cython: language_level=3
import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},)

from tablut import *
import arena
from agent import *
import os
import neat
import visualize
import pickle

chosen_population = 181

# depth_schedule = [6, 6] + ([8] * 100)
# count_schedule = [40, 80, 125, 250, 500, 1000] + ([1000] * 100)
max_winners = 1
depth = 8
count = 120
num_battles = 4
processes = 10

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Checkpointer.restore_checkpoint(f'neat-checkpoint-{chosen_population}')
    genomes = list(p.population.values())
    for genome in genomes:
        genome.fitness = 0
    from time import time
    start_time = time()
    best = arena.tournament(genomes, config, max_winners, depth, count, num_battles=num_battles, processes=processes)[0]
    print('Elapsed:', time() - start_time)

    with open('best_genome', 'wb') as f:
        pickle.dump(best, f)
    print('We\'re done!')


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)