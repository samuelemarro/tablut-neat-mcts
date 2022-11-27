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

chosen_population = 143

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Checkpointer.restore_checkpoint(f'neat-checkpoint-{chosen_population}')
    print(p.population.keys())
    winner = sorted([v for v in p.population.values() if v.fitness is not None], key=lambda v: v.fitness)[-1]

    with open('best_genome', 'wb') as f:
        pickle.dump(winner, f)

    """agent = GeneticAgent(neat.nn.FeedForwardNetwork.create(winner, config))
    opponent = RandomAgent()
    print(arena.battle(agent, opponent, num_battles=10))"""
    
    # visualize.draw_net(config, winner, True)#, node_names=node_names)
    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)