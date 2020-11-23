### Import Packages
import random
import numpy as np
import pandas as pd
from util import *
import pygame

def train_DQN(env, agent, params={}):

    # fetch parameters
    batch_size = load_params(params, "batch_size", 64)
    n_episodes = load_params(params, "n_episodes", 100)
    max_moves = load_params(params, "max_moves", 1000)
    FPS = load_params(params, "FPS", 15)
    experience_replay = load_params(params, "exp_replay", True)

    model_dir = load_params(params, "model_dir", "my_model")
    save_per_episode = load_params(params, "save_per_episode", 1)

    # TRAINING

    done = False

    # Save data for performance analysis
    performance = pd.DataFrame.from_dict({
        "e": [], "reward": [], "score": [], "move": []
    })

    for e in range(1, n_episodes+1):

        # Step 1: Initialization
        state = env.reset()

        cum_reward = 0

        # Step 2: Simulate one trial of the game
        for move in range(1, max_moves+1):
            # visualize the game
            pygame.event.pump()
            if FPS != 0:
                env.render(FPS=FPS)

            # simulate action and outcomes
            action = agent.act(state)
            next_state, reward, done, score = env.step(action)

            # memorize
            agent.remember(state, action, reward, next_state, done)

            # update simulation
            state = next_state
            cum_reward += reward

            if experience_replay:
                agent.replay(batch_size)

            if done:
                break

        # print the training result
        print("progress: {}/{}, score: {}, e: {:.2}, moves: {}/{}" \
                .format(e, n_episodes, score, agent.epsilon, move, max_moves))

        # without experience replay
        if not experience_replay:
            agent.replay(batch_size)

        # save model weight every 50 episodes
        if e % save_per_episode == 0:
            agent.save_weight("{}/weights/{:.0f}.hdf5".format(model_dir, e))
        
        # save training performance
        performance.loc[len(performance)] = [e, cum_reward, score, move]
        if e % save_per_episode == 0:
            performance.to_csv("{}/performance/{:.0f}.csv".format(model_dir, e), index=False)


def test_DQN(env, agent, params=None):

    # fetch parameters
    n_tests = load_params(params, "n_tests", 10)
    max_moves = load_params(params, "max_games", 1000)
    FPS = load_params(params, "FPS", 10)
    verbose = load_params(params, "verbose", True)

    # start testing
    done = 0

    # Save data for performance analysis
    # each row is one instance: [reward, score, move]
    performance = np.zeros((n_tests, 3))

    for e in range(n_tests):

        # Step 1: Initialization
        state = env.reset()
        cum_reward = 0

        # Step 2: Simulate one trial of the game
        for move in range(max_moves):
            # render
            if FPS != 0:
                env.render(FPS=FPS)
            # action
            action = agent.exploit(state)
            # step forward
            next_state, reward, done, score = env.step(action)
            # conclude round
            state = next_state
            cum_reward += reward    # (this will always -100pt upon dead)
            if done:
                break

        # Step 3: Update performance
        performance[e,:] = [cum_reward, score, move]

        # print the training result
        if verbose:
            print("progress: {}/{}, score: {}".format(e+1, n_tests, score))

    return performance