### Import Packages
import random
import numpy as np
import pandas as pd
from util import *
import pygame

def train_DQN(env, agent, params={}):

    # fetch parameters
    state_size = params["state_size"]
    action_size = params["action_size"]

    batch_size = load_params(params, "batch_size", 64)
    n_episodes = load_params(params, "n_episodes", 100)
    max_moves = load_params(params, "max_moves", 1000)
    FPS = load_params(params, "FPS", 10)
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
        state = np.reshape(state, [1,state_size])

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
            next_state = np.reshape(next_state, [1,state_size])

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
    state_size = STATE_SIZE
    action_size = ACTION_SIZE
    n_tests = N_TESTS
    max_moves = MAX_MOVES_TEST
    model_name = TEST_WEIGHT
    fps = FPS_TEST
    # state_size = params["state_size"]
    # action_size = params["action_size"]
    # n_tests = params["n_tests"]
    # max_games = params["max_games"]
    # model_name = params["model_name"]

    # load weights
    agent.load(model_name)

    # start testing
    done = 0

    for e in range(n_tests):

        # Step 1: Initialization
        state = env.reset()
        state = np.reshape(state, [1,state_size])

        # Step 2: Simulate one trial of the game
        for _ in range(max_moves):

            if fps != 0:
                env.render(FPS=fps)

            # use exploit() instead of act()
            action = agent.exploit(state)

            next_state, reward, done, score = env.step(action)

            state = np.reshape(next_state, [1,state_size])

            if done:
                break

        # print the training result
        print("progress: {}/{}, score: {}".format(e, n_tests, score))
