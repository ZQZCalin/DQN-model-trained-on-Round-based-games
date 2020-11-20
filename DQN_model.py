### Import Packages
import sys, os, time 
import importlib
import random
import numpy as np
import pandas as pd
from config import *
from util import *

def train_DQN(env, agent, params={}):

    # fetch parameters
    state_size = params["state_size"]
    action_size = params["action_size"]

    batch_size = load_params(params, "batch_size", 64)
    n_episodes = load_params(params, "n_episodes", 100)
    max_moves = load_params(params, "max_moves", 1000)
    FPS = load_param(params, "FPS", 10)
    experience_replay = load_params(params, "exp_replay", True)

    model_dir = load_params(params, "model_dir", "my_model")

    # TRAINING

    done = False

    # Save data for performance analysis
    performance = pd.DataFrame.from_dict({
        "e": [], "reward": [], "score", "move": []
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
            if fps != 0 and e % RPE == 0:
                env.render(FPS=fps)

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
        
        # save training performance
        performance.loc[len(performance)] = [e, cum_reward, score, move]

        # Step 3: Train DQN based on the agent's memory
        if not EXP_REPLAY:
            agent.replay(batch_size)

        # save model weight every 50 episodes
        if e % 50 == 0:
            agent.save(output_dir + "/weights_" + "{:.0f}".format(e) + ".hdf5")
    
    # End of Training
    train_dict = pd.DataFrame({
        "episode" : cv_episodes,
        "cumulative_reward": cv_cumulated_rewards,
        "moves" : cv_moves,
        "score" : cv_score
    })
    train_dict.to_csv(output_file)
    print("===== Training completed =====")
    print("weights are saved to: {}; performance is saved as: {}".format(output_dir, output_file))


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

if __name__ == "__main__":

    # load configuration
    """
    model_dir = input("Please enter your model directory:\nmodel/")

    print("===== loading configuration =====")
    time.sleep(0.5)

    # get a handle on the module
    mdl = importlib.import_module("models.%s.config" % model_dir)
    # is there an __all__?  if so respect it
    if "__all__" in mdl.__dict__:
        names = mdl.__dict__["__all__"]
    else:
        # otherwise we import all names that don't begin with _
        names = [x for x in mdl.__dict__ if not x.startswith("_")]
    # now drag them in
    globals().update({k: getattr(mdl, k) for k in names})

    print("===== configuration loaded =====")
    """

    # train
    if MODE == "TRAIN":
        # create model directory
        if not check_dir(MODEL_DIR, create=True):
            print("Created model directory as: {}".format(MODEL_DIR))

        # env, agent
        env = ENV
        agent = Agent(STATE_SIZE, ACTION_SIZE, model=SEQUENTIAL)

        # params
        TRAIN_WEIGHT = "{}/{}".format(MODEL_DIR, WEIGHT_DIR)
        PERFORMANCE_FILE = "{}/performance.csv".format(MODEL_DIR)
        ENV_AGENT_FILE = "{}/model.pkl".format(MODEL_DIR)
        CONFIG_FILE = "{}/config.txt".format(MODEL_DIR)

        config = {
            "name": NAME,
            "gamma": GAMMA,
            "epsilon": EPSILON,
            "epsilon_decay": EPSILON_DECAY,
            "epsilon_min": EPSILON_MIN,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "number_of_trains": N_TRAINS,
            "max_moves": MAX_MOVES_TRAIN,
            "notes": NOTES
        }

        """
        # save env and agent class for test purpose
        if not save_env_agent(env, agent, ENV_AGENT_FILE):
            sys.exit()
        else:
            print("Environment and agent are saved as: {}".format(ENV_AGENT_FILE))
        """
        
        # save config as documentation
        if not save_config(config, CONFIG_FILE):
            sys.exit()
        else:
            print("Config is saved as: {}".format(CONFIG_FILE))
        
        # check weight directory
        if not os.path.exists(TRAIN_WEIGHT):
            os.makedir(TRAIN_WEIGHT)
            print("Created weight directory as: {}".format(TRAIN_WEIGHT))
        else:
            text = "Are you sure to overwrite the existing weights?"
            if not yes_no(text):
                sys.exit()

        # check performance file
        if os.path.isfile(PERFORMANCE_FILE):
            text = "Are you sure to overwrite the existing performance?"
            if not yes_no(text):
                sys.exit()

        # train model
        train_DQN(env, agent)
    
    # test
    if MODE == "TEST":
        # load model
        if not check_dir(MODEL_DIR, create=False):
            print("Your model directory does not exist.")
            sys.exit()
        
        TEST_WEIGHT = "{}/{}".format(MODEL_DIR, TEST_WEIGHT)
        ENV_AGENT_FILE = "{}/model.pkl".format(MODEL_DIR)

        if not os.path.isfile(ENV_AGENT_FILE):
            print("Testing model does not exist.")
            sys.exit()
        
        env, agent = load_env_agent(ENV_AGENT_FILE)

        # test
        test_DQN(env, agent)

# ARCHIVED
"""
# if __name__ == "__main__" and False:

    # GAME = "CartPole-v0"
    GAME = "SNAKE"
    # mode = "TRAIN"
    mode = "TEST"

    if GAME == "CartPole-v0":
        # create environment
        env = gym.make("CartPole-v0")
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
    
    if GAME == "SNAKE":
        env = snakeEnv()
        state_size = env.state_size
        action_size = env.action_size

    if mode == "TRAIN":
        # set parameters
        params = {
            # size of state space
            "state_size" : state_size,
            # size of action space
            "action_size" : action_size,
            # batch size
            "batch_size" : 512,
            # number of games to train
            "n_episodes" : 1000,
            # maximum number of games in each epoch
            "max_moves" : 500,
            # output directory
            "output_dir" : "model_output_test_2",
        }

        # create agent
        agent = Agent(params["state_size"], params["action_size"])

        # Train DQN Network
        train_DQN(env, agent, params)

    if mode == "TEST":
        # Play Game
        test_params = {
            "state_size" : state_size,
            "action_size" : action_size,
            "max_games" : 500,
            "n_tests" : 10,
            "model_name" : "test_weight.hdf5",
        }

        agent = Agent(test_params["state_size"], test_params["action_size"])

        test_DQN(env, agent, test_params)
        # random_player(env, agent, test_params, verbose=0)
"""