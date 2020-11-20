### Import Packages

import sys, os, time 
import importlib
import random
# import gym
import numpy as np
import pandas as pd
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
tf.compat.v1.reset_default_graph()

from config import *
from util import *


### DQN Agent

class Agent():
    """
    Parameters:
        - state_size: size of state space
        - action_size: size of action space
        - memory: a deque of length maxlen (2000) to store game experiences
        - gamma: coefficient of future rewards
        - epsilon: exploration coefficient
            - 1 for max exploration and 0 for max exploitation
            - initially set to be 1 because we want the agent to explore as much
              as it can at the beginning
            - epsilon is decreasing because we want the agent to learn from the
              past experiences
        - epsilon_decay: decrease rate of epsilon
        - epsilon_min: minimum threshold of epsilon
        - learning_rate: learning rate / step size
    """
    def __init__(self, state_size, action_size, model=None):
        self.state_size = state_size
        self.action_size = action_size

        # Define Hyper-parameters in config.py
        self.gamma = GAMMA

        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY 
        self.epsilon_min = EPSILON_MIN

        self.learning_rate = LEARNING_RATE

        # self.gamma = 0.95
        # self.epsilon = 1.0
        # self.epsilon_decay = 0.995
        # self.epsilon_min = 0.01
        # self.learning_rate = 0.00025

        if model == None:
            self.model = self._build_model()
        else:
            self.model = model
        
        self.memory = deque(maxlen=2000)
        self.rare_memory = deque(maxlen=500)

    def _build_model(self):
        """
        Model: a shallow network of 3 Dense layers
            - layer 1 (input): Dense of size 24 x state_size
            - layer 2: Dense of size 24 x 1
            - layer 3 (output): Dense of size action_size
            - loss: MSE (an unexpected choice)
            - optimizer: Adam with lr = self.learning_rate
        """
        model = Sequential()

        model.add(Dense(128, input_dim=self.state_size, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))

        # model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        """
        stores the game experience into self.memory
            - parameters: s^t, a^t, r^t, s^{t+1}
        """
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        """
        Action of the agent (2 modes):
            - mode 1: explore / act randomly
                - return a random action
            - mode 2: exploit
                - predict an optimal action based on the state and the model
                - return action with highest reward
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_value = self.model.predict(state)
            # TO DO: explain act_value
            # act_value is a action_size x 1 2D array,
            # so we fetch act_value[0].
            # Each element is the predicted reward.
            return np.argmax(act_value[0])

    def exploit(self, state):
        """
        exploit entirely on DQN model without randomness
        """
        return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):
        """
        Train the DQN network using memory
            - input X: current state s^t
            - target y:
        """
        if len(self.memory) < batch_size:
            return

        mini_batch = random.sample(self.memory, batch_size) \
            + random.sample(self.rare_memory, min(round(batch_size/4), len(self.rare_memory)))

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                """
                # predict the reward of the next state
                future_value = self.model.predict(next_state)
                # add maximum future reward
                target += self.gamma * np.amax(future_value[0])
                """
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            # target / y: max future reward mapped to the currect state
            target_state = self.model.predict(state)
            target_state[0][action] = target

            self.model.fit(state, target_state, epochs=1, verbose=0)
            # To Do: explain why train in this way?

        # decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        load model weights
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        save model weights
        """
        self.model.save_weights(name)

### Train DQN

def train_DQN(env, agent, params=None):

    # fetch parameters
    state_size = STATE_SIZE 
    action_size = ACTION_SIZE

    batch_size = BATCH_SIZE
    n_episodes = N_TRAINS
    max_moves = MAX_MOVES_TRAIN

    output_dir = TRAIN_WEIGHT       # weight directory
    output_file = PERFORMANCE_FILE  # performance

    fps = FPS_TRAIN
    # state_size = params["state_size"]
    # action_size = params["action_size"]
    # batch_size = params["batch_size"]
    # n_episodes = params["n_episodes"]
    # max_moves = params["max_moves"]
    # output_dir = params["output_dir"]

    done = 0

    # Save data for performance analysis
    cv_episodes = []
    cv_cumulated_rewards = []
    cv_score = []
    cv_moves = []

    for e in range(1,n_episodes+1):

        # Step 1: Initialization
        state = env.reset()
        state = np.reshape(state, [1,state_size])

        cum_reward = 0

        # Step 2: Simulate one trial of the game
        for i in range(1,max_moves+1):
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

            if EXP_REPLAY:
                agent.replay(batch_size)

            if done:
                # testing line: try to add a new rare_memory
                agent.rare_memory.append([state, action, reward, next_state, done])

                break

        # print the training result
        print("progress: {}/{}, score: {}, e: {:.2}, moves: {}/{}" \
                .format(e, n_episodes, score, agent.epsilon, i, max_moves))
        
        # save training performance
        cv_episodes.append(e)
        cv_moves.append(i)
        cv_score.append(score)
        cv_cumulated_rewards.append(cum_reward)

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

def random_player(env, agent, params, verbose=0):
    # fetch parameters
    state_size = params["state_size"]
    action_size = params["action_size"]
    max_games = params["max_games"]
    model_name = params["model_name"]

    # load weights
    agent.load(model_name)

    # start testing
    done = False

    # Step 1: Initialization
    state = env.reset()
    state = np.reshape(state, [1,state_size])
    action = 0

    # Step 2: Simulate one trial of the game
    for score in range(max_games):

        env.render()

        # use exploit() instead of act()
        action = (action+1) % 2

        if verbose:
            print(action)

        next_state, reward, done, _ = env.step(action)

        state = np.reshape(next_state, [1,state_size])

        if done:
            break

    print("score:{}".format(score))

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