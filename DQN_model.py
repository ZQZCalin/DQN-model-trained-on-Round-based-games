### Import Packages

import sys, os, time 
import importlib
import random
# import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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
        model.add(Dense(self.action_size, activation="softmax"))

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
        mini_batch = random.sample(self.memory, batch_size)

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
    model_dir = MODEL_DIR
    output_dir = TRAIN_WEIGHT
    fps = FPS_TRAIN
    # state_size = params["state_size"]
    # action_size = params["action_size"]
    # batch_size = params["batch_size"]
    # n_episodes = params["n_episodes"]
    # max_moves = params["max_moves"]
    # output_dir = params["output_dir"]

    # create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        overwrite = input("Are you sure to overwrite the existing weights? y or n")
        if overwrite == "n":
            return

    done = 0

    for e in range(n_episodes):

        # Step 1: Initialization
        state = env.reset()
        state = np.reshape(state, [1,state_size])

        # Step 2: Simulate one trial of the game
        steps = []
        for _ in range(max_moves):
            # visualize the game
            if fps != 0:
                env.render(FPS=fps)

            # simulate action and outcomes
            action = agent.act(state)

            next_state, reward, done, score = env.step(action)

            # reward = reward if not done else -10    # penalize game over action

            next_state = np.reshape(next_state, [1,state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        # print the training result
        print("progress: {}/{}, score: {}, e: {:.2}".format(e,n_episodes,score,agent.epsilon))

        # Step 3: Train DQN based on the agent's memory
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # agent.replay(min(len(agent.memory), batch_size))

        # save model weight every 50 episodes
        if e % 50 == 0:
            agent.save(output_dir + "/weights_" + "{:.0f}".format(e) + ".hdf5")

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

    # create agent
    AGENT = Agent(STATE_SIZE, ACTION_SIZE, model=SEQUENTIAL)

    MODEL_DIR = "models/" + model_dir
    TRAIN_WEIGHT = MODEL_DIR + "/model_weight"
    TEST_WEIGHT = MODEL_DIR + "/" + TEST_WEIGHT

    # train and test
    if MODE == "TRAIN":
        train_DQN(ENV, AGENT)
    if MODE == "TEST":
        test_DQN(ENV, AGENT)

# ARCHIVED
if __name__ == "__main__" and False:

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