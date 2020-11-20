import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from util import *

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
        - e_decay: decrease rate of epsilon
        - epsilon_min: minimum threshold of epsilon
        - learning_rate: learning rate
    """
    def __init__(self, state_size, action_size, model=None, params={}):
        # mandatory
        self.state_size = state_size
        self.action_size = action_size

        # default
        self.gamma = load_params(params, "gamma", 0.95)

        self.epsilon = load_params(params, "epsilon", 1)
        self.e_decay = load_params(params, "e_decay", 0.995)
        self.e_min = load_params(params, "e_min", 0.01)

        self.lr = load_params(params, "lr", 0.01)

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
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))

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
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load_weight(self, name):
        """
        load model weights
        """
        self.model.load_weights(name)

    def save_weight(self, name):
        """
        save model weights
        """
        self.model.save_weights(name)

    def load_model(self, name):
        # load entire model
        self.model = load_model(name)

    def save_model(self, name):
        # save entire model
        self.model.save(name)

if __name__ == "__main__":
    agent = Agent(12, 4) 