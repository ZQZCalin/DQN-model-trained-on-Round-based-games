import sys, os
from util import *

#---------------------------------------------------------
# import the environment directory
sys.path.append(os.path.abspath('Snake'))
# make sure you add "../name" to the system path
from snakeEnv import *

# environment
ENV = snakeEnv()
STATE_SIZE = ENV.state_size
ACTION_SIZE = ENV.action_size

#---------------------------------------------------------
# DQN hyper-parameters
GAMMA = 0.95

EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

LEARNING_RATE = 0.00025

BATCH_SIZE = 512

# DQN Sequential
# Here we provide a Dense Layer models
LAYER = [128, 128, 128]
ACTIVATION = "relu"
LOSS = "mse"

SEQUENTIAL = dense_NN({"layer":LAYER, "act": ACTIVATION, "loss":LOSS,
    "input":STATE_SIZE, "output":ACTION_SIZE, "lr":LEARNING_RATE})
# SEQUENTIAL = None # uncomment if you want to use the default model

#---------------------------------------------------------
# Environment parameters
# If there are multiple reward function, then define this static
REWARD = "NAIVE"
# REWARD = "DETECT_ENCLOSE"

#---------------------------------------------------------
# training parameters
N_TRAINS = 1000
MAX_MOVES_TRAIN = 500
FPS_TRAIN = 15   # set to 0 to disable render
RPE = 50    # render per RPE episodes

WEIGHT_DIR = "model_weights_corrected_v2"

EXP_REPLAY = True   # turn on experience_replay to train every move instead of every game

# testing parameters
N_TESTS = 10
MAX_MOVES_TEST = 10000
TEST_WEIGHT = "model_weights_corrected_v2/weights_850.hdf5"
FPS_TEST = 10   # set to 0 to disable render

# train / test mode
MODE = "TEST"
# MODE = "TRAIN"

# model directory
# MODEL_DIR = "model/snake_2"