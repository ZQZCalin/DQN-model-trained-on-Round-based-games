import sys, os
from util import *

"""
In this document, you need to fill in the parameters of your model.
- If you choose to train: 
    - this config will be saved to "config.txt"
    - env and agent will be saved to "model.pkl"

- If you choose to test, 
    - the program will test on the test_weight, and 
      the previously saved (env, agent)
"""

#---------------------------------------------------------
# Environment parameters

# import the environment directory
sys.path.append(os.path.abspath('Snake'))
# make sure you add "../name" to the system path
from snakeEnv import *

# environment
ENV = snakeEnv()
STATE_SIZE = ENV.state_size
ACTION_SIZE = ENV.action_size


#---------------------------------------------------------
# Agent Parameters

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
OUTPUT_ACT = "linear"
LOSS = "mse"

SEQUENTIAL = dense_NN({"layer":LAYER, "act": ACTIVATION, "act_out": OUTPUT_ACT,
    "loss":LOSS, "input":STATE_SIZE, "output":ACTION_SIZE, "lr":LEARNING_RATE})
# SEQUENTIAL = None # uncomment if you want to use the default model


#---------------------------------------------------------
# Train parameters

N_TRAINS = 50
MAX_MOVES_TRAIN = 10000
#   set FPS to 0 to disable render
FPS_TRAIN = 15
#   turn on experience_replay to train every move instead of every game
EXPERIENCE_REPLAY = True
#   penalize death
PENALIZE_DEATH = False
#   render per RPE episodes
Render_Per_Episode = 1
#   document path
WEIGHT_DIR = "model_weights"
ENV_AG_DIR = "env_agent_backup"
PERFORMANCE_DIR = "performance_backup"
#   save (weight, model, performance) every WPE episodes
Weight_Per_Episode = 5
Model_Per_Episode = 5
Performance_Per_Episode = 5

#---------------------------------------------------------
# testing parameters
N_TESTS = 10
MAX_MOVES_TEST = 1000
TEST_WEIGHT = "model_weights/weights_70.hdf5"

#   set to 0 to disable render
FPS_TEST = 10

#---------------------------------------------------------
# continue training
"""
By using this option, you cannot change any training parameter.
You must use the same parameter as before!
You must also keep your latest weight, model, and performance directory.
"""
N_LAST = 50     # latest episode
N_THIS = 100    # episode you want to train this time

#---------------------------------------------------------
# STARTING PARAMETERS

# train / test mode
MODE_LIST = ["TRAIN", "TEST", "CONTINUE"]
MODE = MODE_LIST[1]

# model directory
MODEL_DIR = "models/snake_6"

#---------------------------------------------------------
# Config parameters

NAME = "SNAKE"

# Add additional notes
# Please document the features of the model that cannot be recorded by values
# You can also go to "config.txt" to add this manually
NOTES = "Naive reward\nDense [128,128,128] with relu\n" + \
    "Output with linear\nLoss: mse\nExperience replay: On"