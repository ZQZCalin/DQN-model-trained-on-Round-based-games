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
# Environment Parameters (For SNAKE Env)
GRIDSIZE = 20
WIDTH  = 20
HEIGHT = 20

COLLIDE_WALL = True
COLLIDE_BODY = True
EXTRA_WALLS  = [(x,10) for x in range(6,15)] + [(x,11) for x in range(6,15)] + [(x,11) for x in range(6,15)]
# EXTRA_WALLS = []

REWARD_TYPE = "basic"
REWARD_VALUES = {
    "eat": 10, "die": -100, "closer": 1, "away": -1
}

SNAKE_LENGTH = 1
MANUAL_CONTROL = False

STATE_SIZE = 12
ACTION_SIZE = 4

# change state type to match the DQN model
    # - "12bool": vector of 12 booleans for Dense models
    # - "CNN": (w x h x 1) matrix for CNN models
STATE_TYPE = "12bool"

#---------------------------------------------------------
# Agent Parameters

GAMMA = 0.95

EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

LEARNING_RATE = 0.00025

# DQN Sequential
# Here we provide a Dense Layer models
LAYER = [128, 128, 128]
ACTIVATION = "relu"
OUTPUT_ACT = "linear"
LOSS = "mse"
DROPOUT_RATE = 0.1
# If you choose to define your own model,
# please go to main.py


#---------------------------------------------------------
# Mode 0: Train Parameters

BATCH_SIZE = 512

N_TRAINS = 100
MAX_MOVES_TRAIN = 1000
#   set FPS to 0 to disable render
FPS_TRAIN = 15
#   turn on experience_replay to train every move instead of every game
EXPERIENCE_REPLAY = True
#   render per RPE episodes
Render_Per_Episode = 1
#   save (weight, model, performance) every WPE episodes
Save_Per_Episode = 1


#---------------------------------------------------------
# Mode 1: Testing Parameters

N_TESTS = 3
MAX_MOVES_TEST = 1000
TEST_WEIGHT = "100.hdf5"

#   set to 0 to disable render
FPS_TEST = 0


#---------------------------------------------------------
# Mode 2: Continue Training (Currently UNAVAILABLE)

N_LAST = 50     # latest episode
N_THIS = 100    # episode you want to train this time


#---------------------------------------------------------
# Mode 3: Test Entire Model
# we assume that the weights are named by "e.hdf5", e integer

N_TEST_REPEAT = 10   # number of repeats in each epoch
MAX_MOVES_TEST_ALL = 1000

TEST_ALL_DIR = "test_all_result"

#---------------------------------------------------------
# CHOOSE MODE AND DIRECTORY BEFORE STARTING

# train / test mode
MODE_LIST = ["TRAIN", "TEST", "CONTINUE", "TEST_ALL"]
MODE = MODE_LIST[3]

# model directory, e.g., "models/my_model_1"
MODEL_DIR = "models/snake_7"

#---------------------------------------------------------
# Please add additional notes below
"""
Dense (128, 128, 128) with Dropout = 0.3
State: basic 12 booleans
Reward: basic
Model: Dense
Loss: MSE
Experience replay: On
"""