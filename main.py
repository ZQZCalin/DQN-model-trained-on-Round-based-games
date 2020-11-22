import os, sys, time
from config import *
from util import *
from snakeEnv import *
from agent import *
from DQN_model import *

if __name__ == "__main__":
    # env and agent
    env_params = {
        "gridSize": GRIDSIZE, "width": WIDTH, "height": HEIGHT,
        "collideWall": COLLIDE_WALL, "collideBody": COLLIDE_BODY,
        "extraWalls": EXTRA_WALLS,
        "stateType": STATE_TYPE, "rewardType": REWARD_TYPE, "rewardValues": REWARD_VALUES,
        "snakeLength": SNAKE_LENGTH, "manualControl": MANUAL_CONTROL
    }
    env = snakeEnv(env_params)

    agent_params = {
        "gamma": GAMMA, "epsilon": EPSILON, "e_decay": EPSILON_DECAY, 
        "e_min": EPSILON_MIN, "lr": LEARNING_RATE
    }
    snake_model_Dense = dense_NN({
        "layer":LAYER, "act": ACTIVATION, "act_out": OUTPUT_ACT,
        "loss": LOSS, "input": STATE_SIZE, "output": ACTION_SIZE, "lr": LEARNING_RATE
    })
    snake_model_CNN = snake_CNN({
        "input_shape": (WIDTH, HEIGHT, 1), "output": ACTION_SIZE,
        "layers": [32, 64, 64], "pool_size": (2, 2),
        "activation": "relu", "act_last": "linear",
        "loss": "mse", "lr": 0.025
    })
    agent_model = snake_model_CNN
    agent = Agent(STATE_SIZE, ACTION_SIZE, model=agent_model, params=agent_params)

    if MODE == "TRAIN":
        # train params
        train_params = {
            "batch_size": BATCH_SIZE, "n_episodes": N_TRAINS,
            "max_moves": MAX_MOVES_TRAIN, "FPS": FPS_TRAIN,
            "exp_replay": EXPERIENCE_REPLAY,
            "model_dir": MODEL_DIR, "save_per_episode": Save_Per_Episode
        }

        # create model directory
        if not check_dir(MODEL_DIR, create=True):
            print("Created model directory as: {}".format(MODEL_DIR))

        # save config as documentation
        path_config = "{}/config.txt".format(MODEL_DIR)
        if not save_config_py(path_config):
            sys.exit()
        else:
            print("Config is saved as: {}".format(path_config))

        # check agent model
        if tf.__version__[0] == "2":
            # tf 2 save model
            path_model = "{}/model".format(MODEL_DIR)
            if os.path.exists(path_model):
                text = "Are you sure to overwrite the existing model?"
                if not yes_no(text):
                    sys.exit()
            else:
                agent.save_model(path_model)
        elif tf.__version__[0] == "1":
            # tf 1 save model
            path_model = "{}/model.h5".format(MODEL_DIR)
            if os.path.isfile(path_model):
                text = "Are you sure to overwrite the existing model?"
                if not yes_no(text):
                    sys.exit()
            else:
                agent.save_model(path_model)
        
        # check weight directory
        path_weight = "{}/weights".format(MODEL_DIR)
        if not os.path.exists(path_weight):
            os.mkdir(path_weight)
            print("Created weight directory as: {}".format(path_weight))
        else:
            text = "Are you sure to overwrite the existing weights?"
            if not yes_no(text):
                sys.exit()

        # check performance file
        path_performance = "{}/performance".format(MODEL_DIR)
        if not os.path.exists(path_performance):
            os.mkdir(path_performance)
            print("Created weight directory as: {}".format(path_performance))
        else:
            text = "Are you sure to overwrite the existing performance?"
            if not yes_no(text):
                sys.exit()

        # train model
        train_DQN(env, agent, train_params)
    
    # test
    if MODE == "TEST":
        # load directory
        if not check_dir(MODEL_DIR, create=False):
            print("Your model directory does not exist.")
            sys.exit()
        
        # load model
        if tf.__version__[0] == "1":
            path_model = "{}/model.h5".format(MODEL_DIR)
            if os.path.isfile(path_model):
                agent.load_model(path_model)
            else:
                print("Model does not exist.")
                sys.exit()

        path_weight = "{}/{}".format(MODEL_DIR, TEST_WEIGHT)
        agent.load_weight(path_weight)

        # test
        test_params = {
            "n_tests": N_TESTS, "max_moves": MAX_MOVES_TEST,
            "FPS": FPS_TEST
        }

        test_DQN(env, agent, test_params)