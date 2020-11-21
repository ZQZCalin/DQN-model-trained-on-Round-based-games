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
    agent_model = dense_NN({
        "layer":LAYER, "act": ACTIVATION, "act_out": OUTPUT_ACT,
        "loss":LOSS, "input":STATE_SIZE, "output":ACTION_SIZE, "lr":LEARNING_RATE
    })
    agent = Agent(STATE_SIZE, ACTION_SIZE, model=agent_model, params=agent_params)

    if MODE == "TRAIN":
        # train params
        train_params = {
            "state_size": STATE_SIZE, "action_size": ACTION_SIZE,
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

        # check agent model directory
        path_model = "{}/model".format(MODEL_DIR)
        if os.path.exists(path_model):
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