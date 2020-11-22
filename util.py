from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
import pickle
import sys, os
import pandas as pd
from shutil import copyfile

def dense_NN(dict):

    model = Sequential()

    # Layer Input
    model.add(Dense(dict["layer"][0], input_dim=dict["input"], activation=dict["act"]))
    # Layer 1 to n
    for i in range(1, len(dict["layer"])):
        model.add(Dense(dict["layer"][i], activation=dict["act"]))
    # Layer Output
    model.add(Dense(dict["output"], activation=dict["act_out"]))

    model.compile(loss=dict["loss"], optimizer=Adam(dict["lr"]))

    return model

def snake_CNN(params):

    model = Sequential()
    # Conv + MaxPooling
    layers = params["layers"]
    model.add(Conv2D(
        layers[0], (3, 3), activation=params["activation"], 
        input_shape=params["input_shape"]
    ))
    for i in range(1, len(layers)):
        model.add(MaxPooling2D(params["pool_size"]))
        model.add(Conv2D(
            layers[i], (3, 3), activation=params["activation"]
        ))
    # Flattern
    model.add(Flatten())
    # Dense
    model.add(Dense(params["output"], activation=params["act_last"]))
    # Compile
    model.compile(loss=params["loss"], optimizer=Adam(params["lr"]))

    return model 

def yes_no(text):
    while True:
        boo = input("{}\n(y)/(n):----".format(text))
        if boo == "y":
            return True 
        if boo == "n":
            return False

def save_env_agent(env, agent, file):

    if not os.path.isfile(file):
        f = open(file, "x")
        f.close()
    else:
        text = "Are you sure to overwrite env and agent?\n" + \
            "(Strongly not recommended if you have already trained your model)"
        if not yes_no(text):
            return False
    
    with open(file, "wb") as f:
        pickle.dump([env, agent], f)
    
    return True

def load_env_agent(file):

    if not os.path.isfile(file):
        return

    with open(file, "rb") as f:
        env, agent = pickle.load(f)

    return env, agent

def check_dir(dir, create=False):

    if os.path.isdir(dir):
        return True
    else:
        if create:
            os.mkdir(dir)
        return False

def save_config(config, file):

    if not os.path.isfile(file):
        f = open(file, "x")
        f.close()
    else:
        text = "Are you sure to overwrite config?\n" + \
            "(Strongly not recommended. If you choose to train a new model," + \
            "please use another directory)"
        if not yes_no(text):
            return False
    
    with open(file, "w") as f:
        for key in config.keys():
            if key == "notes":
                f.write("-"*50+"\n")
                f.write("Additional notes:\n"+config[key])
            else:
                f.write("{} = {}\n".format(key, str(config[key])))

    return True

def save_config_py(file):
    if os.path.isfile(file):
        text = "Are you sure to overwrite the existing config.txt file?\n" + \
            "(Strongly not recommended. If you choose to train a new model," + \
            "please use another directory)"
        if not yes_no(text):
            return False
    if not os.path.isfile("config.py"):
        print("config.py does not exist.")
        return False 
    
    copyfile("config.py", file)
    return True

def continue_training():
    text = "Do you want to continue training? (Strongly not recommended)"
    return yes_no(text)

def save_to_DataFrame(df, list, columns):
    # dict length and list length must match
    temp_df = pd.DataFrame([list], columns=columns)
    return None

def load_params(params, key, default):
    try:
        return params[str(key)]
    except:
        return default 