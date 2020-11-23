import os

MODEL_DIR = "models/snake_10"
ep_index = [int(file.split(".")[0]) for file in os.listdir("{}/weights".format(MODEL_DIR))]
ep_index.sort()
