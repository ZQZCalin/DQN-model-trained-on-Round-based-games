from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def dense_NN(dict):

    model = Sequential()

    model.add(Dense(dict["layer"][0], input_dim=dict["input"], activation=dict["act"]))

    for i in range(1, len(dict["layer"])):
        model.add(Dense(dict["layer"][i], activation=dict["act"]))

    model.add(Dense(dict["output"], activation="softmax"))

    model.compile(loss=dict["loss"], optimizer=Adam(dict["lr"]))

    return model