# DQN-model-trained-on-Round-based-games

### User Instruction
#### Training
To train the model, users need to follow the instructions.
1. Provide an environment class, which has the following methods:
    - reset(): reset the game and return a state (the state must match the input of your DQN model);
    - step(action): given an action, move the game one step forward and return a list [next_state, reward, done, score]
    - render(FPS): render the game every FPS seconds.
2. In [main.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/main.py), create a new environment class.
3. In [config.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/config.py), update the hyperparameters of the agent class. In [main.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/main.py), create a DQN model for the agent. In [main.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/main.py), create a new agent class.
4. In [config.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/config.py), modify training and testing parameters; add additional remarks of the model. See [HERE](#hyperparameters) for additional explanations of each parameter.
5. Run [main.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/main.py). The new file will be saved to the specified "MODEL_DIR", and will contain: the DQN model, the weights, the performance table, and the configuration.

#### Testing Single Model
To test the model you trained, change the MODE in [config.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/config.py) to 1 ("TEST") and modify the testing parameters. Then run [main.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/main.py).

#### Testing Entire Model (Currently Unavailable)

---

#### Hyperparameters
This section documents each parameter in [config.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/config.py).
1. Environment Parameters:
2. Agent Parameters:
    - GAMMA: coefficient of future state, default = 0.95
    - EPSILON: initial exploitation - exploration coefficient (explore if e=1; exploit if e=0), default = 1
    - EPSILON_DECAY: decay rate of epsilon in each round, default = 0.995
    - EPSILON_MIN: minimum value of epsilon, default = 0.01
    - LEARNING_RATE: learning rate of the model, default = 0.00025
    - MODEL: we provide a default Dense network
        - LAYER: a list of integers of the layer size, e.g., [128, 128, 128]
        - ACTIVATION: activation function of intermediate Dense layers, e.g., "relu"
        - OUTPUT_ACT: activation function of the output layer, e.g., linear
        - LOSS: loss function; for Snake game, "mse" is recommended \
    If you choose to use a customized model, you should go to [main.py](https://github.com/ZQZCalin/DQN-model-trained-on-Round-based-games/blob/main/main.py) and define the model by yourself.
3. Training Parameters:
    - BATCH_SIZE: batch_size, default = 64; for Snake game, 512 is recommended
    - N_TRAINS: number of training epoches (games), default = 100
    - MAX_MOVES_TRAIN: maximum number of moves in each epoch, default = 1000
    - FPS_TRAIN: FPS of game rendering, default = 10; set to 0 to disable rendering
    - EXPERIENCE_REPLAY: default = True; if turned on, the agent will train every move instead of every game
    - Save_Per_Episode: save weights and performance every SPE epoches, default = 1
4. Testing Parameters (Single Model):
    - N_TESTS: number of tests, default = 10
    - MAX_MOVES_TEST: max moves in each game, default = 1000
    - TEST_WEIGHT: file name of the model weight stored in "weights/" directory, e.g. "50.hdf5"
    - FPS_TEST: default = 10; set to 0 to disable rendering
5. Continue Training Parameters: (Currently Unavailable)
6. Testing Entire Model Parameters: (Currently Unavailable)
5. Others: 
    - MODE: choose the training mode
        - 0 / "TRAIN": train the model
        - 1 / "TEST": test a single model
        - 2 / "CONTINUE": continue training (currently unavailable)
        - 3 / "TEST_ALL": test the entire model
    - MODEL_DIR: the directory of your saved file, e.g., "models/my_model_1"
    - Finally, please add additional remarks to briefly describe your model.
