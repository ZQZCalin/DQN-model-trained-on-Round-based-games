# DQN-model-trained-on-Round-based-games

### User Instruction
###### Training
To train the model, users need to follow the instructions.
1. Provide an environment class, which has the following methods:
    1. reset(): reset the game and return a state (the state must match the input of your DQN model);
    2. step(action): given an action, move the game one step forward and return a list [next_state, reward, done, score]
    3. render(FPS): render the game every FPS seconds.
2. In "main.py", create a new environment class.
3. In "config.py", update the hyperparameters of the agent class. In "main.py", create a DQN model for the agent. In "main.py", create a new agent class.
4. In "config.py", modify training and testing parameters; add additional remarks of the model.
5. Run "main.py". The new file will be saved to the specified "MODEL_DIR", and will contain: the DQN model, the weights, the performance table, and the configuration.

###### Testing Single Model
To test the model you trained, change the MODE in "config.py" to 1 ("TEST") and modify the testing parameters. Then run "main.py".

###### Testing Entire Model (Currently Unavailable)