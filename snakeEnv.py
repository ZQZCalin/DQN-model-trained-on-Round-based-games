from snakeClass import *
from snakeGame import *
from numpy.random import randint
from math import ceil, floor
import numpy as np
import yaml


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

def pos_to_pixel(pos):
    # convert position to pixel position
    return floor(pos/GRIDSIZE)*GRIDSIZE

STATE_SIZE = 12
ACTION_SIZE = 4

class snakeEnv():
    """
    STATIC:
        - action:
            - up: 0, right: 1, down: 2, left: 3
    Attributes:
        - state
            - Direction, Apple (wrt snake head), Obstacle
                0) up: 0, 1
                1) right: 0, 1
                2) down: 0, 1
                3) left: 0, 1
        - done
            - check if game ends: 0, 1
        - reward(state, next_state):
            - return reward of a certain action:
                - eat apple: 10
                - closer to apple: 1
                - away from apple: -1
                - game-over: -100
                # Additional rewards
                - 

    Environment methods:
        - reset()
            - Random initialization: initialize Snake and Apple()
            - return: initial_state
        - step(action)
            - Given an action a, proceed to the next state
            - return: next_state, reward, done, _
        - render()
            - Render the image of current state
            - Pause every t seconds
    
    """

    def __init__(self):
        self.state = np.zeros((STATE_SIZE, ))
        self.snake = None
        self.apple = None
        self.done = 0

    def reset(self):

        self.snake = Snake(
                pos_to_pixel(randint(GRIDSIZE, WIDTH-GRIDSIZE)), 
                pos_to_pixel(randint(GRIDSIZE, HEIGHT-GRIDSIZE)), 
                SNAKELENGTH, 
                DIRECTION[randint(0, ACTION_SIZE)], 
                SNAKECOLOR, GRIDSIZE)

        self.apple = Apple(
                GRIDSIZE, 
                pos_to_pixel(randint(0, WIDTH)), 
                pos_to_pixel(randint(0, HEIGHT)),
                RED, WIDTH, HEIGHT)

        self.update_state()

        return self.state 

    def update_state(self):
        # update state from self.snake and self.apple
        # called after self.snake and self.apple are updated
        new_state = np.zeros((STATE_SIZE, ))

        # Direction of Snake
        if self.snake.direction == "U":
            new_state[0] = 1
        if self.snake.direction == "R":
            new_state[1] = 1
        if self.snake.direction == "D":
            new_state[2] = 1
        if self.snake.direction == "L":
            new_state[3] = 1

        # Apple position (wrt snake head)
        # (0,0) at Top-Left Corner: U: -y; R: +x
        if self.apple.y < self.snake.y:
            # apple north snake
            new_state[4] = 1
        if self.apple.x > self.snake.x:
            # apple east snake
            new_state[5] = 1
        if self.apple.y > self.snake.y:
            # apple south snake
            new_state[6] = 1
        if self.apple.x < self.snake.x:
            # apple west snake
            new_state[7] = 1
        
        # Obstacle (Walls, body) position (wrt snake head)
        body_x = [rect.x for rect in self.snake.body]
        body_y = [rect.y for rect in self.snake.body]
        if self.snake.direction != "D" and \
        (self.snake.y == 0 or pos_to_pixel(self.snake.y-1) in body_y):
            # obstacle at north
            new_state[8] = 1
        if self.snake.direction != "L" and \
        (self.snake.x == pos_to_pixel(WIDTH-1) or pos_to_pixel(self.snake.x+1) in body_x):
            # obstacle at east
            new_state[9] = 1
        if self.snake.direction != "U" and \
        (self.snake.y == pos_to_pixel(HEIGHT-1) or pos_to_pixel(self.snake.y-1) in body_y):
            # obstacle at south
            new_state[10] = 1
        if self.snake.direction != "R" and \
        (self.snake.x == 0 or pos_to_pixel(self.snake.x-1) in body_x):
            # obstacle at west
            new_state[11] = 1

        self.state = new_state

    def step(self, action):

        current_state = self.state
        ...
        next_state = self.state
        reward_ = reward(current_state, next_state)
        return current_state, reward_, next_state

# Main

env = snakeEnv()
print("hello")