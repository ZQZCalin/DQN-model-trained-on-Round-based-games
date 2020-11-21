from snakeClass import *
from util import *
from game_config import *
from numpy.random import randint
from math import ceil, floor
import numpy as np
import time
from scipy.spatial.distance import euclidean
import pygame
from pygame import Rect

pygame.init()

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

    def __init__(self, params):
        # static
        self.DIRECTION = ["U", "R", "L", "D"]

        # mandatory
        self.gridSize = params["gridSize"]
        self.width = params["width"] * self.gridSize
        self.height = params["height"] * self.gridSize

        # default / optional
        self.collideWall = load_params(params, "collideWall", True)
        self.collideBody = load_params(params, "collideBody", True)
        self.extraWalls = []
        for (x, y) in load_params(params, "extraWalls", []):
            self.extraWalls.append(Rect(x*self.gridSize, y*self.gridSize, self.gridSize, self.gridSize))

        self.FPS = load_params(params, "FPS", 10)

        self.stateType = load_params(params, "stateType", "12bool")
        self.rewardType = load_params(params, "rewardType", "basic")
        basic_reward = {
            "eat": 10, "die": -100, "closer": 1, "away": -1
        }
        self.rewardValues = load_params(params, "rewardValues", basic_reward)

        self.snakeLength = load_params(params, "snakeLength", 1)
        self.manualControl = load_params(params, "manualControl", False)

        # initialize
        self.snake = None
        self.apple = None

        self.done = False
        self.score = 0
        self.best_score = 0

        self.window = pygame.display.set_mode((self.width, self.height))
        self.timer  = pygame.time.Clock()

        # reward attr
        self.last_snake = None
        self.last_apple = None
        self.reward_bool = {
            "eat": False, "closer": False, "away": False
        } 

    def reset(self):

        self.done = False
        self.score = 0

        avoid = [(rect.x, rect.y) for rect in self.extraWalls]
        while True:
            random_x = random.randrange(0, self.width, self.gridSize)
            random_y = random.randrange(0, self.height, self.gridSize)
            if not (random_x, random_y) in avoid:
                break

        self.snake = Snake(
            random_x, random_y, self.snakeLength,
            self.DIRECTION[randint(0, 4)], 
            self.gridSize, self.width, self.height
        )

        self.apple = Apple(self.gridSize, 0, 0, self.width, self.height)
        self.apple.move()

        return self.update_state()

    # =========================================
    # UPDATE RULES
    # =========================================
    def update_state(self):
        if self.stateType == "12bool":
            return self.update_state_12bool()
        
        return

    def update_state_12bool(self):
        # update state from self.snake and self.apple
        # called after self.snake and self.apple are updated
        new_state = np.zeros((12, ))

        # Direction of Snake
        if self.snake.direction == "U":
            new_state[0] = 1
        elif self.snake.direction == "R":
            new_state[1] = 1
        elif self.snake.direction == "D":
            new_state[2] = 1
        elif self.snake.direction == "L":
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
        body_pos = [(rect.x, rect.y) for rect in self.snake.body]
        
        if self.snake.direction != "D" and \
        (self.snake.y <= 0 or \
        (self.snake.x, self.snake.y-self.gridSize) in body_pos):
            # obstacle at north
            new_state[8] = 1
        if self.snake.direction != "L" and \
        (self.snake.x >= self.width - self.gridSize or \
        (self.snake.x+self.gridSize, self.snake.y) in body_pos):
            # obstacle at east
            new_state[9] = 1
        if self.snake.direction != "U" and \
        (self.snake.y >= self.height - self.gridSize or \
        (self.snake.x, self.snake.y+self.gridSize) in body_pos):
            # obstacle at south
            new_state[10] = 1
        if self.snake.direction != "R" and \
        (self.snake.x <= 0 or \
        (self.snake.x-self.gridSize, self.snake.y) in body_pos):
            # obstacle at west
            new_state[11] = 1

        return new_state

    # =========================================
    # REWARD RULES
    # =========================================
    def reward(self):
        if self.rewardType == "basic":
            return self.reward_basic()

        return 0

    def clear_reward_bool(self):
        for key in self.reward_bool.keys():
            self.reward_bool[key] = False

    def reward_basic(self, state, next_state):
        if self.done:
            return self.rewardValues["die"]
        if self.reward_bool["eat"]:
            return self.rewardValues["eat"]
        if self.reward_bool["closer"]:
            return self.rewardValues["closer"]
        if self.reward_bool["away"]:
            return self.rewardValues["away"]
        return 0

    # =========================================
    # STEP FORWARD
    # =========================================
    def step(self, action=None):

        if self.done:
            return

        if (not self.manualControl) and action != None:
            self.snake.changeDirection(action)
            self.last_apple = self.apple.rect 
            self.last_snake = self.snake.head

        # move forward
        self.snake.addHead(collideWall=self.collideWall)
        
        temp_tail = None 
        if self.snake.collideWithWall(self.extraWalls):
            # wall collision
            self.game_over()
        else:
            # delete tail
            temp_tail = this.snake.deleteTail()
        if self.snake.collideWithBody():
            # body collisioni
            self.game_over()
        if self.snake.head.colliderect(self.apple.rect):
            # eat apple
            self.snake.addTail(temp_tail)
            self.move_apple()
            self.score += 1
            if self.score > self.best_score:
                self.best_score = self.score

        # conclude end of round
        if self.manualControl:
            return
        else:
            return self.conclude_round()

    def conclude_round(self):
        # update reward
        self.reward_bool["eat"] = not (self.apple.rect.x == self.last_apple.x \
            and self.apple.rect.y == self.last_apple.y)
        self.last_distance = euclidean(
            [self.last_snake.x, self.last_snake.y],
            [self.last_apple.x, self.last_apple.y]
        )
        self.distance = euclidean(
            [self.snake.head.x, self.snake.head.y],
            [self.apple.rect.x, self.apple.rect.y]
        )
        self.reward_bool["closer"] = distance < last_distance
        self.reward_bool["away"] = distance > last_distance
        
        reward = self.reward()
        self.clear_reward_bool()

        # update state
        next_state = self.update_state()

        return [next_state, reward, self.done, self.score]

    def game_over(self):
        self.done = True

    # =========================================
    # GAME RENDER
    # =========================================
    def render(self, FPS=self.FPS):
        # this line prevents pygame from being recognized as "crashed" by OS
        pygame.event.pump()

        self.window.fill(WHITE)

        #  draw the grid
        for x in range(0, self.width, self.gridSize):
            pygame.draw.line(self.window, GRAY, (x, 0), (x, self.height))
        for y in range(0, self.height, self.gridSize):
            pygame.draw.line(self.window, GRAY, (0, y), (self.width, y))

        #  draw the apple
        pygame.draw.rect(self.window, RED, self.apple.rect)

        #  draw the snake
        for part in self.snake.body:
            pygame.draw.rect(self.window, GREEN, part)
            part_small = part.inflate(-3, -3)
            pygame.draw.rect(self.window, WHITE, part_small, 3)

        # draw extra walls
        for part in self.extraWalls:
            pygame.draw.rect(self.window, GRAY, part)
            
        #  draw the score
        scoreFont = pygame.font.Font('freesansbold.ttf', 18)
        fontSurface = scoreFont.render("Score: %d" % self.score, True, BLACK)
        maxScore = scoreFont.render("Best Score: %d" % self.best_score, True, BLACK)
        self.window.blit(fontSurface, (20, 10))
        self.window.blit(maxScore, (20, 30))

        pygame.display.update()

        # adjust speed to FPS
        self.timer.tick(FPS)

    # =========================================
    # UTILS
    # =========================================
    def move_apple(self):
        avoid = [(rect.x, rect.y) for rect in self.snake.body]\
            + [(rect.x, rect.y) for rect in self.extraWalls]
        self.apple.move(avoid)

if __name__ == "__main__":
    params = {
        "gridSize": 20,
        "width": 20,
        "height": 20
    }
    env = snakeEnv(params)
    env.reset()
    env.render()