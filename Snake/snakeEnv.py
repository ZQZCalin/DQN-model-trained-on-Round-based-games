from snakeClass import *
from snakeGame import *
from util import *
from numpy.random import randint
from math import ceil, floor
import numpy as np
import time
from scipy.spatial.distance import euclidean

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
        # mandatory
        self.gridSize = params["gridSize"]
        self.width = paramss["width"]
        self.height = paramss["height"]

        self.extraWalls = generate_walls()

        # default / optional
        self.collideWall = load_params("collideWall")
        self.collideBody = load_params("collideBody")
        self.extraWalls = load_params("extraWalls")

        self.FPS = load_params("FPS")

        self.stateType = load_params("stateType")
        self.rewardType = load_params("rewardType")
        self.rewardValues = load_params("rewardValues")

        self.snakeLength = load_params("snakeLength")
        self.manualControl = load_params("manualControl")

        self.snake = None
        self.apple = None
        self.done = 0
        self.score = 0

        # self.state_size = STATE_SIZE
        # self.action_size = ACTION_SIZE
        self.state_size = 12
        self.action_size = 4

    def reset(self):

        self.done = 0
        self.score = 0

        self.snake = Snake(
                pos_to_pixel(randint(5*GRIDSIZE, WIDTH-5*GRIDSIZE)), 
                pos_to_pixel(randint(5*GRIDSIZE, HEIGHT-5*GRIDSIZE)), 
                SNAKELENGTH, 
                DIRECTION[randint(0, self.action_size)], 
                SNAKECOLOR, GRIDSIZE)

        self.apple = Apple(
                GRIDSIZE, 
                pos_to_pixel(randint(0, WIDTH)), 
                pos_to_pixel(randint(0, HEIGHT)),
                RED, WIDTH, HEIGHT)

        return self.update_state()

    # =========================================
    # UPDATE RULES
    # =========================================
    def update_state(self):
        if this.stateType == "12bool":
            return this.update_state_12bool()
        
        return

    def update_state_12bool(self):
        # update state from self.snake and self.apple
        # called after self.snake and self.apple are updated
        new_state = np.zeros((self.state_size, ))

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
        body_pos = [(rect.x, rect.y) for rect in self.snake.body]
        if self.snake.direction != "D" and \
        (self.snake.y <= 0 or (self.snake.x, self.snake.y-GRIDSIZE) in body_pos):
            # obstacle at north
            new_state[8] = 1
        if self.snake.direction != "L" and \
        (self.snake.x >= WIDTH-GRIDSIZE or (self.snake.x+GRIDSIZE, self.snake.y) in body_pos):
            # obstacle at east
            new_state[9] = 1
        if self.snake.direction != "U" and \
        (self.snake.y >= HEIGHT-GRIDSIZE or (self.snake.x, self.snake.y+GRIDSIZE) in body_pos):
            # obstacle at south
            new_state[10] = 1
        if self.snake.direction != "R" and \
        (self.snake.x <= 0 or (self.snake.x-GRIDSIZE, self.snake.y) in body_pos):
            # obstacle at west
            new_state[11] = 1

        self.state = new_state
        return self.state

    def update_score(self):
        self.score = self.snake.body.__len__() - SNAKELENGTH
        return self.score

    # =========================================
    # REWARD RULES
    # =========================================
    def reward(self, state, next_state):
        # test reward function 1:
        # score of apple v.s. snake, lower score means closer
        s1 = np.sum(state[4:8])
        s2 = np.sum(next_state[4:8])

        if s1 < s2:
            # far away from apple
            return -1
        if s1 > s2:
            # closer to apple
            return 1

        # other cases
        return 0

    # =========================================
    # STEP FORWARD
    # =========================================
    def step(self, action):
        """ 
        about reward:
        Two base rewards:
        - die: -100
        - get apple: 10
        Additional rewards:
        - NAIVE:
            - closer to apple: 1
            - away from apple: -1
        - DETECT_ENCLOSE:
            - 
        """

        # BE VERY CAREFUL ABOUT THE POSITION & POINTER ISSUES !!!!!

        if self.done:
            return

        current_state = self.state
        current_snake = self.snake
        pos_current = [current_snake.x, current_snake.y]

        reward_ = 0

        # update direction
        # opposite_direction = (DIRECTION_INVERSE[self.snake.direction] + 2) % 4
        if action != (DIRECTION_INVERSE[self.snake.direction] + 2) % 4:
            self.snake.direction = DIRECTION[action]
        
        # update game/motion, done, and reward_
        self.snake.addHead()
        if self.snake.isDead() or self.snake.isOutOfBounds(WIDTH, HEIGHT):
            # game-over
            self.done = 1
            reward_ = -100
        if not(self.snake.head.colliderect(self.apple.rect)):
            # not eat apple
            self.snake.deleteTail()
        else:
            # eat apple
            avoid = [(rect.x, rect.y) for rect in self.snake.body]
            self.apple.move(avoid=avoid)
            reward_ = 10

        next_state = self.update_state()

        if reward_ == 0:
            pos_next = [self.snake.x, self.snake.y]
            pos_apple = [self.apple.x, self.apple.y]
            d1 = euclidean(pos_apple, pos_current)
            d2 = euclidean(pos_apple, pos_next)

            if d1 > d2:
                reward_ = 1
            else:
                reward_ = -1

        return next_state, reward_, self.done, self.update_score()

    # =========================================
    # GAME RENDER
    # =========================================
    def render(self, FPS=15):
        # this line prevents pygame from being recognized as "crashed" by OS
        pygame.event.pump()

        WINDOW.fill(BLACK)

        #  draw the grid
        for x in range(0, WIDTH, GRIDSIZE):
            pygame.draw.line(WINDOW, GRAY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRIDSIZE):
            pygame.draw.line(WINDOW, GRAY, (0, y), (WIDTH, y))

        #  draw the apple
        pygame.draw.rect(WINDOW, self.apple.color, self.apple.rect)

        #  draw the snake
        for part in self.snake.body:
            pygame.draw.rect(WINDOW, self.snake.color, part)
            part_small = part.inflate(-3, -3)
            pygame.draw.rect(WINDOW, WHITE, part_small, 3)
            
        #  draw the score
        scoreFont = pygame.font.Font('freesansbold.ttf', 18)
        fontSurface = scoreFont.render("Score: %d" % self.score, True, WHITE)
        WINDOW.blit(fontSurface, (WIDTH - 100, 10))

        pygame.display.update()

        # adjust speed to FPS
        fpsClock.tick(FPS)


if __name__ == "__main__":
    env = snakeEnv()