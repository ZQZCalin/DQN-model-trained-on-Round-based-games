from pygame import Rect
import random


class Snake:
    """Snake object for snake game.
    Attributes:
    - length
    - x
    - y
    - color
    - direction
    - boxSize
    - body"""

    def __init__(self, x, y, length, direction, color, boxSize):
        self.x = x
        self.y = y
        self.length = length
        self.direction = direction
        self.color = color
        self.boxSize = boxSize
        self.body = []

        k1 = 0
        k2 = 0
        if self.direction == "R":
            k1 = -1
        if self.direction == "L":
            k1 = 1
        if self.direction == "U":
            k2 = 1
        if self.direction == "D":
            k2 = -1

        for i in range(self.length):
            tempRect = Rect(self.x + k1*i * self.boxSize,
                            self.y + k2*i * self.boxSize, 
                            self.boxSize, self.boxSize)
            self.body.append(tempRect)
        self.head = self.body[0]

    def addHead(self):
        if self.direction == 'R':
            newHead = Rect(self.x + self.boxSize,
                           self.y, self.boxSize, self.boxSize)
        elif self.direction == 'L':
            newHead = Rect(self.x - self.boxSize,
                           self.y, self.boxSize, self.boxSize)
        elif self.direction == 'D':
            newHead = Rect(self.x,
                           self.y + self.boxSize, self.boxSize, self.boxSize)
        elif self.direction == 'U':
            newHead = Rect(self.x,
                           self.y - self.boxSize, self.boxSize, self.boxSize)
        self.body.insert(0, newHead)
        self.head = self.body[0]
        self.x = self.head.x
        self.y = self.head.y

    def deleteTail(self):
        del self.body[-1]

    def isDead(self):
        for part in self.body[1:]:
            if self.head.colliderect(part):
                return True
        return False

    def isOutOfBounds(self, max_width, max_height):
        if self.head.x > max_width - self.boxSize:
            return True
        elif self.head.x < 0:
            return True
        if self.head.y > max_height - self.boxSize:
            return True
        elif self.head.y < 0:
            return True
        return False

    def addHead_snakeTwo(self, max_width, max_height):
        if self.direction == 'R':
            newHead = Rect(self.x + self.boxSize,
                           self.y, self.boxSize, self.boxSize)
        elif self.direction == 'L':
            newHead = Rect(self.x - self.boxSize,
                           self.y, self.boxSize, self.boxSize)
        elif self.direction == 'D':
            newHead = Rect(self.x,
                           self.y + self.boxSize, self.boxSize, self.boxSize)
        elif self.direction == 'U':
            newHead = Rect(self.x,
                           self.y - self.boxSize, self.boxSize, self.boxSize)
        if newHead.x > max_width - self.boxSize:
            newHead = Rect(0, self.y, self.boxSize, self.boxSize)
        elif newHead.x < 0:
            newHead = Rect(max_width - self.boxSize, self.y,
                           self.boxSize, self.boxSize)
        if newHead.y > max_height - self.boxSize:
            newHead = Rect(self.x, 0, self.boxSize, self.boxSize)
        elif newHead.y < 0:
            newHead = Rect(self.x, max_height - self.boxSize,
                           self.boxSize, self.boxSize)
        self.body.insert(0, newHead)
        self.head = self.body[0]
        self.x = self.head.x
        self.y = self.head.y


class Apple:
    """Apple Object for the snake game.
    Attributes:
    - boxLength
    - x
    - y"""

    def __init__(self, boxLength, x, y, color, board_x, board_y):
        self.boxLength = boxLength
        self.x = x
        self.y = y
        self.color = color
        self.board_x = board_x
        self.board_y = board_y
        self.rect = Rect(self.x, self.y, self.boxLength, self.boxLength)

    def move(self, avoid=None):
        # avoid: a list of (x,y)
        while True:
            random_x = random.randrange(0, (self.board_x - self.boxLength), self.boxLength)
            random_y = random.randrange(0, (self.board_y - self.boxLength), self.boxLength)
            if avoid == None or not (random_x, random_y) in avoid:
                break

        self.rect = Rect(random_x, random_y, self.boxLength, self.boxLength)
        self.x = random_x
        self.y = random_y

class Wall():
    """
    Wall Object
    """

    def __init__(self, size, body):

        self.size = size
        self.pos = []
        self.body = []

    def reset_pos(self, mode=0, pos=None):
        """ Reset Wall, modes:
            - -1: empty (no walls)
            - 0: only four walls
            - 1: random obstacles
            - 2: user-input walls
        """
        self.pos = []

        if mode == 0:
            None

        if mode == 1:
            None

        if mode == 2:
            self.pos = pos

        

    def reset_body(self):
        # reset body of wall using self.pos

        self.body = []
        for (x,y) in self.pos:
            body.append(Rect(x, y, self.size, self.size))
