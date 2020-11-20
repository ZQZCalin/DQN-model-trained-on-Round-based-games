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

    def __init__(
        self, x, y, length, direction, 
        boxSize, board_x, board_y
    ):
        self.x = x
        self.y = y
        self.length = length
        self.direction = direction

        self.boxSize = boxSize
        self.board_x = board_x
        self.board_y = board_y

        self.body = []
        self.head = None
        self.initialize()

    def initialize(self, avoid=[]):
        k1 = 0
        k2 = 0
        if self.direction == "R":
            k1 = -1
        elif self.direction == "L":
            k1 = 1
        elif self.direction == "U":
            k2 = 1
        elif self.direction == "D":
            k2 = -1

        for i in range(self.length):
            tempRect = Rect(self.x + k1*i * self.boxSize,
                            self.y + k2*i * self.boxSize, 
                            self.boxSize, self.boxSize)
            self.body.append(tempRect)
        self.head = self.body[0]

    def changeDirection(direction):
        # input: 0,1,2,3
        DIRECTION = ["U", "R", "D", "L"]
        current = DIRECION.index(self.direction)
        opposite = (current + 2) % 4

        if direction != opposite and direction != current:
            self.direction = DIRECTION[direction]

    def addHead(self, collideWall=True):
        k1 = 0
        k2 = 0
        if self.direction == "R":
            k1 = 1
        elif self.direction == "L":
            k1 = -1
        elif self.direction == "U":
            k2 = -1
        elif self.direction == "D":
            k2 = 1

        self.x += k1 * self.boxSize
        self.y += k2 * self.boxSize

        if not collideWall:
            self.x = self.x % self.board_x
            self.y = self.y % self.board_y

        newHead = Rect(self.x, self.y, self.boxSize, self.boxSize)
        self.body.insert(0, newHead)
        self.head = self.body[0]

    def deleteTail(self):
        return self.body.pop()

    def addTail(self, rect):
        this.body.push(rect)

    def collideWithBody(self):
        for part in self.body[1:]:
            if self.head.colliderect(part):
                return True
        return False

    def collideWithWall(self, extraWalls=[]):
        wallRect = Rect(0, 0, self.board_x, self.board_y)
        if not wallRect.contains(self.head):
            return True

        for part in extraWalls:
            if self.head.colliderect(part):
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

    def __init__(self, boxLength, x, y, board_x, board_y):
        self.boxLength = boxLength
        self.x = x
        self.y = y
        self.board_x = board_x
        self.board_y = board_y
        self.rect = Rect(self.x, self.y, self.boxLength, self.boxLength)

    def move(self, avoid=[]):
        # avoid: a list of (x,y)
        while True:
            random_x = random.randrange(0, self.board_x, self.boxLength)
            random_y = random.randrange(0, self.board_y, self.boxLength)
            if not (random_x, random_y) in avoid:
                break

        self.rect = Rect(random_x, random_y, self.boxLength, self.boxLength)
        self.x = random_x
        self.y = random_y
