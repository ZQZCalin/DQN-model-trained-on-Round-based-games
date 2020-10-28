import pygame
from pygame.locals import *
import sys
import random
import os
from snakeClass import Apple, Snake

pygame.init()

RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GRAY = (0, 50, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 51)

WIDTH = 400
HEIGHT = 400

FPS = 15
fpsClock = pygame.time.Clock()


WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")

SNAKECOLOR = GREEN
# GRIDSIZE = 10
GRIDSIZE = 20
SNAKELENGTH = 3

DIRECTION = ["U", "R", "D", "L"]
DIRECTION_INVERSE = {"U":0, "R":1, "D":2, "L":3}

SNAKE = Snake(200, 200, 3, "R", SNAKECOLOR, GRIDSIZE)
APPLE = Apple(GRIDSIZE, 100, 100, RED, WIDTH, HEIGHT)


def gameButtons():
        global button_rect1, button_rect2, width_1, height_1, width_2, height_2
        choice = pygame.font.Font('freesansbold.ttf', 26)
        windowCenter = WINDOW.get_rect().center
    # Snake1 Button
        choice1Surface = choice.render("Snake 1", True, WHITE)
        choice1Rect = choice1Surface.get_rect()
        choice1Rect.center = windowCenter
        choice1Rect.y -= 0.05 * HEIGHT
        button_rect1 = choice1Rect.inflate(28, 7)
        pygame.draw.rect(WINDOW, GREEN, button_rect1)
        pygame.draw.rect(WINDOW, WHITE, button_rect1, 2)
        WINDOW.blit(choice1Surface, (choice1Rect.x, choice1Rect.y))
        width_1 = choice1Surface.get_width() + 28
        height_1 = choice1Surface.get_height() + 7
        # Snake2 Button
        choice2Surface = choice.render("Snake 2", True, WHITE)
        choice2Rect = choice2Surface.get_rect()
        choice2Rect.center = windowCenter
        choice2Rect.y += 0.05 * HEIGHT
        button_rect2 = choice2Rect.inflate(28, 7)
        pygame.draw.rect(WINDOW, GREEN, button_rect2)
        pygame.draw.rect(WINDOW, WHITE, button_rect2, 2)
        WINDOW.blit(choice2Surface, (choice2Rect.x, choice2Rect.y))
        width_2 = choice2Surface.get_width() + 28
        height_2 = choice2Surface.get_height() + 7


def settingButtons():
    global scButtonRect, fpsButtonRect, gsButtonRect, sc, fps, gs
    # sc button left
    scLeft = pygame.font.Font('freesansbold.ttf', 22)
    sclSurface = scLeft.render("Snake Color", True, GREEN)
    WINDOW.blit(sclSurface, (WIDTH * 0.1, HEIGHT * 0.69))
    # sc button right
    sc = "green"
    scButtonRect = Rect(WIDTH * 0.6, HEIGHT * 0.68, 122, 28)
    scRight = pygame.font.Font('freesansbold.ttf', 22)
    scrSurface = scRight.render(sc, True, WHITE)
    scrRect = scrSurface.get_rect()
    scrRect.center = scButtonRect.center
    pygame.draw.rect(WINDOW, GREEN, scButtonRect)
    pygame.draw.rect(WINDOW, WHITE, scButtonRect, 2)
    WINDOW.blit(scrSurface, (scrRect.x, scrRect.y))
    # FPS button left
    fpsLeft = pygame.font.Font('freesansbold.ttf', 22)
    fpslSurface = fpsLeft.render("Snake Speed", True, GREEN)
    WINDOW.blit(fpslSurface, (WIDTH * 0.1, HEIGHT * 0.79))
    # FPS button right
    fps = "15"
    fpsButtonRect = Rect(WIDTH * 0.6, HEIGHT * 0.78, 122, 28)
    fpsRight = pygame.font.Font('freesansbold.ttf', 22)
    fpsrSurface = fpsRight.render(fps, True, WHITE)
    fpsrRect = fpsrSurface.get_rect()
    fpsrRect.center = fpsButtonRect.center
    pygame.draw.rect(WINDOW, GREEN, fpsButtonRect)
    pygame.draw.rect(WINDOW, WHITE, fpsButtonRect, 2)
    WINDOW.blit(fpsrSurface, (fpsrRect.x, fpsrRect.y))
    # Grid Size button left
    gsLeft = pygame.font.Font('freesansbold.ttf', 22)
    gslSurface = gsLeft.render("Grid Size", True, GREEN)
    WINDOW.blit(gslSurface, (WIDTH * 0.1, HEIGHT * 0.89))
    # Grid Size button right
    gs = "1 X 1"
    gsButtonRect = Rect(WIDTH * 0.6, HEIGHT * 0.88, 122, 28)
    gsRight = pygame.font.Font('freesansbold.ttf', 22)
    gsrSurface = gsRight.render("1 X 1", True, WHITE)
    gsrRect = gsrSurface.get_rect()
    gsrRect.center = gsButtonRect.center
    pygame.draw.rect(WINDOW, GREEN, gsButtonRect)
    pygame.draw.rect(WINDOW, WHITE, gsButtonRect, 2)
    WINDOW.blit(gsrSurface, (gsrRect.x, gsrRect.y))


def settingButtonsFunctions(event):
    global scButtonRect, fpsButtonRect, gsButtonRect, sc, fps, gs
    global SNAKECOLOR, FPS, GRIDSIZE
    mousePosition = pygame.mouse.get_pos()
    # change color
    if (mousePosition[0] >= scButtonRect.x and
            mousePosition[0] <= scButtonRect.x + 122 and
            mousePosition[1] >= scButtonRect.y and
            mousePosition[1] <= scButtonRect.y + 28):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if sc == 'green':
                        sc = 'blue'
                        SNAKECOLOR = BLUE
                    elif sc == 'blue':
                        sc = 'yellow'
                        SNAKECOLOR = YELLOW
                    else:
                        sc = 'green'
                        SNAKECOLOR = GREEN
                    scButtonRect = Rect(WIDTH * 0.6, HEIGHT * 0.68, 122, 28)
                    scRight = pygame.font.Font('freesansbold.ttf', 22)
                    scrSurface = scRight.render(sc, True, WHITE)
                    scrRect = scrSurface.get_rect()
                    scrRect.center = scButtonRect.center
                    pygame.draw.rect(WINDOW, GREEN, scButtonRect)
                    pygame.draw.rect(WINDOW, WHITE, scButtonRect, 2)
                    WINDOW.blit(scrSurface, (scrRect.x, scrRect.y))
    # change fps
    if (mousePosition[0] >= fpsButtonRect.x and
            mousePosition[0] <= fpsButtonRect.x + 122 and
            mousePosition[1] >= fpsButtonRect.y and
            mousePosition[1] <= fpsButtonRect.y + 28):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if fps == '15':
                        fps = '30'
                        FPS = 30
                    else:
                        fps = '15'
                        FPS = 15
                    fpsButtonRect = Rect(WIDTH * 0.6, HEIGHT * 0.78, 122, 28)
                    fpsRight = pygame.font.Font('freesansbold.ttf', 22)
                    fpsrSurface = fpsRight.render(fps, True, WHITE)
                    fpsrRect = fpsrSurface.get_rect()
                    fpsrRect.center = fpsButtonRect.center
                    pygame.draw.rect(WINDOW, GREEN, fpsButtonRect)
                    pygame.draw.rect(WINDOW, WHITE, fpsButtonRect, 2)
                    WINDOW.blit(fpsrSurface, (fpsrRect.x, fpsrRect.y))
    # change grid Size
    if (mousePosition[0] >= gsButtonRect.x and
            mousePosition[0] <= gsButtonRect.x + 122 and
            mousePosition[1] >= gsButtonRect.y and
            mousePosition[1] <= gsButtonRect.y + 28):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if gs == '1 X 1':
                        gs = '2 X 2'
                        GRIDSIZE = 20
                    elif gs == '2 X 2':
                        gs = '4 X 4'
                        GRIDSIZE = 25
                    else:
                        gs = '1 X 1'
                        GRIDSIZE = 10
                    gsButtonRect = Rect(WIDTH * 0.6, HEIGHT * 0.88, 122, 28)
                    gsRight = pygame.font.Font('freesansbold.ttf', 22)
                    gsrSurface = gsRight.render(gs, True, WHITE)
                    gsrRect = gsrSurface.get_rect()
                    gsrRect.center = gsButtonRect.center
                    pygame.draw.rect(WINDOW, GREEN, gsButtonRect)
                    pygame.draw.rect(WINDOW, WHITE, gsButtonRect, 2)
                    WINDOW.blit(gsrSurface, (gsrRect.x, gsrRect.y))


def showStartScreen():
    global GAMECHOICE, SNAKECOLOR
    settingButtons()
    while True:
        title = pygame.font.Font('freesansbold.ttf', 45)
        titleSurface = title.render("SNAKE!", True, GREEN)
        windowCenter = WINDOW.get_rect().center
        titleRect = titleSurface.get_rect()
        titleRect.center = windowCenter
        WINDOW.blit(titleSurface, (titleRect.x, titleRect.y - HEIGHT * 0.2))
        gameButtons()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            mousePosition = pygame.mouse.get_pos()
            if (mousePosition[0] >= button_rect1.x and
                    mousePosition[0] <= button_rect1.x + width_1 and
                    mousePosition[1] >= button_rect1.y and
                    mousePosition[1] <= button_rect1.y + height_1):
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        GAMECHOICE = 1
                        return
            if (mousePosition[0] >= button_rect2.x and
                    mousePosition[0] <= button_rect2.x + width_2 and
                    mousePosition[1] >= button_rect2.y and
                    mousePosition[1] <= button_rect2.y + height_2):
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        GAMECHOICE = 2
                        return
            settingButtonsFunctions(event)
        pygame.display.update()
        fpsClock.tick(FPS)


def gameoverPrint():
    title = pygame.font.Font('freesansbold.ttf', 38)
    titleSurface = title.render("GAMEOVER", True, GREEN)
    windowCenter = WINDOW.get_rect().center
    titleRect = titleSurface.get_rect()
    titleRect.center = windowCenter
    WINDOW.blit(titleSurface, (titleRect.x, titleRect.y))
    # scores
    scoreFont = pygame.font.Font('freesansbold.ttf', 22)
    scoreSurface = scoreFont.render("Your Score: %d" % SCORE, True, WHITE)
    yourScoreRect = scoreSurface.get_rect()
    yourScoreRect.center = windowCenter
    WINDOW.blit(scoreSurface, (yourScoreRect.x, yourScoreRect.y + 50))
    highScoreSurface = scoreFont.render("High Score: %s" % HIGHSCORE, True, WHITE)
    highScoreRect = highScoreSurface.get_rect()
    highScoreRect.center = windowCenter
    WINDOW.blit(highScoreSurface, (highScoreRect.x, highScoreRect.y + 100))
    # play agian
    agianSign = "Press 1 to play agian!"
    signFont = pygame.font.Font('freesansbold.ttf', 30)
    signSurface = signFont.render("%s" % agianSign, True, RED)
    signRect = signSurface.get_rect()
    signRect.center = windowCenter
    WINDOW.blit(signSurface, (signRect.x, signRect.y + 150))


def showGameOver():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == K_1:
                    return
        gameoverPrint()
        pygame.display.update()
        fpsClock.tick(FPS)


def drawScore(rect=SNAKE):
    scoreFont = pygame.font.Font('freesansbold.ttf', 18)
    score = len(rect.body) - rect.length
    fontSurface = scoreFont.render("Score: %d" % score, True, WHITE)
    WINDOW.blit(fontSurface, (WIDTH - 100, 10))
    return score


def saveScore():
    if SCORE > int(HIGHSCORE):
        with(open(scoreFilePath, 'w')) as sFile:
            sFile.write('%d' % SCORE)


def mainGame():
    global SCORE, FPS
    original_FPS = FPS
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == K_UP or event.key == K_w:
                    if SNAKE.direction != "D":
                        SNAKE.direction = "U"
                elif event.key == K_DOWN or event.key == K_s:
                    if SNAKE.direction != "U":
                        SNAKE.direction = "D"
                elif event.key == K_RIGHT or event.key == K_d:
                    if SNAKE.direction != "L":
                        SNAKE.direction = "R"
                elif event.key == K_LEFT or event.key == K_a:
                    if SNAKE.direction != "R":
                        SNAKE.direction = "L"
                # the cheat key is SPACE
                if event.key == K_SPACE:
                    if FPS == original_FPS:
                        FPS = int(original_FPS / 2)
                    else:
                        FPS = original_FPS
        SNAKE.addHead()
        if SNAKE.isDead() or SNAKE.isOutOfBounds(WIDTH, HEIGHT):
            break
        if not(SNAKE.head.colliderect(APPLE.rect)):
            SNAKE.deleteTail()
        else:
            APPLE.move()

        WINDOW.fill(BLACK)
        #  draw the grid
        for x in range(0, WIDTH, GRIDSIZE):
            pygame.draw.line(WINDOW, GRAY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRIDSIZE):
            pygame.draw.line(WINDOW, GRAY, (0, y), (WIDTH, y))
        #  draw the apple
        pygame.draw.rect(WINDOW, APPLE.color, APPLE.rect)
        #  draw the snake
        for part in SNAKE.body:
            pygame.draw.rect(WINDOW, SNAKE.color, part)
            part_small = part.inflate(-3, -3)
            pygame.draw.rect(WINDOW, WHITE, part_small, 3)
        #  draw the score
        SCORE = drawScore()
        pygame.display.update()
        fpsClock.tick(FPS)


def mainGame_snakeTwo():
    global SCORE, FPS
    original_FPS = FPS
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == K_UP or event.key == K_w:
                    if SNAKE.direction != "D":
                        SNAKE.direction = "U"
                elif event.key == K_DOWN or event.key == K_s:
                    if SNAKE.direction != "U":
                        SNAKE.direction = "D"
                elif event.key == K_RIGHT or event.key == K_d:
                    if SNAKE.direction != "L":
                        SNAKE.direction = "R"
                elif event.key == K_LEFT or event.key == K_a:
                    if SNAKE.direction != "R":
                        SNAKE.direction = "L"
                # the cheat key is SPACE
                if event.key == K_SPACE:
                    if FPS == original_FPS:
                        FPS = int(original_FPS / 2)
                    else:
                        FPS = original_FPS
        SNAKE.addHead_snakeTwo(WIDTH, HEIGHT)
        if SNAKE.isDead():
            break
        if not(SNAKE.head.colliderect(APPLE.rect)):
            SNAKE.deleteTail()
        else:
            APPLE.move()

        WINDOW.fill(BLACK)
        #  draw the grid
        for x in range(0, WIDTH, SNAKE.boxSize):
            pygame.draw.line(WINDOW, GRAY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, SNAKE.boxSize):
            pygame.draw.line(WINDOW, GRAY, (0, y), (WIDTH, y))
        #  draw the apple
        pygame.draw.rect(WINDOW, APPLE.color, APPLE.rect)
        #  draw the snake
        for part in SNAKE.body:
            pygame.draw.rect(WINDOW, SNAKE.color, part)
            part_small = part.inflate(-3, -3)
            pygame.draw.rect(WINDOW, WHITE, part_small, 3)
        #  draw the score
        SCORE = drawScore()
        pygame.display.update()
        fpsClock.tick(FPS)


# ----------------------------------------------------------------------------
manual_play = 0
if manual_play:
    showStartScreen()
    while True:
        scoreFilePath = os.path.join(".", "HighScore.txt")
        if not (os.path.exists(scoreFilePath)):
            HIGHSCORE = 0
        else:
            with(open(scoreFilePath)) as sFile:
                HIGHSCORE = sFile.readline()
        if HIGHSCORE == '':
            HIGHSCORE = 0
        SNAKE = Snake(200, 200, 3, "R", SNAKECOLOR, GRIDSIZE)
        APPLE = Apple(GRIDSIZE, 100, 100, RED, WIDTH, HEIGHT)
        if GAMECHOICE == 1:
            mainGame()
        else:
            mainGame_snakeTwo()
        saveScore()
        showGameOver()
