import pygame
from pygame.locals import *
import sys
import math
import random

def main():
    # window size
    (w, h) = (600, 400)
    # racket center
    (x, y) = (10, h//2)
    (bx, by) = (w//2, h//2)
    vx, vy = 8, 5

    pygame.init() # initialize pygame
    # set screen
    screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
    pygame.display.set_caption("Pygame Pong") # set window title

    pygame.mixer.init(frequency = 44100) # initial setting
    pygame.mixer.music.load("./hit.mp3") # hit sound

    while (1):
        pygame.display.update() # update display
        pygame.time.wait(20) # interval of updating
        screen.fill((0, 0, 0, 0)) # fill screen with black R:0 G:0 B:0

        # racket
        pygame.draw.line(screen, (255, 255, 255), (x, y-40), (x, y+40), 10)
        # ball
        pygame.draw.line(screen, (255, 0, 0), (bx-3, by), (bx+3, by), 8)

        pressed_key = pygame.key.get_pressed()
        # racket movement
        if pressed_key[K_UP]:
            y -= 10
        if pressed_key[K_DOWN]:
            y += 10

        # racket inside window
        if y < 20:
            y = 20
        if y > h-20:
            y = h-20

        # ball movement
        bx += vx
        by += vy

        # ball inside window
        if (bx > w and vx > 0):
            vx = -vx   
        if (0 > by and vy < 0) or (by > h and vy > 0):
            vy = -vy

        # racket hit
        if (y-40 < by < y+40 and vx < 0 and 7 < bx < 15):
            vx = -vx
            pygame.mixer.music.play(1)

        # ball went outside
        if bx < -50:
            pygame.time.wait(500)
            bx, by = (w//2, h//2)
            (x, y) = (10, h//2) # racket center
            # theta = 2*math.pi*random.random()
            theta = 135
            vx = int(math.cos(theta) * 7) + 2*math.cos(theta)/abs(math.cos(theta))
            vy = int(math.sin(theta) * 7) + 2*math.sin(theta)/abs(math.sin(theta))

        for event in pygame.event.get():
            # close button pushed
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            # escape key pressed
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

if __name__ == '__main__':
    main()
