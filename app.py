import pygame
from pygame.locals import *
import random
import cv2

pygame.init()
w, h = 400, 400
screen = pygame.display.set_mode((w, h))
running = True

smallfont = pygame.font.SysFont('Corbel', 30)
nextText = smallfont.render('Generate new picture', True, (0, 0, 0))
quitText = smallfont.render('X', True, (0, 0, 0))


def updateImage():
    image = ['images/img.png', 'images/img2.png',
             'images/img3.png'][random.randint(0, 2)]
    image = pygame.image.load(image)
    image = pygame.transform.scale(image, (300, 300))
    image.convert()
    rect = image.get_rect()
    rect.center = w/2, h/2
    return image, rect


def getNextButtonRect():
    rect = nextText.get_rect()
    rect.center = w/2, h-30
    return rect


def getQuitButtonRect():
    rect = quitText.get_rect()
    rect.center = w-28, h-30
    return rect


image, rect = updateImage()


while running:
    nextButtonRect = getNextButtonRect()
    quitButtonRect = getQuitButtonRect()
    for event in pygame.event.get():
        if event.type == MOUSEBUTTONDOWN:
            if nextButtonRect.collidepoint(event.pos):
                image, rect = updateImage()
            if quitButtonRect.collidepoint(event.pos):
                running = False
    screen.fill((0, 0, 0))
    screen.blit(image, rect)
    pygame.draw.rect(screen, (34, 139, 34), [68, h-45, 265, 28])
    pygame.draw.rect(screen, (255, 0, 0), [w-40, h-45, 25, 27])
    screen.blit(quitText, quitButtonRect)
    screen.blit(nextText, nextButtonRect)
    pygame.display.update()

pygame.quit()
