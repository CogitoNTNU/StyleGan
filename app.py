import pygame
from pygame.locals import *
import random
import cv2
import models.generator
import models.adverserial
import models.discriminator
from tensorflow.keras.optimizers import Adam
import numpy as np
from models.generator import random_generator_input
import cv2
IMG_SIZE = 512

gen = models.generator.get_generator(latent_dim=512, channels=64, target_size=IMG_SIZE, latent_style_layers=2)
gen.compile("adam","binary_crossentropy")

pygame.init()
w, h = 400, 400
screen = pygame.display.set_mode((w, h))
running = True

smallfont = pygame.font.SysFont('Corbel', 30)
nextText = smallfont.render('Generate new picture', True, (0, 0, 0))
quitText = smallfont.render('X', True, (0, 0, 0))


def generate_image():
    r_noise_input = random_generator_input(1, 512, IMG_SIZE)
    image = gen.predict(r_noise_input)[0]
    image = (image+1)*127.5
    image = image.astype(np.uint8)
    return image
def updateImage():
    image = generate_image()
    image = pygame.surfarray.make_surface(image)

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
         if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                image, rect = updateImage()
    screen.fill((0, 0, 0))
    screen.blit(image, rect)
    pygame.draw.rect(screen, (34, 139, 34), [68, h-45, 265, 28])
    pygame.draw.rect(screen, (255, 0, 0), [w-40, h-45, 25, 27])
    screen.blit(quitText, quitButtonRect)
    screen.blit(nextText, nextButtonRect)
    pygame.display.update()

pygame.quit()
