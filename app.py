import pygame
from pygame.locals import *
import random
import cv2
import time
import models.generator
import models.adverserial
import models.discriminator
from tensorflow.keras.optimizers import Adam
import numpy as np
from models.generator import random_generator_input
import cv2
import glob
IMG_SIZE = 256
LATENT_SIZE = 512

# initialize model
gen = models.generator.get_generator(latent_dim=512, channels=64, target_size=IMG_SIZE, latent_style_layers=2)
gen.compile("adam","binary_crossentropy")

# Initialize pygame
pygame.init()
w, h = 1100, 768
screen = pygame.display.set_mode((w, h))
running = True

# Initialize text and font
smallfont = pygame.font.SysFont('Corbel', 30)
nextText = smallfont.render('Generate new', True, (0, 0, 0))
saveText = smallfont.render('Save Image', True, (0, 0, 0))
quitText = smallfont.render('X', True, (0, 0, 0))

# Generate random nooise to be used for all generated pictures

def generate_new():
    global r_noise_input, r_noises
    r_noise_input = random_generator_input(1, 512, IMG_SIZE)
    r_noise2 = r_noise_input.copy()
    r_noise3 = r_noise_input.copy()
    r_noise4 = r_noise_input.copy()
    r_noise5 = r_noise_input.copy()
    r_noise6 = r_noise_input.copy()
    r_noise7 = r_noise_input.copy()
    r_noise8 = r_noise_input.copy()
    r_noise9 = r_noise_input.copy()
    r_noise2[1] = np.random.normal(size=(1,512))
    r_noise3[1] = np.random.normal(size=(1,512))
    r_noise4[1] = np.random.normal(size=(1,512))
    r_noise5[1] = np.random.normal(size=(1,512))
    r_noise6[1] = np.random.normal(size=(1,512))
    r_noise7[1] = np.random.normal(size=(1,512))
    r_noise8[1] = np.random.normal(size=(1,512))
    r_noise9[1] = np.random.normal(size=(1,512))
    r_noises = [r_noise_input, r_noise2, r_noise3, r_noise4, r_noise5, r_noise6, r_noise7, r_noise8, r_noise9]
    placeholder = r_noise_input.copy()

generate_new()



n = 10
counter = 0


def update_latent_vector():
    current_latent = (placeholder.copy()[1]*(n-counter) + r_noise2.copy()[1]*counter)/n
    r_noise_input[1] = current_latent.copy()


def generate_image(noise=r_noise_input):
    image = gen.predict(noise)[0]
    image = (image+1)*127.5
    image = image.astype(np.uint8)
    return image

def updateImage():
    global counter
    global r_noises
    global n
    global original_images
    images = []
    original_images = []
    rects = []
    x = 0
    y = 0
    for i in range(9):
        image = generate_image(r_noises[i])
        original_images.append(image.copy())
        image = pygame.surfarray.make_surface(image)
        images.append(image)

        rect = image.get_rect()
        if i%3==0 and i != 0:
            x = 0
            y += 1
        rect.center = IMG_SIZE*(2*x+1)/2, IMG_SIZE*(2*y+1)/2
        rects.append(rect)
        x += 1

        # if counter <= n:
        #     counter += 1
        # update_latent_vector()

    return images, rects


def getNextButtonRect():
    rect = nextText.get_rect()
    rect.center = IMG_SIZE*3+20 + ((w-20)-(IMG_SIZE*3+20))/2, 40
    return rect


def getQuitButtonRect():
    rect = quitText.get_rect()
    rect.center = w-28, h-30
    return rect

def draw_save_buttons(button_color=(34, 139, 34)):
    for x in range(3):
        for y in range(3):
            pygame.draw.rect(screen, button_color, [IMG_SIZE*3+20+x*90, 160+(90*y), 80, 80])


def blit_and_update_screen(button_color=(34, 139, 34)):
    screen.fill((180, 180, 180))
    for i in range(len(images)):
        screen.blit(images[i], rects[i])
    # Generate_new box
    pygame.draw.rect(screen, button_color, [IMG_SIZE*3+20, 20, (w-20)-(IMG_SIZE*3+20), 40])

    pygame.draw.rect(screen, (255, 0, 0), [w-40, h-45, 25, 27]) # Exit box
    draw_save_buttons()

    # Blit button text
    screen.blit(quitText, quitButtonRect)
    screen.blit(nextText, nextButtonRect)
    #screen.blit(saveText, nextButtonRect)

    # Update
    pygame.display.update()

images, rects = updateImage()

nextButtonRect = getNextButtonRect()
quitButtonRect = getQuitButtonRect()

def mouse_in_generate_new(mouse):
    return IMG_SIZE*3+20 < mouse[0] < (w-20) and 20 < mouse[1] < 60

def mouse_in_exit_square(mouse):
    return w-40 < mouse[0] < w-15 and h-45 < mouse[1] < h-18

def mouse_in_save_image(mouse):
    return IMG_SIZE*3+20 < mouse[0] < IMG_SIZE*3+20+2*90+80 and 160 < mouse[1] < 160+(90*2)+80

while running:
    mouse = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == MOUSEBUTTONDOWN:
            if mouse_in_generate_new(mouse):
                # updates color in generate new rectangle and updates images
                generate_new()
                pygame.draw.rect(screen, (255,0,0), [IMG_SIZE*3+20, 20, (w-20)-(IMG_SIZE*3+20), 40])
                pygame.display.update()
                images, rects = updateImage()
            if mouse_in_save_image(mouse):
                img_pointed_at = []
                x = [IMG_SIZE*3+20+(2*90), IMG_SIZE*3+20+90, IMG_SIZE*3+20]
                y = [160+90*2, 160+90, 160]
                for i in range(len(x)):
                    if mouse[0] > x[i]:
                        img_pointed_at.append(2-i)
                        break
                for i in range(len(y)):
                    if mouse[1] > y[i]:
                        img_pointed_at.append(2-i)
                        break
                print(img_pointed_at)
                saved_image = cv2.cvtColor(original_images[img_pointed_at[0]+3*(img_pointed_at[1])],cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(time.time())+".jpeg", saved_image)
                
            if mouse_in_exit_square(mouse):
                running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                generate_new()
                pygame.draw.rect(screen, (255,0,0), [IMG_SIZE*3+20, 20, (w-20)-(IMG_SIZE*3+20), 40])
                pygame.display.update()
                images, rects = updateImage()
    blit_and_update_screen()

pygame.quit()
