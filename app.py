import pygame
from pygame.locals import *
import random
import cv2
import time
import os
import models.generator
import models.adverserial
import models.discriminator
from tensorflow.keras.optimizers import Adam
import numpy as np
from models.generator import random_generator_input
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

# Generate random noise to be used for all generated pictures in an epoch
def generate_new():
    global r_noise_input, r_noises, placeholder
    r_noise_input = random_generator_input(1, 512, IMG_SIZE)
    r_noises = []
    placeholder = r_noise_input.copy()
    for x in range(9):
        noise = r_noise_input.copy()
        noise[1] = np.random.normal(size=(1,512))
        r_noises.append(noise)
    
# Initialize noises
generate_new()


# Inintialize variables for latent space simulation
n = 10
counter = 0


def update_latent_vector():
    # Updates latent space vector image based on timestep count
    current_latent = (placeholder.copy()[1]*(n-counter) + r_noise2.copy()[1]*counter)/n
    r_noise_input[1] = current_latent.copy()


def generate_image(noise=r_noise_input):
    # Create an image given a noise input vector
    image = gen.predict(noise)[0]
    image = (image+1)*127.5
    image = image.astype(np.uint8)
    return image

def updateImage():
    # return images and grid coordinates for each image center
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
    # returns center coordinates for "Generate New" text
    rect = nextText.get_rect()
    rect.center = IMG_SIZE*3+20 + ((w-20)-(IMG_SIZE*3+20))/2, 40
    return rect

def getSaveImageRect():
    # returns center coordinates for "Save Image" text
    rect = saveText.get_rect()
    rect.center = IMG_SIZE*3+10 + ((w-20)-(IMG_SIZE*3+20))/2, 140
    return rect

def getQuitButtonRect():
    # returns center coordinates for "X" text on quit button
    rect = quitText.get_rect()
    rect.center = w-28, h-30
    return rect

def draw_save_buttons(button_color=(34, 139, 34)):
    # Draw 3x3 grid of squares with a given color
    for x in range(3):
        for y in range(3):
            pygame.draw.rect(screen, button_color, [IMG_SIZE*3+20+x*90, 160+(90*y), 80, 80])


def blit_and_update_screen(button_color=(34, 139, 34)):
    screen.fill((180, 180, 180))
    for i in range(len(images)):
        screen.blit(images[i], rects[i])
    # generate "Generate new" box
    pygame.draw.rect(screen, button_color, [IMG_SIZE*3+20, 20, (w-20)-(IMG_SIZE*3+20), 40])

    pygame.draw.rect(screen, (255, 0, 0), [w-40, h-45, 25, 27]) # Exit box
    draw_save_buttons(button_color) # Generate save buttons

    # Blit button text
    screen.blit(quitText, quitButtonRect)
    screen.blit(nextText, nextButtonRect)
    screen.blit(saveText, saveTextRect)

    # Update screen changes
    pygame.display.update()

images, rects = updateImage()

nextButtonRect = getNextButtonRect()
quitButtonRect = getQuitButtonRect()
saveTextRect = getSaveImageRect()

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
                # updates color and updates images
                generate_new()
                blit_and_update_screen((64, 148, 245))
                # pygame.draw.rect(screen, (64, 148, 245), [IMG_SIZE*3+20, 20, (w-20)-(IMG_SIZE*3+20), 40])
                # pygame.display.update()
                images, rects = updateImage()
            if mouse_in_save_image(mouse):
                # Finds out which image you want saved, and saves images to destination folder
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
                blit_and_update_screen((64, 148, 245))
                pygame.display.update()
                time.sleep(0.3)
                saved_image = cv2.cvtColor(original_images[img_pointed_at[0]+3*(img_pointed_at[1])],cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join("saved_images/",str(time.time())+".jpeg"), saved_image)
                
            if mouse_in_exit_square(mouse):
                # exits program if you press exit button
                running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Generate new image if Enter key is pressed
                generate_new()
                blit_and_update_screen((64, 148, 245))
                # pygame.draw.rect(screen, (64, 148, 245), [IMG_SIZE*3+20, 20, (w-20)-(IMG_SIZE*3+20), 40])
                # pygame.display.update()
                images, rects = updateImage()
    blit_and_update_screen()

pygame.quit()
