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
import glob
IMG_SIZE = 512

gen = models.generator.get_generator(latent_dim=512, channels=64, target_size=IMG_SIZE, latent_style_layers=2)
gen.compile("adam","binary_crossentropy")

# pygame.init()
# w, h = 400, 400
# screen = pygame.display.set_mode((w, h))
# running = True

# smallfont = pygame.font.SysFont('Corbel', 30)
# nextText = smallfont.render('Generate new picture', True, (0, 0, 0))
# quitText = smallfont.render('X', True, (0, 0, 0))

# Generate random nooise to be used for all generated pictures
r_noise_input = random_generator_input(1, 512, IMG_SIZE)
r_noise2 = r_noise_input.copy()
r_noise2[1] = np.random.normal(size=(1,512))


placeholder = r_noise_input.copy()

counter = 0
n = 100

def update_latent_vector():
    current_latent = (placeholder.copy()[1]*(n-counter) + r_noise2.copy()[1]*counter)/n
    r_noise_input[1] = current_latent.copy()



def generate_image():
    image = gen.predict(r_noise_input)[0]
    image = (image+1)*127.5
    image = image.astype(np.uint8)
    return image
def updateImage():
    global counter
    global r_noise_input
    global r_noise2
    global n
    image = generate_image()
    #image = pygame.surfarray.make_surface(image)

    if counter <= n:
        counter += 1

    update_latent_vector()


    print(counter)

    # rect = image.get_rect()
    # rect.center = w/2, h/2
    return image#, rect


def getNextButtonRect():
    rect = nextText.get_rect()
    rect.center = w/2, h-30
    return rect


def getQuitButtonRect():
    rect = quitText.get_rect()
    rect.center = w-28, h-30
    return rect


img_array = []
img = updateImage()
while counter <= n:
    img = updateImage()
    img_array.append(img)

# for image in img_array:
#     cv2.imshow('Color image', image)
#     import time
#     time.sleep(4)
#     cv2.destroyAllWindows()

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (512,512))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# for img_index in range(len(img_array)):
#     cv2.imshow('Color image', img_array[img_index])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




# image, rect = updateImage()

# time_elapsed_since_last_action = 0
# clock = pygame.time.Clock()
# while running:
#     nextButtonRect = getNextButtonRect()
#     quitButtonRect = getQuitButtonRect()
#     for event in pygame.event.get():
#         if event.type == MOUSEBUTTONDOWN:
#             if nextButtonRect.collidepoint(event.pos):
#                 image, rect = updateImage()
#             if quitButtonRect.collidepoint(event.pos):
#                 running = False
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_RETURN:
#                 image, rect = updateImage()

#     dt = clock.tick() 
#     time_elapsed_since_last_action += dt
#     if time_elapsed_since_last_action > 50:
#         image, rect = updateImage()
    
#     screen.fill((0, 0, 0))
#     screen.blit(image, rect)
#     pygame.draw.rect(screen, (34, 139, 34), [68, h-45, 265, 28])
#     pygame.draw.rect(screen, (255, 0, 0), [w-40, h-45, 25, 27])
#     screen.blit(quitText, quitButtonRect)
#     screen.blit(nextText, nextButtonRect)
#     pygame.display.update()

# pygame.quit()
