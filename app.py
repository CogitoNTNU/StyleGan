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
IMG_SIZE = 128
LATENT_SIZE = 512

gen = models.generator.get_generator(latent_dim=512, channels=64, target_size=IMG_SIZE, latent_style_layers=2)
gen.compile("adam","binary_crossentropy")

# pygame.init()
# w, h = 400, 400
# screen = pygame.display.set_mode((w, h))
# running = True

# smallfont = pygame.font.SysFont('Corbel', 30)
# nextText = smallfont.render('Generate new picture', True, (0, 0, 0))
# quitText = smallfont.render('X', True, (0, 0, 0))

num_vectors = 4
# Generate random nooise to be used for all generated pictures
r_noise_input = random_generator_input(1, LATENT_SIZE, IMG_SIZE)
r_latent = np.random.normal(size=(num_vectors,LATENT_SIZE))


placeholder = r_noise_input.copy()

counter = 0
frames_pr_second = 20
frames_pr_image= 50



def update_latent_vector():
    current_latent = np.zeros((1, LATENT_SIZE))
    current_latent[0] = (r_latent[counter//frames_pr_image]*(frames_pr_image-counter%frames_pr_image) +
                         r_latent[counter//frames_pr_image+1]*(counter%frames_pr_image))/frames_pr_image
    r_noise_input[1] = current_latent.copy()

def update_latent_vector_return():
    current_latent = np.zeros((1, LATENT_SIZE))
    current_latent[0] = (r_latent[-1]*(frames_pr_image-counter%frames_pr_image) +
                         r_latent[0]*(counter%frames_pr_image))/frames_pr_image
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

    if counter <= frames_pr_image*(num_vectors-1)-2:
        counter += 1
        update_latent_vector()

    elif counter <= frames_pr_image*(num_vectors)-2:
        counter += 1
        update_latent_vector_return()


    print(counter)

    # rect = image.get_rect()
    # rect.center = w/2, h/2
    return image#, rect


img_array = []
img = updateImage()
while counter <= frames_pr_image*(num_vectors)-2:
    img = updateImage()
    img_array.append(img)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), frames_pr_second, (IMG_SIZE,IMG_SIZE))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
