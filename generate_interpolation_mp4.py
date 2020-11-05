import numpy as np
from models.generator import random_generator_input
import cv2
# saves an mp4 file that interpolates between the latent vectors in latent_list
# Takes in an instance of a generator, the image size, the latent vectors to interpolate between, and the filename of the video
#
def generate_interpolation_mp4(generator,img_size, latent_list, seconds_pr_image=1, savePath="video.avi"):
    num_vectors = len(latent_list)
    # Generate random noise to be used for all generated pictures
    latent_size = latent_list[0].shape[-1]
    r_noise_input = random_generator_input(1, latent_size, img_size)
    r_latent = latent_list
    r_noise_input[1][0] = r_latent[0]


    frames_pr_second = 60
    frames_pr_image = int(frames_pr_second*seconds_pr_image)
    global counter
    counter=0
    def update_latent_vector():
        current_latent = np.zeros((1, latent_size))
        current_latent[0] = (r_latent[counter // frames_pr_image] * (frames_pr_image - counter % frames_pr_image) +
                             r_latent[counter // frames_pr_image + 1] * (counter % frames_pr_image)) / (frames_pr_image)
        r_noise_input[1] = current_latent.copy()

    def update_latent_vector_return():
        current_latent = np.zeros((1, latent_size))
        current_latent[0] = (r_latent[-1] * (frames_pr_image - counter % frames_pr_image) +
                             r_latent[0] * (counter % frames_pr_image)) / frames_pr_image
        r_noise_input[1] = current_latent.copy()

    def generate_image():
        image = generator.predict(r_noise_input)[0]
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        return image

    def updateImage():
        global counter
        global r_noise_input
        global r_noise2
        global n
        image = generate_image()
        # image = pygame.surfarray.make_surface(image)

        if counter <= frames_pr_image * (num_vectors - 1) - 2:
            counter += 1
            update_latent_vector()

        elif counter <= frames_pr_image * (num_vectors) - 2:
            counter += 1
            update_latent_vector_return()

        print(counter)

        # rect = image.get_rect()
        # rect.center = w/2, h/2
        return image  # , rect

    img_array = []
    while counter <= frames_pr_image * (num_vectors) - 2:
        img = updateImage()
        img_array.append(img)

    out = cv2.VideoWriter(savePath, cv2.VideoWriter_fourcc(*'DIVX'), frames_pr_second, (img_size, img_size))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == "__main__":
    import models.generator
    import models.adverserial
    import models.discriminator
    from tensorflow.keras.optimizers import Adam
    IMG_SIZE = 64
    LATENT_SIZE = 8
    FILTERS=64
    CHANNELS=64
    NUM_IMAGES=3
    latents = np.random.normal(size=(NUM_IMAGES, LATENT_SIZE))

    disc = models.discriminator.get_resnet_discriminator(IMG_SIZE, filters=FILTERS)
    disc_optimizer = Adam(
        lr=1
    )
    disc.compile(optimizer=disc_optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    gen = models.generator.get_skip_generator(latent_dim=LATENT_SIZE, channels=CHANNELS, target_size=IMG_SIZE, latent_style_layers=2)
    gen.compile("adam", "binary_crossentropy")

    adv = models.adverserial.get_adverserial(gen, disc)
    adv_optimizer = Adam(
        lr=1
    )
    adv.compile(optimizer=adv_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    adv.load_weights("weights.h5")

    generate_interpolation_mp4(gen,IMG_SIZE,latents,1)
