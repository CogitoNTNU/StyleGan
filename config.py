# General options, used in almost all files:
IMG_SIZE = 64 #Image will end up being IMG_SIZE x IMG_SIZE
DATA_FOLDER = "datasets/cats" # Your training data will be here

# ----------------------------------------------------------------------------------
# Dataprep options
IMAGES_TO_CONVERT = "cats" # Must be a folder of jpg or png images
PACKAGE_SIZE = 128 # How many images each chunk of training data will contain


# ----------------------------------------------------------------------------------
# Training specific options

CONTINUE_TRAINING=False
MODEL_TRAIN_GEN_WEIGHTS= "weights/gen_2020-12-08_22-03-02.h5" # CONTINUE_TRAINING must be True
MODEL_TRAIN_DISC_WEIGHTS= "weights/DISC_2020-12-08_22-03-02.h5" # CONTINUE_TRAINING must be True

GENERATOR_LEARNING_RATE = 0.001  # Default 0.002 Apparently the latent FC mapping network has a 100x lower learning rate? (appendix B)
DISCRIMINATOR_LEARNING_RATE = 0.001
BETA_1 = 0.0  # Exponential decay rate for first moment estimates, BETA_1=0 in the paper. Makes sense since the discriminator changes?
BETA_2 = 0.99
EPSILON = 1e-8
BATCH_SIZE = 8
NUM_BATCHES = 10000000
SAVE_INTERVAL = 200 # Saves when # batches has been processed.

# Generator parameters
LATENT_DIM = 8
CHANNELS = 16
LATENT_STYLE_LAYERS = 2

# Discriminator parameters
FILTERS = 16


# ----------------------------------------------------------------------------------
# App
USE_UNTRAINED_MODEL = False
MODEL_APP_WEIGHTS = "weights/gen_2020-12-08_22-03-02.h5" #USE_UNTRAINED_MODEL must be False