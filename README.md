# StyleGan

## Data

1. Download the abstract art dataset from https://www.kaggle.com/greg115/abstract-art
2. Place into folder `datasets/abstract_art_512`
3. Run the command `python data_tools/data.py` to generate `.h5` files for training.



## IDUN notes

Login: `ssh -l <username> idun-login1.hpc.ntnu.no`
Start interactive GPU session: `srun --nodes=1 --partition=GPUQ --time=00:30:00 --gres=gpu:1 --pty bash`

How to install dependencies and upload dataset?

Use the module manager to load python and cudnn

## Ideas

Transfer learning for the discriminator
Use existing weights for the convolutional layers and only train the classifiction/discrimination part
