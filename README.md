# StyleGan

## Data

1. Download the abstract art dataset from https://www.kaggle.com/greg115/abstract-art
2. Place into folder `datasets/abstract_art_512`
3. Run the command `python data_tools/data.py` to generate `.h5` files for training.

Another good dataset for debugging: https://www.kaggle.com/spandan2/cats-faces-64x64-for-generative-models

### ImageMagick commands

`./magick mogrify -gravity center -crop 2:3 -quality 85 -resize 512x768 -monitor test/*.jpg`

## IDUN notes

Login: `ssh -l <username> idun-login1.hpc.ntnu.no`
Start interactive GPU session: `srun --nodes=1 --partition=GPUQ --time=00:30:00 --gres=gpu:1 --pty bash`
Load modules:
`module load CUDA`
`module load cuDNN`
`module load GCCcore/.9.3.0`
`module load Python/3.8.2`
Checkout git repo: `git clone https://github.com/CogitoNTNU/StyleGan.git`
Switch to branch: `git checkout <branch-name>`
Create virtual environment `python -m venv env`
Activate `source env/bin/activate`
Upgrade pip `pip install --upgrade pip`
Install dependencies `pip install -r requirements.txt`
Mount network drive (ensure that `/tmp/mntpt/` exists locally): `sudo mount.cifs //idun-samba1.hpc.ntnu.no/<username> /tmp/mntpt -o username=<username>, domain=WIN-NTNU-NO`
Transfer zip file (from `/tmp/mntpt/StyleGan`) `sudo cp ~/Desktop/archive.zip datasets/` 
Unzip: `sudo unzip archive.zip` (remember to do this directly from IDUN or it will take forever)


## Ideas

Transfer learning for the discriminator
Use existing weights for the convolutional layers and only train the classifiction/discrimination part


## Issues

The current h5 files take up around 50GB of space for 512x512 resolution!