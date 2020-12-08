# StyleGAN2
StyleGAN2 is an GAN architecture which provides state-of-the-art results in data-driven unconditional generative image modeling. This implementation is based on a [research paper from NVIDIA from 2020](https://arxiv.org/abs/1912.04958)

## What is a GAN
Two neural networks contest with each other in a game. Given a training set, this technique learns to generate new data with the same statistics as the training set. [Learn more about GAN here](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)

## About us
We are a project group within the student organization Cogito at NTNU (Norwegian University of Science and Technology). The group has worked on this through the fall of 2020.

## Motivation
We wanted to get a better understanding of the hottest image creation technology right now, and NVIDIA had published extremely good results in a paper just months before. We were also unable to find anyone else who had tried to recreate StyleGAN2 so we saw it as a good challange to take.

## Results
Early results were plagued by mode collapse, which means that two vastly different latent vectors gave almost exactly the same output image. We got more success by tuning the learning rate down by a thousand.
![image](https://user-images.githubusercontent.com/45593399/101482364-e9f43b00-3956-11eb-8aae-0da74bb87308.png)
![image](https://user-images.githubusercontent.com/45593399/101482584-3b9cc580-3957-11eb-94ae-30c7c82b22b5.png)
![image](https://user-images.githubusercontent.com/45593399/101482844-9504f480-3957-11eb-86b2-29bf0cdae3dd.png)

The image below shows an pygame application to easily create new images, save them, [and make interpolation videos between them similar to this](https://youtu.be/6E1_dgYlifc)
![image](https://user-images.githubusercontent.com/45593399/101482932-bbc32b00-3957-11eb-9acd-96383a673426.png)

## Improvements
The easiest change we could make would be to train the system longer. NVIDIA ran their model for 51 GPU-years with Titan graphic cards, while we trained the models on a couple of days with modern graphic cards.  

Another major improvement would be to use a month worth of training time to really hoan in the right learning rate, model size and model shape before commiting to letting the model train for real.  

We could also consider hosting an application on a website with different models to make our creation more usable.

## Setup and how to train on your own
The current h5 files take up around 50GB of space for 512x512 resolution! So we will rather give the steps to how you can train your own version.


