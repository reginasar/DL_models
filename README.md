# Deep Learning Models

Under construction 🛠️

Some models I've played with, using TensorFlow.

## Autoencoder

A convolutional autoencoder. Unsupervised algorithm that takes an image as input, encodes it into a lower-dimension space and aims to decode/reconstruct the initial image.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reginasar/DL_models/blob/main/colab/autoencoder.ipynb)

- Training: ae_train.py
- Model: models/ae_CNN.py
- Configuration: config/ae.yaml

### Variational Autoencoder

Same as with the autoencoder, switching the 'variational' variable to True (in the configuration file or in the fourth colab cell). In this case the inputs are encoded to parametrs that describe a variational distribution, allowing to sample from it after training to generate new data.

## Contrastive Learning (SimCLR)

## U-net
