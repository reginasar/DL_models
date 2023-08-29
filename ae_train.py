import tensorflow as tf
import tensorflow_probability as tfp
import keras
import numpy as np
import yaml
from models.cnn_ae import encoder, decoder


config = yaml.load(open("config/ae.yaml", "r"), Loader=yaml.FullLoader)
input_shape = eval(config["input_shape"])
base_depth = eval(config["base_depth"])
encoded_size = eval(config["encoded_size"])
batch_size = eval(config["batch_size"])
epochs = eval(config["epochs"])
optimizer = eval(config["optimizer"])
loss = eval(config["loss"])
learning_rate = eval(config["learning_rate"])
metrics = eval(config["metrics"])
path_model = eval(config["path_model"])
variational = eval(config["variational"])
distribution = eval(config["distribution"])

#--------load data--------------------------------
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#-------------------------------------------------

rng = np.random.default_rng(seed=12345)
perm_indices = rng.permutation(np.arange(x_train.shape[0]))

x_val = np.copy(x_train[perm_indices[:5000], :, :]) / 255.
y_val = np.copy(y_train[perm_indices[:5000]])

x_train = np.copy(x_train[perm_indices[5000:], :, :])  / 255.
y_train = np.copy(y_train[perm_indices[5000:]])

x_test /= 255.

if variational:
    loss = lambda x, rv_x: -rv_x.log_prob(x) #negloglik
    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(encoded_size), scale=1), reinterpreted_batch_ndims=1)
    encoder = encoder(base_depth, input_shape, encoded_size, variational, prior)
    decoder = decoder(base_depth, encoded_size, input_shape, variational, distribution)
else:
    encoder = encoder(base_depth, input_shape, encoded_size, variational)
    decoder = decoder(base_depth, encoded_size, input_shape, variational)

encoder.summary()
decoder.summary()


ae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

ae.compile(optimizer=optimizer, loss=loss, metrics=metrics)

hist = ae.fit(x_train, x_train, epochs=epochs, validation_data=(x_val, x_val), batch_size=batch_size)

if path_model[-1]!="/":
    path_model += "/"

ae.save_weights(path_model+'ae_weights.sav')

