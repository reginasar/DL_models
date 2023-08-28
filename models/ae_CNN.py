import tensorflow as tf
import tensorflow_probability as tfp

def encoder(base_depth, indim, encoder_dim, variational=False, prior=[]):
    seq_model = tfk.Sequential([tf.keras.layers.InputLayer(input_shape=[indim]),
    tf.keras.layers.Conv2D(filters=base_depth, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=base_depth, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=base_depth, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=base_depth, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.Flatten()])

    if variational:
        seq_model.add(tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(encoder_dim), activation=None))
        seq_model.add(tfp.layers.MultivariateNormalTriL(encoder_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.6)))
    else:
        seq_model.add(tf.keras.layers.Dense(encoder_dim))

    return seq_model


def decoder(base_depth, encoder_dim, outdim, variational=False):
    seq_model = tfk.Sequential([tf.keras.layers.InputLayer(input_shape=[encoder_dim])])
    
    if variational:
        seq_model.add(tf.keras.layers.Reshape([1, 1, encoder_dim]))

    seq_model.add(tf.keras.layers.Conv2Dtranspose(filters=base_depth, kernel_size=3, padding="same", activation="relu"))
    seq_model.add(tf.keras.layers.Conv2Dtranspose(filters=base_depth, kernel_size=3, padding="same", activation="relu"))
    seq_model.add(tf.keras.layers.Conv2Dtranspose(filters=base_depth, kernel_size=3, padding="same", activation="relu"))
    seq_model.add(tf.keras.layers.Conv2Dtranspose(filters=base_depth, kernel_size=3, padding="same", activation="relu"))
    seq_model.add(tf.keras.layers.Conv2D(filters=base_depth, kernel_size=3, padding="same", activation=None))

    if variational:
        seq_model.add(tf.keras.layers.Flatten())
        seq_model.add(tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(encoder_dim),activation=None))
        seq_model.add(tfp.layers.IndependentNormal(encoder_dim, tfp.distributions.Normal.sample))

    return seq_model



