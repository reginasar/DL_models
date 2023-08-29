import tensorflow as tf
import tensorflow_probability as tfp

def encoder(base_depth, indim, encoder_dim, variational=False, prior=[]):
    seq_model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=indim),
    tf.keras.layers.Conv2D(filters=base_depth, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=base_depth, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=base_depth, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=base_depth, kernel_size=2, padding="valid", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten()])

    if variational:
        seq_model.add(tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(encoder_dim), activation=None))
        seq_model.add(tfp.layers.MultivariateNormalTriL(encoder_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.6)))
    else:
        seq_model.add(tf.keras.layers.Dense(encoder_dim))

    return seq_model


def decoder(base_depth, encoder_dim, outdim, variational=False, distrib=''):
    seq_model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=[encoder_dim]),
    tf.keras.layers.Reshape([1, 1, encoder_dim]),
    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2DTranspose(filters=base_depth, kernel_size=2, padding="valid", activation="relu"),
    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2DTranspose(filters=base_depth, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2DTranspose(filters=base_depth, kernel_size=3, padding="valid", activation="relu"),
    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", activation=None)])

    if variational:
        seq_model.add(tf.keras.layers.Flatten())
        if distrib=='Normal' or distrib=='normal':
            seq_model.add(tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(outdim), activation=None))
            seq_model.add(tfp.layers.IndependentNormal(outdim, tfp.distributions.Normal.sample))
        elif distrib=='Bernoulli' or distrib=='bernoulli':
            seq_model.add(tf.keras.layers.Dense(tfp.layers.IndependentBernoulli.params_size(outdim), activation=None))
            seq_model.add(tfp.layers.IndependentBernoulli(outdim, tfp.distributions.Bernoulli.logits))

    return seq_model





