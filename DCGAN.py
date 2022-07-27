import random
import tensorflow as tf
import numpy as np
from PIL import Image


learning_rate_g = 0.0001
learning_rate_d = 0.00003
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_g)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_d)
batch_size = 128
latent_size = [100, ]


def get_gen():
    generator = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4 * 4 * 1024, input_shape=latent_size, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Reshape((4, 4, 1024)),

        tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh', use_bias=False),
    ])
    return generator


def get_disc():
    discriminator = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    return discriminator


def tf_func(y):
    y = tf.reshape(y, (64, 64, 3))
    y = tf.cast((y - 127.5) / 127.5, dtype=tf.float64)
    return y


def get_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '/content/drive/My Drive/ML_Data/cat_train/', labels=None, label_mode=None,
        batch_size=1, image_size=(64, 64))
    dataset = dataset.map(tf_func)
    dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(128).prefetch(3)
    return dataset


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(images, g_opt, d_opt, generator, discriminator):
    noise = tf.random.normal([batch_size, latent_size[0]])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(epochs, g_opt, d_opt, dataset, generator, discriminator):
    for epoch in range(epochs):
        gen_losses = []
        disc_losses = []
        for image_batch in dataset:
            gen, disc = train_step(image_batch, g_opt, d_opt, generator, discriminator)
            gen_losses.append(gen)
            disc_losses.append(disc)

        gen_avg = np.mean(np.array(gen_losses))
        disc_avg = np.mean(np.array(disc_losses))
        print('Epoch {}'.format(epoch + 1, ) + ': Gen avg loss {0}: Disc avg loss {1}'.format(gen_avg, disc_avg))
        generator_net.save_weights('/content/drive/My Drive/ML_Models/DCGAN_Model_GEN.h5', overwrite=True)
        disc_net.save_weights('/content/drive/My Drive/ML_Models/DCGAN_Model_DISC.h5', overwrite=True)
    print('*************************')


image_dataset = get_dataset()
generator_net = get_gen()
disc_net = get_disc()

generator_net.load_weights('/content/drive/My Drive/ML_Models/DCGAN_Model_GEN.h5')
disc_net.load_weights('/content/drive/My Drive/ML_Models/DCGAN_Model_DISC.h5')

# train(100, gen_optimizer, disc_optimizer, image_dataset, generator_net, disc_net)

generator_net.save_weights('/content/drive/My Drive/ML_Models/DCGAN_Model_GEN.h5', overwrite=True)
disc_net.save_weights('/content/drive/My Drive/ML_Models/DCGAN_Model_DISC.h5', overwrite=True)

sample(50, generator_net)
