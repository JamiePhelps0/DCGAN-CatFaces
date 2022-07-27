import tensorflow as tf
import numpy as np
from PIL import Image
import os

codings = 100
batch_size = 128
lmbda = 10
beta_1 = 0.5
beta_2 = 0.999
n_critic = 5
learning_rate = 7e-5


@tf.function
def tf_func(x):
    x = tf.reshape(x, (64, 64, 3))
    x = (x - 127.5) / 127.5
    return x


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Jamie Phelps/Documents/Cats/Cat-faces-dataset-master/cat_train/', labels=None, label_mode=None,
    batch_size=1, image_size=(64, 64))
dataset = dataset.map(tf_func)
dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(1024).prefetch(1)

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4 * 4 * 1024, input_shape=(codings,), use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(0.2),

    tf.keras.layers.Reshape((4, 4, 1024)),

    tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(0.2),

    tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(0.2),

    tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(0.2),

    tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(0.2),

    tf.keras.layers.Conv2DTranspose(6, (4, 4), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(0.2),

    tf.keras.layers.Conv2D(6, (4, 4), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(0.2),

    tf.keras.layers.Conv2D(3, (1, 1), padding='same', activation='tanh'),
])
critic = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1),
])


@tf.function
def train_critic_step(x):
    z = np.random.normal(size=[batch_size, codings]).astype(np.float32)
    with tf.GradientTape() as discriminator_tape:
        x_fake = generator(z, training=True)
        true_score = critic(x, training=True)
        fake_score = critic(x_fake, training=True)

        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        diff = x_fake - x
        inter = x + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = critic(inter, training=True)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        # epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
        # x_hat = epsilon * x + (1 - epsilon) * x_fake
        # gradients = tf.gradients(critic(x_hat), [x_hat])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        # gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        loss = tf.reduce_mean(fake_score) - tf.reduce_mean(true_score) + lmbda * gradient_penalty

    grads_discriminator_loss = discriminator_tape.gradient(target=loss, sources=critic.trainable_variables)

    critic_optimizer.apply_gradients(zip(grads_discriminator_loss, critic.trainable_variables))
    return loss


@tf.function
def train_gen_step():
    z = np.random.normal(size=[batch_size, codings]).astype(np.float32)
    with tf.GradientTape() as generator_tape:
        x = generator(z, training=True)
        fake_score = critic(x, training=True)

        loss = -tf.reduce_mean(fake_score)

    grads_generator_loss = generator_tape.gradient(target=loss, sources=generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(grads_generator_loss, generator.trainable_variables))
    return loss


generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2)

checkpoint_dir = 'WGAN_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=critic_optimizer,
                                 generator=generator,
                                 discriminator=critic)


def train(epochs):
    for epoch in range(epochs):
        gen_losses = []
        critic_losses = []
        for image_batch in dataset:
            c_loss = np.mean([train_critic_step(image_batch) for _ in range(n_critic)])
            gen_loss = train_gen_step()
            gen_losses.append(gen_loss)
            critic_losses.append(c_loss)

        gen_avg = np.mean(np.array(gen_losses))
        disc_avg = np.mean(np.array(critic_losses))

        # Save the model every 15 epochs
        noise = tf.random.normal(shape=[1, codings])

        image = generator(noise)
        image = tf.cast((image + 1) * 127.5, tf.int32)
        array = np.array(image, dtype=np.uint8).reshape((64, 64, 3))
        image = Image.fromarray(array)
        image.show(title=epoch + 1)
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {0}: Critic Loss {1}: Generator Loss {2}:'.format(epoch + 1, disc_avg, gen_avg))


# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(500)

print(generator)

for i in range(50):
    noise = tf.random.normal(shape=[1, codings])
    print(noise)
    image = generator(noise)
    image = tf.cast((image + 1) * 127.5, tf.int32)
    array = np.array(image, dtype=np.uint8).reshape((64, 64, 3))
    image = Image.fromarray(array)
    image.save('C:/Users/Jamie Phelps/Pictures/FakeCat/cat{0}'.format(i) + '.png')
