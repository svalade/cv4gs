import numpy as np
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
data_trn, data_tst = fashion_mnist.load_data()

data_trn = np.expand_dims(data_trn[0].astype('float32')/255.0, axis=-1), data_trn[1]
data_tst = np.expand_dims(data_tst[0].astype('float32')/255.0, axis=-1), data_tst[1]
# print(data_trn[0].shape, data_trn[0].dtype, data_trn[0].max(), data_trn[0].min())

data_val = data_trn[0][-5000:], data_trn[1][-5000:]
data_trn = data_trn[0][:-5000], data_trn[1][:-5000]

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPooling2D(2),
    # tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    # tf.keras.layers.MaxPooling2D(2),
    # tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    # tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()
run = 00
log_dir = './log/conv/run_%d'%run
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

model.fit(
    data_trn[0], data_trn[1],
    epochs=100,
    steps_per_epoch=100,
    validation_data=data_val,
    callbacks=[tensorboard_callback]
)
