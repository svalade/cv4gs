import tensorflow as tf


def train(dim_hidden_layer, epochs=30):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    data_trn, data_tst = fashion_mnist.load_data()

    data_trn = data_trn[0].astype('float32')/255.0, data_trn[1]
    data_tst = data_tst[0].astype('float32')/255.0, data_tst[1]
    # print(data_trn[0].shape, data_trn[0].dtype, data_trn[0].max(), data_trn[0].min())

    data_val = data_trn[0][-5000:], data_trn[1][-5000:]
    data_trn = data_trn[0][:-5000], data_trn[1][:-5000]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(dim_hidden_layer, activation="relu"),
        # tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.summary()
    log_dir = './log/mlp/run_%d'%dim_hidden_layer
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

    model.fit(
        data_trn[0], data_trn[1],
        epochs=epochs,
        steps_per_epoch=100,
        validation_data=data_val,
        callbacks=[tensorboard_callback]
    )



for d in [100, 300, 1000]: #, 10000]:
    train(d, epochs=100)
