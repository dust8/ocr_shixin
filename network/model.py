import os.path

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    GRU,
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Flatten,
    Lambda,
    MaxPooling2D,
    Permute,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam


class ShiXinModel:
    def __init__(
        self,
        architecture="dust",
        image_width=160,
        image_height=70,
        n_class=36,
        n_len=4,
        greedy=True,
        beam_width=10,
        top_paths=1,
    ):
        self.architecture = globals()[architecture]
        self.image_width = image_width
        self.image_height = image_height
        self.n_class = n_class
        self.n_len = n_len

        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = top_paths

        self.model = None

    def compile(self):
        # 有 ctc blank 需要加 1
        outs = self.architecture(
            self.image_width, self.image_height, self.n_class + 1, self.n_len
        )
        base, train, optimizer = outs
        self.model = Model(inputs=base[0], outputs=base[1])
        self.train_model = Model(inputs=train[0], outputs=train[1])

        self.model.compile(
            optimizer=optimizer, loss={"out_dense": lambda y_true, y_pred: y_pred}
        )
        self.train_model.compile(
            optimizer=optimizer, loss={"ctc": lambda y_true, y_pred: y_pred}
        )

    def load_checkpoint(self, target):
        """ Load a model with checkpoint file"""

        if os.path.isfile(target):
            if self.model is None or self.train_model is None:
                self.compile()

            self.model.load_weights(target)
            self.train_model.load_weights(target)

    def get_callbacks(self, logdir, checkpoint, monitor="val_loss", verbose=0):
        callbacks = [
            EarlyStopping(
                monitor=monitor, patience=20, restore_best_weights=True, verbose=verbose
            ),
            CSVLogger(filename=os.path.join(logdir, "epochs.log")),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
            ),
        ]
        return callbacks

    def fit(self, **kwargs):
        out = self.train_model.fit(**kwargs)
        return out

    def predict(self, ctc_decode=True, **kwargs):
        self.model._make_predict_function()

        out = self.model.predict(**kwargs)

        if not ctc_decode:
            return np.log(out.clip(min=1e-8))

        out = K.get_value(
            K.ctc_decode(
                out,
                np.ones(out.shape[0]) * out.shape[1],
                self.greedy,
                self.beam_width,
                self.top_paths,
            )[0][0]
        )
        return out


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def dust(img_w, img_h, n_class, n_len):
    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    rnn_size = 512

    inputs = Input(shape=(img_h, img_w, 1), name="the_input")
    labels = Input(shape=[n_len], name="the_labels")
    input_length = Input(shape=[1], name="input_length")
    label_length = Input(shape=[1], name="lable_length")

    x = inputs

    # cnn
    x = Conv2D(conv_filters, kernel_size, padding="same", name="conv1",)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size), name="max1")(x)

    x = Conv2D(conv_filters, kernel_size, padding="same", name="conv2",)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size), name="max2")(x)

    x = Conv2D(conv_filters, kernel_size, padding="same", name="conv3",)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(pool_size, 1), name="max3")(x)

    # 图片高跟宽交换
    x = Permute((2, 1, 3))(x)

    # cnn -> rnn
    x = TimeDistributed(Flatten())(x)

    # rnn
    x = Bidirectional(GRU(rnn_size, return_sequences=True), name="gru_1")(x)
    x = Bidirectional(GRU(rnn_size, return_sequences=True), name="gru_2")(x)

    # 字符分类
    y_pred = Dense(n_class, activation="softmax", name="out_dense")(x)

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    optimizer = Adam(1e-3, amsgrad=True)

    return (
        ([inputs], y_pred),
        ([inputs, labels, input_length, label_length], loss_out),
        optimizer,
    )
