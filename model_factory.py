# model_factory.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    Bidirectional,
    LSTM,
    Dense,
    Concatenate,
    Dropout,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    BatchNormalization
)
from tensorflow.keras.models import Model
import config

Config = config.Config()

def build_hybrid_spatio_sequential(
    vocab_size,
    max_len,
    num_numeric_features,
    embedding_dim=300
):

    # =====================================
    # INPUT
    # =====================================

    text_input = Input(
        shape=(max_len,),
        name='text_input'
    )

    numeric_input = Input(
        shape=(num_numeric_features,),
        name='numeric_input'
    )

    # =====================================
    # EMBEDDING
    # =====================================

    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        trainable=True
    )(text_input)

    # =====================================
    # CNN BLOCK
    # =====================================

    x = Conv1D(
        filters=Config.CNN_FILTERS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        activation=Config.ACTIVATION,
        padding='same'
    )(x)

    # =====================================
    # BiLSTM BLOCK
    # =====================================

    x = Bidirectional(
        LSTM(
            Config.LSTM_UNITS,
            return_sequences=True
        )
    )(x)

    # =====================================
    # GLOBAL POOLING
    # =====================================

    #x = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    x = Concatenate()([avg_pool, max_pool])

    # =====================================
    # FEATURE INJECTION
    # =====================================

    merged = Concatenate()([
        x,
        numeric_input
    ])

    # =====================================
    # DENSE BLOCK
    # =====================================

    z = Dense(
        Config.DENSE_UNITS,
        activation=Config.ACTIVATION
    )(merged)

    z = BatchNormalization()(z)


    z = Dense(
        128,
        activation=Config.ACTIVATION
    )(z)

    z = Dropout(
        Config.DROPOUT_RATE
    )(z)

    output = Dense(
        1,
        activation='linear'
    )(z)

    # =====================================
    # MODEL
    # =====================================

    model = Model(
        inputs=[
            text_input,
            numeric_input
        ],
        outputs=output
    )

    # =====================================
    # COMPILE
    # =====================================

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=Config.INITIAL_LR
    )

    model.compile(
        optimizer=optimizer,
        loss=Config.LOSS,
        metrics=Config.METRICS
    )

    return model