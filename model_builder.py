import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_model(config, vocab):

    from tensorflow.keras.layers import (
        Input, Embedding, LSTM, Dense, Concatenate
    )
    from tensorflow.keras.models import Model

    # =========================
    # TEXT INPUT
    # =========================
    essay_input = Input(shape=(config.MAX_LEN_ESSAY,), name="essay_in")
    ref_input   = Input(shape=(config.MAX_LEN_REF,), name="ref_in")

    # =========================
    # NUMERIC INPUT
    # =========================
    sim_input = Input(shape=(1,), name="sim_in")
    rub_input = Input(shape=(1,), name="rub_in")
    len_input = Input(shape=(1,), name="len_in")
    gen_input = Input(shape=(1,), name="gen_in")

    # =========================
    # EMBEDDING
    # =========================
    embedding = Embedding(vocab, config.EMBED_DIM, input_length=config.MAX_LEN)

    e = embedding(essay_input)
    e = LSTM(64)(e)

    r = embedding(ref_input)
    r = LSTM(64)(r)

    # =========================
    # MERGE
    # =========================
    merged = Concatenate()([
        e, r,
        sim_input,
        rub_input,
        len_input
    ])

    x = Dense(128, activation="relu")(merged)
    x = Dense(64, activation="relu")(x)

    output = Dense(1, activation="linear")(x)

    model = Model(
        inputs=[
            essay_input,
            ref_input,
            sim_input,
            rub_input,
            len_input
        ],
        outputs=output
    )

    return model