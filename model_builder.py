# model_builder.py tidak digunakan lagi karena kita sudah memindahkan logika pembuatan model ke dalam model_factory.py. Namun, jika Anda ingin melihat contoh bagaimana model dibuat, berikut adalah kode yang sebelumnya ada di model_builder.py sebelum dipindahkan:
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_model(config, vocab):

    from tensorflow.keras.layers import (
        Input,
        Embedding,
        Conv1D,
        MaxPooling1D,
        Bidirectional,
        LSTM,
        Dense,
        Dropout,
        GlobalMaxPooling1D,
        Concatenate,
        BatchNormalization
    )

    from tensorflow.keras.models import Model

    # ====================================
    # TEXT INPUT
    # ====================================

    essay_input = Input(
        shape=(config.MAX_LEN_ESSAY,),
        name="essay_in"
    )

    ref_input = Input(
        shape=(config.MAX_LEN_REF,),
        name="ref_in"
    )

    # ====================================
    # NUMERIC FEATURES
    # ====================================

    sim_input = Input(
        shape=(1,),
        name="sim_in"
    )

    rub_input = Input(
        shape=(1,),
        name="rub_in"
    )

    len_input = Input(
        shape=(1,),
        name="len_in"
    )

    course_input = Input(
        shape=(1,),
        name="course_in"
    )

    # ====================================
    # EMBEDDING
    # ====================================

    embedding = Embedding(
        input_dim=vocab,
        output_dim=config.EMBED_DIM
    )

    # ====================================
    # ESSAY BRANCH
    # ====================================

    e = embedding(essay_input)

    e = Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(e)

    e = MaxPooling1D(2)(e)

    e = Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    )(e)

    e = GlobalMaxPooling1D()(e)

    # ====================================
    # REFERENCE BRANCH
    # ====================================

    r = embedding(ref_input)

    r = Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(r)

    r = MaxPooling1D(2)(r)

    r = Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    )(r)

    r = GlobalMaxPooling1D()(r)

    # ====================================
    # FEATURE FUSION
    # ====================================

    merged = Concatenate()([
        e,
        r,
        sim_input,
        rub_input,
        len_input,
        course_input
    ])

    # ====================================
    # DENSE BLOCK
    # ====================================

    x = Dense(
        256,
        activation="relu"
    )(merged)

    x = BatchNormalization()(x)

    x = Dropout(
        config.DROPOUT_RATE
    )(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(
        config.DROPOUT_RATE
    )(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    # ====================================
    # OUTPUT
    # ====================================

    output = Dense(
        1,
        activation="linear"
    )(x)

    # ====================================
    # MODEL
    # ====================================

    model = Model(
        inputs=[
            essay_input,
            ref_input,
            sim_input,
            rub_input,
            len_input,
            course_input
        ],
        outputs=output
    )

    return model