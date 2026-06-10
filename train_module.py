# train_module.py

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

from model_factory import build_hybrid_spatio_sequential

import config

Config = config.Config()

def run_training(
    X_text,
    X_numeric,
    y,
    vocab_size,
    max_len
):

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    # =============================================================================
    # VALIDASI INPUT
    # =============================================================================
    if len(X_text) != len(X_numeric):
        raise ValueError(
            f"Jumlah X_text ({len(X_text)}) "
            f"!= X_numeric ({len(X_numeric)})"
        )

    if len(X_text) != len(y):
        raise ValueError(
            f"Jumlah X_text ({len(X_text)}) "
            f"!= y ({len(y)})"
        )

    print(f"✓ Text shape    : {X_text.shape}")
    print(f"✓ Numeric shape : {X_numeric.shape}")
    print(f"✓ Label shape   : {y.shape}")

    # =============================================================================
    # NORMALISASI TARGET
    # IMPORTANT:
    # Model regression DL lebih stabil pada range 0-1
    # =============================================================================
    y = y.astype("float32")

    print(f"✓ Target normalized to range 0-1")

    # =============================================================================
    # VALIDATION SPLIT MANUAL
    # JANGAN gunakan validation_split bawaan Keras
    # =============================================================================
    (
        X_text_train,
        X_text_val,
        X_num_train,
        X_num_val,
        y_train,
        y_val
    ) = train_test_split(
        X_text,
        X_numeric,
        y,
        test_size=Config.VALIDATION_SPLIT,
        random_state=42,
        shuffle=True,
        stratify=np.digitize(y, bins=np.linspace(0,1,6))
    )

    print(f"\n✓ Train samples      : {len(y_train)}")
    print(f"✓ Validation samples : {len(y_val)}")

    # =============================================================================
    # FEATURE SIZE
    # =============================================================================
    num_numeric_features = X_numeric.shape[1]

    print(f"✓ Numeric feature dim: {num_numeric_features}")

    # =============================================================================
    # BUILD MODEL
    # =============================================================================
    model = build_hybrid_spatio_sequential(
        vocab_size=vocab_size,
        max_len=max_len,
        num_numeric_features=num_numeric_features
    )

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    model.summary()

    # =============================================================================
    # CALLBACKS
    # =============================================================================
    callbacks = [

        # =========================================
        # EARLY STOPPING
        # =========================================
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=Config.ES_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        # =========================================
        # LR REDUCTION
        # =========================================
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=Config.RLRP_FACTOR,
            patience=Config.RLRP_PATIENCE,
            min_lr=Config.MIN_LR,
            verbose=1
        ),

        # =========================================
        # SAVE BEST MODEL
        # =========================================
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{Config.OUTPUT_DIR}/best_model.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

    ]

    print("\n" + "=" * 60)
    print("START TRAINING")
    print("=" * 60)

    # =============================================================================
    # TRAINING
    # =============================================================================
    history = model.fit(

        # =========================================
        # MULTI INPUT
        # =========================================
        x=[
            X_text_train,
            X_num_train
        ],

        y=y_train,

        # =========================================
        # VALIDATION
        # =========================================
        validation_data=(
            [
                X_text_val,
                X_num_val
            ],
            y_val
        ),

        # =========================================
        # TRAINING CONFIG
        # =========================================
        epochs=Config.EPOCHS,

        batch_size=Config.BATCH_SIZE,

        callbacks=callbacks,

        verbose=1,

        shuffle=True

    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("\nLABEL DISTRIBUTION")
    print(f"MIN : {y.min()}")
    print(f"MAX : {y.max()}")
    print(f"MEAN: {y.mean()}")


    return model, history