import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from tensorflow import keras
from tensorflow.keras import layers


class AutoencoderHyperModel(kt.HyperModel):

    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):

        inputs = keras.Input(shape=(self.input_dim,))

        x = inputs

        # =====================================================
        # ENCODER
        # =====================================================

        n_layers = hp.Int(
            "n_layers",
            min_value=1,
            max_value=4,
        )

        for i in range(n_layers):

            units = hp.Choice(
                f"units_{i}",
                values=[32, 64, 128, 256],
            )

            x = layers.Dense(
                units,
                activation="relu",
            )(x)

            dropout = hp.Float(
                f"dropout_{i}",
                min_value=0.0,
                max_value=0.5,
                step=0.1,
            )

            x = layers.Dropout(dropout)(x)

        # =====================================================
        # LATENT
        # =====================================================

        latent_dim = hp.Choice(
            "latent_dim",
            values=[8, 16, 32, 64],
        )

        latent = layers.Dense(
            latent_dim,
            activation="relu",
        )(x)

        # =====================================================
        # DECODER
        # =====================================================

        x = latent

        for i in reversed(range(n_layers)):

            units = hp.Choice(
                f"decoder_units_{i}",
                values=[32, 64, 128, 256],
            )

            x = layers.Dense(
                units,
                activation="relu",
            )(x)

        outputs = layers.Dense(self.input_dim)(x)

        model = keras.Model(inputs, outputs)

        learning_rate = hp.Choice(
            "learning_rate",
            values=[1e-2, 1e-3, 1e-4],
        )

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss="mse",
        )

        return model


def train_autokeras_autoencoder(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 30,
    max_trials: int = 10,
):

    X_train = train_df.values.astype(np.float32)
    X_val = val_df.values.astype(np.float32)

    input_dim = X_train.shape[1]

    hypermodel = AutoencoderHyperModel(
        input_dim=input_dim
    )

    tuner = kt.RandomSearch(
        hypermodel,
        objective="val_loss",
        max_trials=max_trials,
        overwrite=True,
        directory="autokeras_search",
        project_name="anomaly_detection",
    )

    tuner.search(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=256,
        verbose=1,
    )

    best_model = tuner.get_best_models(1)[0]

    return best_model


def compute_reconstruction_errors_autokeras(
    model,
    df: pd.DataFrame,
):

    X = df.values.astype(np.float32)

    pred = model.predict(
        X,
        verbose=0,
    )

    errors = np.mean(
        (X - pred) ** 2,
        axis=1,
    )

    return errors


def score_dataset_autokeras(
    model,
    df: pd.DataFrame,
    threshold: float,
):

    errors = compute_reconstruction_errors_autokeras(
        model,
        df,
    )

    scores = pd.DataFrame(
        {
            "reconstruction_error": errors,
            "is_anomaly": errors > threshold,
        },
        index=df.index,
    )

    return scores